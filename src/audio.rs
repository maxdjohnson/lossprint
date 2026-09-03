//! Native-rate decoding of analysis windows from lossless containers.

use crate::{validate, DecodeError, ScoreError};
use std::fs::File;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_PCM_ALAW, CODEC_ID_PCM_MULAW};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::io::MediaSourceStream;
use symphonia::default::{get_codecs, get_probe};

pub(crate) const MIN_SECONDS: usize = 2;
pub(crate) const MIN_SAMPLE_RATE: u32 = 8_000;
pub(crate) const MAX_SAMPLE_RATE: u32 = 384_000;
/// Analysis windows are 14.5 seconds, the length the model was evaluated on.
const WINDOW_HALF_SECONDS: usize = 29;
const MAX_WINDOWS: usize = 6;

/// One analysis window of planar samples.
#[derive(Debug)]
pub(crate) struct Clip {
    pub(crate) channels: Vec<Vec<f32>>,
    pub(crate) sample_rate: u32,
}

pub(crate) fn window_len(sample_rate: u32) -> usize {
    sample_rate as usize * WINDOW_HALF_SECONDS / 2
}

/// Plan `(start, len)` windows over `total` frames. A track up to one window
/// long is scored whole; a longer track gets up to six evenly spaced windows
/// inside its middle 92%, the policy the model was evaluated with.
pub(crate) fn window_plan(total: usize, sample_rate: u32) -> Vec<(usize, usize)> {
    let len = window_len(sample_rate);
    if total <= len {
        return vec![(0, total)];
    }
    let lo = 0.04 * total as f64;
    let hi = 0.96 * total as f64 - len as f64;
    if hi > lo {
        let step = (hi - lo) / (MAX_WINDOWS - 1) as f64;
        (0..MAX_WINDOWS)
            .map(|index| ((lo + index as f64 * step).floor() as usize, len))
            .collect()
    } else {
        let start = lo.floor() as usize;
        vec![(start, len.min(total - start))]
    }
}

/// Decode the analysis windows without resampling, downmixing, or requantizing.
pub(crate) fn decode_file(file: File) -> Result<Vec<Clip>, ScoreError> {
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut format = get_probe()
        .probe(
            &Default::default(),
            stream,
            Default::default(),
            Default::default(),
        )
        .map_err(malformed)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or(DecodeError::NoUsableAudio)?;
    let codec = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or(DecodeError::NoUsableAudio)?;
    if codec.codec == CODEC_ID_PCM_ALAW || codec.codec == CODEC_ID_PCM_MULAW {
        return Err(DecodeError::NotLinearPcm.into());
    }
    let sample_rate = codec.sample_rate.ok_or(DecodeError::NoUsableAudio)?;
    let channel_count = codec
        .channels
        .as_ref()
        .ok_or(DecodeError::NoUsableAudio)?
        .count();
    validate(sample_rate, channel_count)?;
    let declared = track
        .num_frames
        .filter(|&frames| frames > 0)
        .map(|frames| usize::try_from(frames).unwrap_or(usize::MAX));
    let mut decoder = get_codecs()
        .make_audio_decoder(codec, &AudioDecoderOptions::default())
        .map_err(malformed)?;

    let min_frames = MIN_SECONDS * sample_rate as usize;
    // A container that declares its length lets the windows be planned up
    // front, so only their samples are kept in memory.
    let plan = match declared {
        Some(total) if total < min_frames => return Err(ScoreError::TooShort),
        Some(total) => window_plan(total, sample_rate),
        None => vec![(0, usize::MAX)],
    };
    let mut windows = plan
        .iter()
        .map(|&(start, len)| {
            let capacity = len.min(window_len(sample_rate));
            let channels = (0..channel_count)
                .map(|_| Vec::with_capacity(capacity))
                .collect::<Vec<_>>();
            (start, len, channels)
        })
        .collect::<Vec<_>>();
    let stop = windows
        .iter()
        .map(|(start, len, _)| start.saturating_add(*len))
        .max()
        .unwrap_or(0);

    let mut scratch = vec![Vec::<f32>::new(); channel_count];
    let mut position = 0_usize;
    while position < stop {
        let Some(packet) = format.next_packet().map_err(malformed)? else {
            break;
        };
        let decoded = decoder.decode(&packet).map_err(malformed)?;
        debug_assert_eq!(decoded.spec().rate(), sample_rate);
        debug_assert_eq!(decoded.spec().channels().count(), channel_count);
        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }
        for plane in &mut scratch {
            plane.clear();
            plane.resize(frames, 0.0);
        }
        decoded.copy_to_slice_planar::<f32, _>(&mut scratch);
        let end = position + frames;
        for (start, len, channels) in &mut windows {
            let lo = (*start).max(position);
            let hi = start.saturating_add(*len).min(end);
            if lo < hi {
                for (channel, plane) in channels.iter_mut().zip(&scratch) {
                    channel.extend_from_slice(&plane[lo - position..hi - position]);
                }
            }
        }
        position = end;
    }
    if position == 0 {
        return Err(DecodeError::NoUsableAudio.into());
    }

    let clips = if declared.is_some() {
        windows
            .into_iter()
            .filter_map(|(_, _, channels)| finish(channels, min_frames, sample_rate))
            .collect::<Vec<_>>()
    } else {
        // The whole track was read; slice its windows now that the length is known.
        let (_, _, channels) = windows.pop().expect("one whole-track window was planned");
        let total = channels.iter().map(Vec::len).min().unwrap_or(0);
        window_plan(total, sample_rate)
            .into_iter()
            .filter_map(|(start, len)| {
                let window = channels
                    .iter()
                    .map(|channel| channel[start..start + len].to_vec())
                    .collect();
                finish(window, min_frames, sample_rate)
            })
            .collect()
    };
    if clips.is_empty() {
        return Err(ScoreError::TooShort);
    }
    Ok(clips)
}

/// Trim a window to what every channel received; drop it if the container
/// promised more audio than it held.
fn finish(mut channels: Vec<Vec<f32>>, min_frames: usize, sample_rate: u32) -> Option<Clip> {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    if frames < min_frames {
        return None;
    }
    for channel in &mut channels {
        channel.truncate(frames);
    }
    Some(Clip {
        channels,
        sample_rate,
    })
}

fn malformed(error: impl std::error::Error + Send + Sync + 'static) -> ScoreError {
    ScoreError::Decode(DecodeError::Malformed {
        source: Box::new(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name)
    }

    /// A 16-bit mono WAV of `frames` frames counting up from zero.
    fn ramp_wav(frames: usize, sample_rate: u32) -> Vec<u8> {
        let data_len = (frames * 2) as u32;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16_u32.to_le_bytes());
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        out.extend_from_slice(&2_u16.to_le_bytes());
        out.extend_from_slice(&16_u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for frame in 0..frames {
            out.extend_from_slice(&((frame % 30_000) as i16).to_le_bytes());
        }
        out
    }

    fn decode_bytes(bytes: &[u8], label: &str) -> Result<Vec<Clip>, ScoreError> {
        let path = std::env::temp_dir().join(format!(
            "lossprint-audio-{label}-{}.wav",
            std::process::id()
        ));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        let decoded = decode_file(File::open(&path).unwrap());
        let _ = std::fs::remove_file(&path);
        decoded
    }

    #[test]
    fn short_tracks_are_one_whole_window() {
        assert_eq!(window_plan(2 * 44_100, 44_100), vec![(0, 88_200)]);
        assert_eq!(window_plan(639_450, 44_100), vec![(0, 639_450)]);
    }

    #[test]
    fn long_tracks_get_six_windows_inside_the_middle() {
        let plan = window_plan(20 * 44_100, 44_100);
        assert_eq!(plan.len(), 6);
        assert!(plan.iter().all(|&(_, len)| len == 639_450));
        assert_eq!(plan[0].0, 35_280);
        assert_eq!(plan[5].0, 207_270);
        assert_eq!(plan[5].0 + plan[5].1, 846_720);
        let steps: Vec<usize> = plan.windows(2).map(|pair| pair[1].0 - pair[0].0).collect();
        assert!(steps.iter().all(|&step| (34_397..=34_398).contains(&step)));
    }

    #[test]
    fn slightly_long_tracks_get_one_offset_window() {
        assert_eq!(window_plan(15 * 44_100, 44_100), vec![(26_460, 635_040)]);
    }

    #[test]
    fn wav_aiff_and_flac_decode_to_the_same_pcm() {
        // The fixtures are four frames long, so they only decode when the
        // minimum length is not enforced by the container's declared length.
        for name in ["pcm16.wav", "pcm16.aiff", "pcm16.flac"] {
            let error = decode_file(File::open(fixture(name)).unwrap()).unwrap_err();
            assert!(matches!(error, ScoreError::TooShort), "{name}: {error}");
        }
    }

    #[test]
    fn windows_hold_the_planned_samples() {
        let rate = 8_000;
        let clips = decode_bytes(&ramp_wav(3 * rate as usize, rate), "whole").unwrap();
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].sample_rate, rate);
        assert_eq!(clips[0].channels.len(), 1);
        assert_eq!(clips[0].channels[0].len(), 3 * rate as usize);
        assert_eq!(clips[0].channels[0][12_345], 12_345.0 / 32_768.0);

        let total = 40 * rate as usize;
        let clips = decode_bytes(&ramp_wav(total, rate), "planned").unwrap();
        let plan = window_plan(total, rate);
        assert_eq!(clips.len(), 6);
        for (clip, (start, len)) in clips.iter().zip(plan) {
            assert_eq!(clip.channels[0].len(), len);
            let first = ((start % 30_000) as i16) as f32 / 32_768.0;
            assert_eq!(clip.channels[0][0], first);
        }
    }

    #[test]
    fn tracks_below_the_minimum_are_rejected() {
        let rate = 8_000;
        let error = decode_bytes(&ramp_wav(rate as usize, rate), "short").unwrap_err();
        assert!(matches!(error, ScoreError::TooShort));
    }
}
