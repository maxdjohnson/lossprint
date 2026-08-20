//! Native-rate decoding for lossless containers.

use crate::{validate, DecodeError, ScoreError};
use std::fs::File;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_PCM_ALAW, CODEC_ID_PCM_MULAW};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::io::MediaSourceStream;
use symphonia::default::{get_codecs, get_probe};

pub(crate) const MAX_SECONDS: usize = 20;
pub(crate) const MIN_SAMPLE_RATE: u32 = 8_000;
pub(crate) const MAX_SAMPLE_RATE: u32 = 384_000;

pub(crate) struct Clip {
    pub(crate) channels: Vec<Vec<f32>>,
    pub(crate) sample_rate: u32,
}

/// Decode at most 20 seconds without resampling, downmixing, or requantizing.
pub(crate) fn decode_file(file: File) -> Result<Clip, ScoreError> {
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut format = get_probe()
        .probe(
            &Default::default(),
            stream,
            Default::default(),
            Default::default(),
        )
        .map_err(malformed)?;
    let codec = format
        .default_track(TrackType::Audio)
        .and_then(|track| track.codec_params.as_ref())
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

    let wanted_frames = MAX_SECONDS * sample_rate as usize;
    let mut channels = (0..channel_count)
        .map(|_| Vec::with_capacity(wanted_frames))
        .collect::<Vec<_>>();
    let mut decoder = get_codecs()
        .make_audio_decoder(codec, &AudioDecoderOptions::default())
        .map_err(malformed)?;

    while let Some(packet) = format.next_packet().map_err(malformed)? {
        let decoded = decoder.decode(&packet).map_err(malformed)?;
        debug_assert_eq!(decoded.spec().rate(), sample_rate);
        debug_assert_eq!(decoded.spec().channels().count(), channel_count);
        append(decoded, &mut channels);
        if channels[0].len() >= wanted_frames {
            break;
        }
    }

    if channels[0].is_empty() {
        return Err(DecodeError::NoUsableAudio.into());
    }
    for channel in &mut channels {
        channel.truncate(wanted_frames);
    }
    Ok(Clip {
        channels,
        sample_rate,
    })
}

fn append(decoded: GenericAudioBufferRef<'_>, output: &mut [Vec<f32>]) {
    let frames = decoded.frames();
    let mut tails = output
        .iter_mut()
        .map(|plane| {
            let start = plane.len();
            plane.resize(start + frames, 0.0);
            &mut plane[start..]
        })
        .collect::<Vec<_>>();
    decoded.copy_to_slice_planar::<f32, _>(&mut tails);
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
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name)
    }

    fn decode_fixture(name: &str) -> Clip {
        decode_file(File::open(fixture(name)).unwrap()).unwrap()
    }

    #[test]
    fn wav_aiff_and_flac_decode_to_the_same_pcm() {
        let wav = decode_fixture("pcm16.wav");
        let aiff = decode_fixture("pcm16.aiff");
        let flac = decode_fixture("pcm16.flac");
        assert_eq!(wav.sample_rate, 44_100);
        assert_eq!(wav.channels, aiff.channels);
        assert_eq!(wav.channels, flac.channels);
        assert_eq!(
            wav.channels,
            vec![
                vec![-1.0, -0.5, -1.0 / 32_768.0, 0.0],
                vec![
                    32_767.0 / 32_768.0,
                    0.5,
                    1.0 / 32_768.0,
                    12_345.0 / 32_768.0,
                ],
            ]
        );
    }

    #[test]
    fn preserves_native_float_samples_and_layout() {
        let clip = decode_fixture("float32-stereo.wav");

        assert_eq!(clip.sample_rate, 48_000);
        assert_eq!(clip.channels, vec![vec![1.25, 0.25], vec![-1.5, -0.75]]);
    }

    #[test]
    fn preserves_mono_instead_of_duplicating_it() {
        let clip = decode_fixture("float32-mono.wav");

        assert_eq!(clip.sample_rate, 32_000);
        assert_eq!(clip.channels, vec![vec![0.5, -0.25]]);
    }
}
