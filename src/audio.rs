//! Native-rate decoding for lossless containers.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_PCM_ALAW, CODEC_ID_PCM_MULAW};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::default::{get_codecs, get_probe};

const EXTENSIONS: [&str; 4] = ["aif", "aiff", "flac", "wav"];
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 384_000;

pub struct Clip {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            EXTENSIONS
                .iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate))
        })
}

/// Decode at most `max_seconds` without resampling, downmixing, or requantizing.
pub fn decode(path: &Path, max_seconds: usize) -> Result<Clip> {
    let file = File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    let mut format = get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .with_context(|| format!("could not recognize {}", path.display()))?;
    let track = format
        .default_track(TrackType::Audio)
        .context("audio file has no default track")?;
    let codec = track
        .codec_params
        .as_ref()
        .context("audio track has no codec parameters")?
        .audio()
        .context("default track is not audio")?;
    if codec.codec == CODEC_ID_PCM_ALAW || codec.codec == CODEC_ID_PCM_MULAW {
        bail!("A-law and mu-law audio are not lossless PCM")
    }
    let sample_rate = codec
        .sample_rate
        .context("audio track has no sample rate")?;
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        bail!(
            "{sample_rate} Hz sample rate is outside the supported {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz range"
        )
    }
    let channel_count = codec
        .channels
        .as_ref()
        .context("audio track has no channel layout")?
        .count();
    if !(1..=2).contains(&channel_count) {
        bail!("{channel_count} channels; only mono and stereo are supported")
    }
    let wanted_frames = max_seconds * sample_rate as usize;
    let mut channels = (0..channel_count)
        .map(|_| Vec::with_capacity(wanted_frames))
        .collect::<Vec<_>>();
    let mut decoder = get_codecs()
        .make_audio_decoder(codec, &AudioDecoderOptions::default())
        .context("could not create audio decoder")?;

    while let Some(packet) = format
        .next_packet()
        .context("could not read audio packet")?
    {
        let decoded = decoder
            .decode(&packet)
            .context("could not decode audio packet")?;
        debug_assert_eq!(decoded.spec().rate(), sample_rate);
        debug_assert_eq!(decoded.spec().channels().count(), channel_count);
        append(decoded, &mut channels);
        if channels[0].len() >= wanted_frames {
            break;
        }
    }

    if channels[0].is_empty() {
        bail!("audio track contains no decodable samples")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name)
    }

    fn decode_fixture(name: &str) -> Clip {
        decode(&fixture(name), 20).unwrap()
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
