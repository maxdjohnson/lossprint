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
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_FILE: AtomicUsize = AtomicUsize::new(0);
    const PCM16: [i16; 8] = [i16::MIN, i16::MAX, -16_384, 16_384, -1, 1, 0, 12_345];
    const FLAC: [u8; 113] = [
        0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1b,
        0x00, 0x00, 0x1b, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x04, 0x26, 0x38, 0xb8, 0x0f,
        0x4f, 0x5a, 0x90, 0x14, 0x66, 0x30, 0xcd, 0x05, 0xa4, 0x66, 0xa8, 0x4e, 0x84, 0x00, 0x00,
        0x28, 0x20, 0x00, 0x00, 0x00, 0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x20,
        0x6c, 0x69, 0x62, 0x46, 0x4c, 0x41, 0x43, 0x20, 0x31, 0x2e, 0x34, 0x2e, 0x33, 0x20, 0x32,
        0x30, 0x32, 0x33, 0x30, 0x36, 0x32, 0x33, 0x00, 0x00, 0x00, 0x00, 0xff, 0xf8, 0x69, 0x18,
        0x00, 0x03, 0xb6, 0x02, 0x80, 0x00, 0xc0, 0x00, 0xff, 0xff, 0x00, 0x00, 0x02, 0x7f, 0xff,
        0x40, 0x00, 0x00, 0x01, 0x30, 0x39, 0x3c, 0xe1,
    ];

    fn temp_path(extension: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "lossprint-native-audio-{}-{}.{}",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed),
            extension
        ))
    }

    fn float_wav(samples: &[f32], channels: u16, rate: u32) -> Vec<u8> {
        let data_len = std::mem::size_of_val(samples) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * u32::from(channels) * 4).to_le_bytes());
        bytes.extend_from_slice(&(channels * 4).to_le_bytes());
        bytes.extend_from_slice(&32_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn pcm16_wav() -> Vec<u8> {
        let mut bytes = float_wav(&[], 2, 44_100);
        bytes[20..22].copy_from_slice(&1_u16.to_le_bytes());
        bytes[28..32].copy_from_slice(&176_400_u32.to_le_bytes());
        bytes[32..34].copy_from_slice(&4_u16.to_le_bytes());
        bytes[34..36].copy_from_slice(&16_u16.to_le_bytes());
        let data_len = (PCM16.len() * 2) as u32;
        bytes[4..8].copy_from_slice(&(36 + data_len).to_le_bytes());
        bytes[40..44].copy_from_slice(&data_len.to_le_bytes());
        bytes.extend(PCM16.iter().flat_map(|sample| sample.to_le_bytes()));
        bytes
    }

    fn pcm16_aiff() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"FORM");
        bytes.extend_from_slice(&0_u32.to_be_bytes());
        bytes.extend_from_slice(b"AIFFCOMM");
        bytes.extend_from_slice(&18_u32.to_be_bytes());
        bytes.extend_from_slice(&2_u16.to_be_bytes());
        bytes.extend_from_slice(&4_u32.to_be_bytes());
        bytes.extend_from_slice(&16_u16.to_be_bytes());
        bytes.extend_from_slice(&[0x40, 0x0e, 0xac, 0x44, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(b"SSND");
        bytes.extend_from_slice(&(8 + PCM16.len() as u32 * 2).to_be_bytes());
        bytes.extend_from_slice(&[0; 8]);
        bytes.extend(PCM16.iter().flat_map(|sample| sample.to_be_bytes()));
        let form_len = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&form_len.to_be_bytes());
        bytes
    }

    fn decode_bytes(extension: &str, bytes: &[u8]) -> Clip {
        let path = temp_path(extension);
        fs::write(&path, bytes).unwrap();
        let clip = decode(&path, 20).unwrap();
        fs::remove_file(path).unwrap();
        clip
    }

    #[test]
    fn wav_aiff_and_flac_decode_to_the_same_pcm() {
        let wav = decode_bytes("wav", &pcm16_wav());
        let aiff = decode_bytes("aiff", &pcm16_aiff());
        let flac = decode_bytes("flac", &FLAC);
        assert_eq!(wav.sample_rate, 44_100);
        assert_eq!(wav.channels, aiff.channels);
        assert_eq!(wav.channels, flac.channels);
    }

    #[test]
    fn preserves_native_float_samples_and_layout() {
        let samples = [1.25_f32, -1.5, 0.25, -0.75];
        let clip = decode_bytes("wav", &float_wav(&samples, 2, 48_000));

        assert_eq!(clip.sample_rate, 48_000);
        assert_eq!(clip.channels, vec![vec![1.25, 0.25], vec![-1.5, -0.75]]);
    }

    #[test]
    fn preserves_mono_instead_of_duplicating_it() {
        let samples = [0.5_f32, -0.25];
        let clip = decode_bytes("wav", &float_wav(&samples, 1, 32_000));

        assert_eq!(clip.sample_rate, 32_000);
        assert_eq!(clip.channels, vec![samples.to_vec()]);
    }
}
