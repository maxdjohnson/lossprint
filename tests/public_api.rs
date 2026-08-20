use lossprint::{DecodeError, Encoder, Scanner, ScoreError, TrackScore};
use std::fs::File;
use std::io::Write;

type Scored = Result<TrackScore, ScoreError>;

/// The categories a caller acts on, reachable without an exhaustive match.
///
/// The wildcard arms are mandatory, not decorative: both enums are
/// `#[non_exhaustive]`, so this will not compile without them.
fn classify(error: &ScoreError) -> &'static str {
    match error {
        ScoreError::Decode(DecodeError::NotLinearPcm) => "not_linear_pcm",
        ScoreError::Decode(DecodeError::NoUsableAudio) => "no_usable_audio",
        ScoreError::Decode(DecodeError::Malformed { .. }) => "malformed",
        ScoreError::Decode(_) => "decode",
        ScoreError::UnsupportedSampleRate { .. } => "unsupported_sample_rate",
        ScoreError::UnsupportedChannelCount { .. } => "unsupported_channel_count",
        ScoreError::TooShort => "too_short",
        ScoreError::Inference(_) => "inference",
        _ => "score",
    }
}

/// Interleaved 16-bit PCM noise, `frames` frames of `channels` channels.
fn noise(frames: usize, channels: usize) -> Vec<i16> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..frames * channels)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as i16 / 4
        })
        .collect()
}

/// Build an interleaved 16-bit PCM WAV in memory.
fn wav(interleaved: &[i16], channels: u16, sample_rate: u32) -> Vec<u8> {
    let data_len = (interleaved.len() * 2) as u32;
    let block_align = channels * 2;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in interleaved {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Score bytes through a real file, since that is the only entry point.
fn score(scanner: &Scanner, bytes: &[u8], label: &str) -> Scored {
    let path = std::env::temp_dir().join(format!("lossprint-{label}-{}.wav", std::process::id()));
    File::create(&path)
        .expect("temporary file is creatable")
        .write_all(bytes)
        .expect("temporary file is writable");
    let scored = scanner.score_file(File::open(&path).expect("temporary file is readable"));
    let _ = std::fs::remove_file(&path);
    scored
}

#[test]
fn a_scanner_is_shareable_and_scores_a_real_file() {
    fn assert_send_and_sync<T: Send + Sync>() {}
    assert_send_and_sync::<Scanner>();

    let scanner = Scanner::new().expect("the embedded model initializes");
    let bytes = wav(&noise(44_100, 2), 2, 44_100);

    let scored = score(&scanner, &bytes, "ok").expect("a real file scores");

    let transcode = scored.transcode_probability();
    assert!(
        (0.0..=1.0).contains(&transcode),
        "transcode probability is a probability: {transcode}"
    );
    let total: f32 = scored.encoder_probabilities().map(|(_, p)| p).sum();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "encoder probabilities are a distribution: {total}"
    );
    let (encoder, probability) = scored.most_likely_encoder();
    assert!(scored
        .encoder_probabilities()
        .all(|(_, other)| other <= probability + 1e-6));
    assert!(!encoder.as_str().is_empty());
}

#[test]
fn encoder_identifiers_round_trip_outside_the_crate() {
    let parsed: Encoder = "fdk_aac".parse().expect("identifier names an encoder");

    assert_eq!(parsed, Encoder::FdkAac);
    assert_eq!(parsed.as_str(), "fdk_aac");
    assert_eq!(parsed.to_string(), "fdk_aac");
    assert!("definitely_not_an_encoder".parse::<Encoder>().is_err());
}

#[test]
fn encoder_order_is_alphabetical() {
    let mut encoders = [Encoder::Mp3, Encoder::Opus, Encoder::Musepack];
    encoders.sort_unstable();
    assert_eq!(encoders, [Encoder::Mp3, Encoder::Musepack, Encoder::Opus]);
}

#[test]
fn decode_errors_stay_reachable_through_the_source_chain() {
    let error = ScoreError::from(DecodeError::NotLinearPcm);
    assert_eq!(classify(&error), "not_linear_pcm");

    let source =
        std::error::Error::source(&error).expect("decode errors remain in the source chain");
    assert!(matches!(
        source.downcast_ref::<DecodeError>(),
        Some(DecodeError::NotLinearPcm)
    ));
}

#[test]
fn unsupported_audio_is_reported_as_a_typed_error() {
    let scanner = Scanner::new().expect("the embedded model initializes");

    let rate = score(&scanner, &wav(&noise(2_000, 2), 2, 1_000), "rate");
    assert_eq!(classify(&rate.unwrap_err()), "unsupported_sample_rate");

    let channels = score(&scanner, &wav(&noise(44_100, 6), 6, 44_100), "channels");
    assert_eq!(
        classify(&channels.unwrap_err()),
        "unsupported_channel_count"
    );

    let short = score(&scanner, &wav(&noise(200, 2), 2, 44_100), "short");
    assert_eq!(classify(&short.unwrap_err()), "too_short");
}
