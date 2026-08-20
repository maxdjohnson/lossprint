use lossprint::{DecodeError, Encoder, MediaSource, Scanner, ScoreError, TrackScore};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

type Scored = Result<TrackScore, ScoreError>;

fn assert_media_source<T: MediaSource + ?Sized>() {}

/// A `Send` but `!Sync` reader, accepted because `MediaSource` requires only `Send`.
struct NotSync(Cursor<Vec<u8>>);

impl Read for NotSync {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buffer)
    }
}

impl Seek for NotSync {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.0.seek(position)
    }
}

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

#[test]
fn scanner_accepts_every_documented_source_shape() {
    fn assert_send_and_sync<T: Send + Sync>() {}

    assert_send_and_sync::<Scanner>();

    assert_media_source::<File>();
    assert_media_source::<BufReader<File>>();
    assert_media_source::<Cursor<Vec<u8>>>();
    assert_media_source::<&mut Cursor<Vec<u8>>>();
    assert_media_source::<dyn MediaSource>();
    assert_media_source::<&mut dyn MediaSource>();
    assert_media_source::<Box<dyn MediaSource>>();
    assert_media_source::<NotSync>();
    assert_media_source::<&mut NotSync>();
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
fn unsupported_audio_is_reported_without_decoding() {
    let scanner = Scanner::new().expect("the embedded model initializes");
    let silence = vec![0.0_f32; 44_100];

    let rate = scanner.score_samples(&[silence.as_slice()], 1_000);
    assert_eq!(classify(&rate.unwrap_err()), "unsupported_sample_rate");

    let channels = scanner.score_samples(&[], 44_100);
    assert_eq!(
        classify(&channels.unwrap_err()),
        "unsupported_channel_count"
    );

    let short = scanner.score_samples(&[&silence[..100]], 44_100);
    assert_eq!(classify(&short.unwrap_err()), "too_short");
}

/// Build a 16-bit PCM WAV in memory so the two entry points can be compared.
fn wav(samples: &[(i16, i16)], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 4) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&2_u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    out.extend_from_slice(&4_u16.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for (left, right) in samples {
        out.extend_from_slice(&left.to_le_bytes());
        out.extend_from_slice(&right.to_le_bytes());
    }
    out
}

#[test]
fn decoding_and_supplying_the_same_pcm_score_identically() {
    let scanner = Scanner::new().expect("the embedded model initializes");
    let rate = 44_100;
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let samples = (0..rate)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as i16 / 4, (state >> 32) as i16 / 4)
        })
        .collect::<Vec<_>>();

    let decoded: Scored = scanner.score(Cursor::new(wav(&samples, rate)));
    let decoded = decoded.expect("synthetic WAV scores");

    let left = samples
        .iter()
        .map(|(value, _)| f32::from(*value) / 32_768.0)
        .collect::<Vec<_>>();
    let right = samples
        .iter()
        .map(|(_, value)| f32::from(*value) / 32_768.0)
        .collect::<Vec<_>>();
    let supplied = scanner
        .score_samples(&[&left, &right], rate)
        .expect("planar samples score");

    assert_eq!(
        decoded.transcode_probability(),
        supplied.transcode_probability()
    );
    assert_eq!(
        decoded.most_likely_encoder(),
        supplied.most_likely_encoder()
    );
    for ((one, left), (two, right)) in decoded
        .encoder_probabilities()
        .zip(supplied.encoder_probabilities())
    {
        assert_eq!(one, two);
        assert_eq!(left, right);
    }
}
