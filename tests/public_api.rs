use lossprint::{
    DecodeError, Encoder, InitializationError, MediaSource, ModelError, Scanner, ScoreError,
    TrackScore, UnknownEncoder,
};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

type Scored = Result<TrackScore, ScoreError>;

fn assert_send_and_sync<T: Send + Sync>() {}

fn assert_debug<T: std::fmt::Debug>() {}

fn assert_public_error<T: std::error::Error + Send + Sync + 'static>() {}

fn assert_media_source<T: MediaSource + ?Sized>() {}

/// A `Send` but `!Sync` reader: accepted because `MediaSource` requires only `Send`.
struct NotSync {
    inner: Cursor<Vec<u8>>,
    _reads: Cell<usize>,
}

impl Read for NotSync {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buffer)
    }
}

impl Seek for NotSync {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

fn score_borrowed(scanner: &Scanner, source: &mut Cursor<Vec<u8>>) -> Scored {
    scanner.score(source)
}

fn score_borrowed_dyn(scanner: &Scanner, source: &mut dyn MediaSource) -> Scored {
    scanner.score(source)
}

fn score_boxed_dyn(scanner: &Scanner, source: Box<dyn MediaSource>) -> Scored {
    scanner.score(source)
}

fn score_planar_samples(scanner: &Scanner, channels: &[&[f32]], sample_rate: u32) -> Scored {
    scanner.score_samples(channels, sample_rate)
}

fn score_owned_samples(scanner: &Scanner, channels: &[Vec<f32>]) -> Scored {
    let borrowed = channels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    scanner.score_samples(&borrowed, 44_100)
}

fn inspect_score(score: &TrackScore) {
    fn assert_probability_iterator<I: Iterator<Item = (Encoder, f32)>>(_: I) {}

    let _: f32 = score.transcode_probability();
    let _: f32 = score.encoder_probability(Encoder::Mp3);
    let _: (Encoder, f32) = score.most_likely_encoder();
    assert_probability_iterator(score.encoder_probabilities());
}

/// The categories a caller acts on, reachable without an exhaustive match.
fn classify_score_error(error: &ScoreError) -> &'static str {
    match error {
        ScoreError::Decode(DecodeError::NotLinearPcm) => "not_linear_pcm",
        ScoreError::Decode(DecodeError::NoUsableAudio) => "no_usable_audio",
        ScoreError::Decode(DecodeError::Malformed { .. }) => "malformed",
        ScoreError::Decode(_) => "future_decode_error",
        ScoreError::UnsupportedSampleRate { rate, .. } => {
            assert!(*rate > 0);
            "unsupported_sample_rate"
        }
        ScoreError::UnsupportedChannelCount { channels, .. } => {
            assert!(*channels != 1);
            "unsupported_channel_count"
        }
        ScoreError::TooShort => "too_short",
        ScoreError::Inference(_) => "inference",
        _ => "future_score_error",
    }
}

#[test]
fn documented_public_types_and_call_shapes_compile() {
    assert_send_and_sync::<Scanner>();
    assert_debug::<Scanner>();
    assert_debug::<TrackScore>();
    assert_public_error::<InitializationError>();
    assert_public_error::<ModelError>();
    assert_public_error::<DecodeError>();
    assert_public_error::<ScoreError>();
    assert_public_error::<UnknownEncoder>();

    assert_media_source::<File>();
    assert_media_source::<BufReader<File>>();
    assert_media_source::<Cursor<Vec<u8>>>();
    assert_media_source::<&mut Cursor<Vec<u8>>>();
    assert_media_source::<dyn MediaSource>();
    assert_media_source::<&mut dyn MediaSource>();
    assert_media_source::<Box<dyn MediaSource>>();
    // Send but not Sync: rejected before, accepted now.
    assert_media_source::<NotSync>();
    assert_media_source::<&mut NotSync>();

    let _: fn(&Scanner, &mut Cursor<Vec<u8>>) -> Scored = score_borrowed;
    let _: fn(&Scanner, &mut dyn MediaSource) -> Scored = score_borrowed_dyn;
    let _: fn(&Scanner, Box<dyn MediaSource>) -> Scored = score_boxed_dyn;
    let _: fn(&Scanner, &[&[f32]], u32) -> Scored = score_planar_samples;
    let _: fn(&Scanner, &[Vec<f32>]) -> Scored = score_owned_samples;
    let _: fn(&TrackScore) = inspect_score;

    assert_eq!(Encoder::Mp3.as_str(), "mp3");
    assert_eq!(Encoder::Mp3.to_string(), "mp3");
}

#[test]
fn encoders_work_as_collection_keys_and_round_trip_through_text() {
    let parsed: Encoder = "fdk_aac".parse().expect("identifier names an encoder");
    assert_eq!(parsed, Encoder::FdkAac);
    assert_eq!(parsed.as_str().parse::<Encoder>(), Ok(Encoder::FdkAac));
    // `UnknownEncoder` is deliberately not constructible outside the crate.
    assert!("definitely_not_an_encoder".parse::<Encoder>().is_err());

    let mut hashed = HashMap::new();
    hashed.insert(Encoder::Opus, 0.5_f32);
    assert_eq!(hashed.get(&Encoder::Opus), Some(&0.5));

    let mut sorted = BTreeMap::new();
    sorted.insert(Encoder::Musepack, 0.1_f32);
    sorted.insert(Encoder::Mp3, 0.9_f32);
    assert_eq!(
        sorted.keys().copied().collect::<Vec<_>>(),
        vec![Encoder::Mp3, Encoder::Musepack]
    );
}

#[test]
fn public_errors_can_be_classified_without_exhaustive_matches() {
    let decode_error = ScoreError::from(DecodeError::NotLinearPcm);
    assert_eq!(classify_score_error(&decode_error), "not_linear_pcm");
    let source =
        std::error::Error::source(&decode_error).expect("decode errors remain in the source chain");
    assert!(matches!(
        source.downcast_ref::<DecodeError>(),
        Some(DecodeError::NotLinearPcm)
    ));

    assert_eq!(classify_score_error(&ScoreError::TooShort), "too_short");
}

#[test]
fn unsupported_audio_is_reported_without_decoding() {
    let scanner = match Scanner::new() {
        Ok(scanner) => scanner,
        // The embedded model is a stub on docs.rs builds.
        Err(_) => return,
    };
    let silence = vec![0.0_f32; 44_100];

    let rate = scanner.score_samples(&[silence.as_slice()], 1_000);
    assert_eq!(
        classify_score_error(&rate.unwrap_err()),
        "unsupported_sample_rate"
    );

    let channels = scanner.score_samples(&[], 44_100);
    assert_eq!(
        classify_score_error(&channels.unwrap_err()),
        "unsupported_channel_count"
    );

    let short = scanner.score_samples(&[&silence[..100]], 44_100);
    assert_eq!(classify_score_error(&short.unwrap_err()), "too_short");
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
    let scanner = match Scanner::new() {
        Ok(scanner) => scanner,
        // The embedded model is a stub on docs.rs builds.
        Err(_) => return,
    };

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

    let decoded = scanner
        .score(Cursor::new(wav(&samples, rate)))
        .expect("synthetic WAV scores");

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
    for ((one, left), (two, right)) in decoded
        .encoder_probabilities()
        .zip(supplied.encoder_probabilities())
    {
        assert_eq!(one, two);
        assert_eq!(left, right);
    }
}
