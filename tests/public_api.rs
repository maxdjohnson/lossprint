use lossprint::{
    audio::Error as AudioError, Encoder, Error as LossprintError, InitializationError, MediaSource,
    Scanner, ScoreError, TrackScore,
};
use std::io::Cursor;

fn assert_send_and_sync<T: Send + Sync>() {}

fn assert_public_error<T: std::error::Error + Send + Sync + 'static>() {}

fn assert_media_source<T: MediaSource + ?Sized>() {}

fn assert_audio_module_media_source<T: lossprint::audio::MediaSource + ?Sized>() {}

fn score_borrowed(
    scanner: &Scanner,
    source: &mut Cursor<Vec<u8>>,
) -> Result<TrackScore, ScoreError> {
    scanner.score(source)
}

fn score_borrowed_dyn(
    scanner: &Scanner,
    source: &mut dyn MediaSource,
) -> Result<TrackScore, ScoreError> {
    scanner.score(source)
}

fn score_boxed_dyn(
    scanner: &Scanner,
    source: Box<dyn MediaSource>,
) -> Result<TrackScore, ScoreError> {
    scanner.score(source)
}

fn inspect_score(score: &TrackScore) {
    fn assert_probability_iterator<I: Iterator<Item = (Encoder, f32)>>(_: I) {}

    let _: f32 = score.transcode_probability();
    let _: f32 = score.encoder_probability(Encoder::Mp3);
    let _: (Encoder, f32) = score.most_likely_encoder();
    assert_probability_iterator(score.encoder_probabilities());
}

fn classify_score_error(error: &ScoreError) -> &'static str {
    match error {
        ScoreError::Audio(AudioError::TooShort) => "too_short",
        ScoreError::Audio(_) => "audio",
        ScoreError::Inference(_) => "inference",
        _ => "future_score_error",
    }
}

fn classify_error(error: &LossprintError) -> &'static str {
    match error {
        LossprintError::Initialization(_) => "initialization",
        LossprintError::Score(error) => classify_score_error(error),
        _ => "future_error",
    }
}

#[test]
fn documented_public_types_and_call_shapes_compile() {
    assert_send_and_sync::<Scanner>();
    assert_public_error::<InitializationError>();
    assert_public_error::<AudioError>();
    assert_public_error::<ScoreError>();
    assert_public_error::<LossprintError>();

    assert_media_source::<Cursor<Vec<u8>>>();
    assert_media_source::<&mut Cursor<Vec<u8>>>();
    assert_media_source::<dyn MediaSource>();
    assert_media_source::<&mut dyn MediaSource>();
    assert_media_source::<Box<dyn MediaSource>>();
    assert_audio_module_media_source::<Cursor<Vec<u8>>>();

    let _: fn(&Scanner, &mut Cursor<Vec<u8>>) -> Result<TrackScore, ScoreError> = score_borrowed;
    let _: fn(&Scanner, &mut dyn MediaSource) -> Result<TrackScore, ScoreError> =
        score_borrowed_dyn;
    let _: fn(&Scanner, Box<dyn MediaSource>) -> Result<TrackScore, ScoreError> = score_boxed_dyn;
    let _: fn(&TrackScore) = inspect_score;
    let _: lossprint::Result<()> = Ok(());

    assert_eq!(Encoder::Mp3.as_str(), "mp3");
    assert_eq!(Encoder::Mp3.to_string(), "mp3");
}

#[test]
fn public_errors_can_be_classified_without_exhaustive_matches() {
    let score_error = ScoreError::from(AudioError::TooShort);
    assert_eq!(classify_score_error(&score_error), "too_short");
    let source =
        std::error::Error::source(&score_error).expect("audio errors remain in the source chain");
    assert!(matches!(
        source.downcast_ref::<AudioError>(),
        Some(AudioError::TooShort)
    ));

    let error = LossprintError::from(score_error);
    assert_eq!(classify_error(&error), "too_short");
}
