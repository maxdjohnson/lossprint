#![warn(missing_docs)]
//! Detect lossy transcodes hiding in lossless audio files.
//!
//! Reuse a [`Scanner`] when scoring multiple tracks so that its model and
//! spectrogram transforms stay initialized.
//!
//! ```no_run
//! use lossprint::{Encoder, Scanner};
//! use std::fs::File;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let scanner = Scanner::new()?;
//! let score = scanner.score(File::open("track.flac")?)?;
//!
//! println!("P(transcode) = {:.3}", score.transcode_probability());
//! println!(
//!     "P(mp3 | transcode) = {:.3}",
//!     score.encoder_probability(Encoder::Mp3)
//! );
//! if score.transcode_probability() >= 0.5 {
//!     println!("transcode");
//! }
//! # Ok(())
//! # }
//! ```

pub mod audio;
mod model;
mod spectrogram;

use model::{Model, WindowScore, ENCODER_COUNT};
use spectrogram::TransformCache;
use std::fmt;

pub use audio::MediaSource;

/// A convenience result for code that both constructs and uses a scanner.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure while initializing a [`Scanner`].
///
/// The underlying model-runtime error is available through
/// [`std::error::Error::source`] without exposing the runtime as part of this
/// crate's public API.
#[derive(Debug, thiserror::Error)]
#[error("model initialization failed: {source}")]
pub struct InitializationError {
    #[source]
    source: Box<dyn std::error::Error + Send + Sync + 'static>,
}

impl InitializationError {
    fn new(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self {
            source: Box::new(error),
        }
    }
}

/// A failure while scoring an audio media source.
///
/// Backend-specific errors remain available through [`std::error::Error::source`]
/// without becoming part of this crate's public API.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScoreError {
    /// The source could not be decoded or prepared.
    #[error("{0}")]
    Audio(#[from] audio::Error),
    /// The model could not score prepared audio.
    #[error("model inference failed: {0}")]
    Inference(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl ScoreError {
    fn inference(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Inference(Box::new(error))
    }
}

/// A convenience error for code that both constructs and uses a scanner.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Scanner initialization failed.
    #[error(transparent)]
    Initialization(#[from] InitializationError),
    /// Scoring failed.
    #[error(transparent)]
    Score(#[from] ScoreError),
}

const ENCODERS: [Encoder; ENCODER_COUNT] = [
    Encoder::Mp3,
    Encoder::FfmpegAac,
    Encoder::AacAt,
    Encoder::FdkAac,
    Encoder::Vorbis,
    Encoder::Opus,
    Encoder::Mp2,
    Encoder::Wma,
    Encoder::Musepack,
];

/// A source encoder class predicted by the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Encoder {
    /// MP3.
    Mp3,
    /// FFmpeg's native AAC encoder.
    FfmpegAac,
    /// Apple AudioToolbox AAC.
    AacAt,
    /// Fraunhofer FDK AAC.
    FdkAac,
    /// Vorbis.
    Vorbis,
    /// Opus.
    Opus,
    /// MP2.
    Mp2,
    /// Windows Media Audio.
    Wma,
    /// Musepack.
    Musepack,
}

impl Encoder {
    /// Return this encoder's stable machine-readable identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::FfmpegAac => "ffmpeg_aac",
            Self::AacAt => "aac_at",
            Self::FdkAac => "fdk_aac",
            Self::Vorbis => "vorbis",
            Self::Opus => "opus",
            Self::Mp2 => "mp2",
            Self::Wma => "wma",
            Self::Musepack => "musepack",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Mp3 => 0,
            Self::FfmpegAac => 1,
            Self::AacAt => 2,
            Self::FdkAac => 3,
            Self::Vorbis => 4,
            Self::Opus => 5,
            Self::Mp2 => 6,
            Self::Wma => 7,
            Self::Musepack => 8,
        }
    }
}

impl fmt::Display for Encoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Model probabilities pooled across a track's analysis windows.
#[derive(Clone, Debug)]
pub struct TrackScore {
    transcode_probability: f32,
    encoder_probabilities: [f32; ENCODER_COUNT],
}

impl TrackScore {
    /// Return the probability that this track is a lossy transcode.
    pub const fn transcode_probability(&self) -> f32 {
        self.transcode_probability
    }

    /// Return the probability for one encoder class.
    ///
    /// This probability is conditional on the track being a transcode.
    pub const fn encoder_probability(&self, encoder: Encoder) -> f32 {
        self.encoder_probabilities[encoder.index()]
    }

    /// Iterate over every encoder and conditional probability.
    ///
    /// The iteration order is unspecified and may change with the model.
    pub fn encoder_probabilities(&self) -> impl Iterator<Item = (Encoder, f32)> + '_ {
        ENCODERS
            .into_iter()
            .zip(self.encoder_probabilities.iter().copied())
    }

    /// Return the most likely source encoder and its conditional probability.
    pub fn most_likely_encoder(&self) -> (Encoder, f32) {
        self.encoder_probabilities()
            .reduce(|best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
            .expect("the fixed model has encoder classes")
    }
}

/// A reusable audio-transcode scanner.
pub struct Scanner {
    model: Model,
    transforms: TransformCache,
}

impl Scanner {
    /// Construct a scanner using the downloaded-and-embedded model.
    pub fn new() -> std::result::Result<Self, InitializationError> {
        let model = Model::new().map_err(InitializationError::new)?;
        Ok(Self {
            model,
            transforms: TransformCache::default(),
        })
    }

    /// Score one WAV, AIFF, or FLAC media source.
    pub fn score<S: MediaSource>(&self, source: S) -> std::result::Result<TrackScore, ScoreError> {
        let windows = prepare(source, &self.transforms)?;
        self.score_windows(&windows)
    }

    fn score_windows(&self, windows: &[f32]) -> std::result::Result<TrackScore, ScoreError> {
        let scores = self.model.run(windows).map_err(ScoreError::inference)?;
        Ok(TrackScore::pool(&scores))
    }
}

fn prepare<S: MediaSource>(
    source: S,
    transforms: &TransformCache,
) -> std::result::Result<Vec<f32>, audio::Error> {
    let clip = audio::decode(source)?;
    let crop_len = clip.sample_rate as usize / 2;
    let offsets = spectrogram::window_offsets(clip.channels[0].len(), crop_len);
    if offsets.is_empty() {
        return Err(audio::Error::TooShort);
    }
    let transform = transforms.get(clip.sample_rate);
    Ok(transform.write_windows(&clip.channels, &offsets))
}

impl TrackScore {
    fn pool(scores: &[WindowScore]) -> Self {
        debug_assert!(!scores.is_empty());
        let mut transcode_logit_sum = 0.0;
        let mut encoder_log_sums = [0.0; ENCODER_COUNT];

        for score in scores {
            let probability = score.transcode_probability.clamp(1e-6, 1.0 - 1e-6);
            transcode_logit_sum += probability.ln() - (-probability).ln_1p();
            for (sum, probability) in encoder_log_sums.iter_mut().zip(score.encoder_probabilities) {
                *sum += probability.clamp(1e-6, 1.0).ln();
            }
        }

        let windows = scores.len() as f32;
        let mean_logit = transcode_logit_sum / windows;
        let mean_logs = encoder_log_sums.map(|sum| sum / windows);
        let max_log = mean_logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let geometric = mean_logs.map(|value| (value - max_log).exp());
        let encoder_total: f32 = geometric.iter().sum();
        Self {
            transcode_probability: 1.0 / (1.0 + (-mean_logit).exp()),
            encoder_probabilities: geometric.map(|probability| probability / encoder_total),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("backend detail")]
    struct BackendError;

    #[test]
    fn public_errors_preserve_backend_sources() {
        fn assert_public_error<T: std::error::Error + Send + Sync + 'static>() {}
        assert_public_error::<InitializationError>();
        assert_public_error::<audio::Error>();
        assert_public_error::<ScoreError>();
        assert_public_error::<Error>();

        let initialization = InitializationError::new(BackendError);
        assert_eq!(
            initialization.to_string(),
            "model initialization failed: backend detail"
        );
        assert_eq!(
            std::error::Error::source(&initialization)
                .unwrap()
                .to_string(),
            "backend detail"
        );

        let audio = ScoreError::from(audio::Error::Probe(Box::new(BackendError)));
        assert!(matches!(&audio, ScoreError::Audio(_)));
        assert_eq!(
            audio.to_string(),
            "could not recognize audio format: backend detail"
        );
        let audio_source = std::error::Error::source(&audio).unwrap();
        assert_eq!(
            audio_source.to_string(),
            "could not recognize audio format: backend detail"
        );
        assert_eq!(audio_source.source().unwrap().to_string(), "backend detail");

        let inference = ScoreError::inference(BackendError);
        assert!(matches!(&inference, ScoreError::Inference(_)));
        assert_eq!(
            inference.to_string(),
            "model inference failed: backend detail"
        );
        assert_eq!(
            std::error::Error::source(&inference).unwrap().to_string(),
            "backend detail"
        );
    }

    #[test]
    fn model_outputs_map_to_encoders() {
        let score = TrackScore::pool(&[WindowScore {
            transcode_probability: 0.75,
            encoder_probabilities: [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.28],
        }]);

        assert!((score.transcode_probability() - 0.75).abs() < 1e-6);
        let expected = [
            (Encoder::Mp3, 0.02),
            (Encoder::FfmpegAac, 0.04),
            (Encoder::AacAt, 0.06),
            (Encoder::FdkAac, 0.08),
            (Encoder::Vorbis, 0.1),
            (Encoder::Opus, 0.12),
            (Encoder::Mp2, 0.14),
            (Encoder::Wma, 0.16),
            (Encoder::Musepack, 0.28),
        ];
        for ((actual_encoder, actual_probability), (expected_encoder, expected_probability)) in
            score.encoder_probabilities().zip(expected)
        {
            assert_eq!(actual_encoder, expected_encoder);
            assert!((actual_probability - expected_probability).abs() < 1e-6);
        }
    }

    #[test]
    fn encoder_identifiers_are_stable() {
        assert_eq!(
            ENCODERS.map(Encoder::as_str),
            [
                "mp3",
                "ffmpeg_aac",
                "aac_at",
                "fdk_aac",
                "vorbis",
                "opus",
                "mp2",
                "wma",
                "musepack",
            ]
        );
    }

    #[test]
    fn track_score_uses_geometric_probability_pooling() {
        let score = TrackScore::pool(&[
            WindowScore {
                transcode_probability: 0.2,
                encoder_probabilities: [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            WindowScore {
                transcode_probability: 0.8,
                encoder_probabilities: [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ]);

        assert!((score.transcode_probability() - 0.5).abs() < 1e-6);
        let mp3 = 0.36_f32.sqrt();
        let aac = 0.06_f32.sqrt();
        let floor = 1e-6_f32;
        let total = mp3 + aac + 7.0 * floor;
        assert!((score.encoder_probability(Encoder::Mp3) - mp3 / total).abs() < 1e-6);
        assert!((score.encoder_probability(Encoder::FfmpegAac) - aac / total).abs() < 1e-6);
        assert!((score.encoder_probability(Encoder::AacAt) - floor / total).abs() < 1e-8);
        assert!((score.encoder_probability(Encoder::Musepack) - floor / total).abs() < 1e-8);
    }

    #[test]
    fn scanner_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        assert_send_and_sync::<Scanner>();
    }

    #[test]
    fn common_seekable_readers_are_media_sources() {
        fn assert_media_source<T: MediaSource>() {}
        assert_media_source::<std::fs::File>();
        assert_media_source::<std::io::Cursor<Vec<u8>>>();
        assert_media_source::<std::io::BufReader<std::fs::File>>();
    }
}
