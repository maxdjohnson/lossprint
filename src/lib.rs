#![warn(missing_docs)]
//! Detect lossy transcodes hiding in lossless audio files.
//!
//! Reuse a [`Scanner`] when scoring multiple tracks so that its model and
//! spectrogram transforms stay initialized.
//!
//! ```no_run
//! use lossprint::{Codec, Scanner};
//!
//! # fn main() -> lossprint::Result<()> {
//! let scanner = Scanner::new()?;
//! let score = scanner.score_file("track.flac")?;
//!
//! println!("P(transcode) = {:.3}", score.transcode_probability());
//! println!(
//!     "P(mp3 | transcode) = {:.3}",
//!     score.codec_probability(Codec::Mp3)
//! );
//! if score.transcode_probability() >= 0.5 {
//!     println!("transcode");
//! }
//! # Ok(())
//! # }
//! ```

mod audio;
mod model;
mod spectrogram;

use model::{Model, WindowScore, CODEC_COUNT};
use spectrogram::TransformCache;
use std::fmt;
use std::path::Path;

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

/// A failure while scoring an audio file.
///
/// The variants distinguish errors tied to one input from failures in the
/// shared model runtime. Backend-specific errors remain available through
/// [`std::error::Error::source`] without becoming part of this crate's public
/// API.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScoreError {
    /// The input file could not be decoded or prepared.
    #[error("{0}")]
    Audio(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    /// The model could not score prepared audio.
    #[error("model inference failed: {0}")]
    Inference(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl ScoreError {
    fn audio(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Audio(Box::new(error))
    }

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

const CODECS: [Codec; CODEC_COUNT] = [
    Codec::Mp3,
    Codec::Aac,
    Codec::AacAt,
    Codec::FdkAac,
    Codec::Vorbis,
    Codec::Opus,
    Codec::Mp2,
    Codec::Wma,
    Codec::Musepack,
];

/// A source codec or encoder class predicted by the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    /// MP3.
    Mp3,
    /// Generic AAC.
    Aac,
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

impl Codec {
    const fn index(self) -> usize {
        match self {
            Self::Mp3 => 0,
            Self::Aac => 1,
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

impl fmt::Display for Codec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::AacAt => "aac_at",
            Self::FdkAac => "fdk_aac",
            Self::Vorbis => "vorbis",
            Self::Opus => "opus",
            Self::Mp2 => "mp2",
            Self::Wma => "wma",
            Self::Musepack => "musepack",
        })
    }
}

/// Model probabilities pooled across a track's analysis windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackScore {
    transcode_probability: f32,
    codec_probabilities: [f32; CODEC_COUNT],
}

impl TrackScore {
    /// Return the probability that this track is a lossy transcode.
    pub const fn transcode_probability(&self) -> f32 {
        self.transcode_probability
    }

    /// Return the probability for one codec or encoder class.
    ///
    /// This probability is conditional on the track being a transcode.
    pub const fn codec_probability(&self, codec: Codec) -> f32 {
        self.codec_probabilities[codec.index()]
    }

    /// Iterate over every codec and conditional probability in model order.
    pub fn codec_probabilities(&self) -> impl Iterator<Item = (Codec, f32)> + '_ {
        CODECS
            .into_iter()
            .zip(self.codec_probabilities.iter().copied())
    }

    /// Return the most likely source codec and its conditional probability.
    pub fn most_likely_codec(&self) -> (Codec, f32) {
        self.codec_probabilities()
            .reduce(|best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
            .expect("the fixed model has codec classes")
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

    /// Score one WAV, AIFF, or FLAC file.
    pub fn score_file(
        &self,
        path: impl AsRef<Path>,
    ) -> std::result::Result<TrackScore, ScoreError> {
        let windows = prepare(path.as_ref(), &self.transforms).map_err(ScoreError::audio)?;
        self.score_windows(&windows)
    }

    /// Score files serially in input order.
    ///
    /// Each track's windows are scored together in one model call. Every input
    /// produces one same-index result containing its [`ScoreError`], if any.
    pub fn score_files<P>(&self, files: &[P]) -> Vec<std::result::Result<TrackScore, ScoreError>>
    where
        P: AsRef<Path>,
    {
        files
            .iter()
            .map(|path| self.score_file(path.as_ref()))
            .collect()
    }

    fn score_windows(&self, windows: &[f32]) -> std::result::Result<TrackScore, ScoreError> {
        let scores = self.model.run(windows).map_err(ScoreError::inference)?;
        Ok(TrackScore::pool(&scores))
    }
}

fn prepare(
    path: &Path,
    transforms: &TransformCache,
) -> std::result::Result<Vec<f32>, audio::Error> {
    let clip = audio::decode(path)?;
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
        let mut codec_log_sums = [0.0; CODEC_COUNT];

        for score in scores {
            let probability = score.transcode_probability.clamp(1e-6, 1.0 - 1e-6);
            transcode_logit_sum += probability.ln() - (-probability).ln_1p();
            for (sum, probability) in codec_log_sums.iter_mut().zip(score.codec_probabilities) {
                *sum += probability.clamp(1e-6, 1.0).ln();
            }
        }

        let windows = scores.len() as f32;
        let mean_logit = transcode_logit_sum / windows;
        let mean_logs = codec_log_sums.map(|sum| sum / windows);
        let max_log = mean_logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let geometric = mean_logs.map(|value| (value - max_log).exp());
        let codec_total: f32 = geometric.iter().sum();
        Self {
            transcode_probability: 1.0 / (1.0 + (-mean_logit).exp()),
            codec_probabilities: geometric.map(|probability| probability / codec_total),
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

        let audio = ScoreError::audio(BackendError);
        assert!(matches!(&audio, ScoreError::Audio(_)));
        assert_eq!(audio.to_string(), "backend detail");
        assert_eq!(
            std::error::Error::source(&audio).unwrap().to_string(),
            "backend detail"
        );

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
    fn model_outputs_map_to_codecs() {
        let score = TrackScore::pool(&[WindowScore {
            transcode_probability: 0.75,
            codec_probabilities: [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.28],
        }]);

        assert!((score.transcode_probability() - 0.75).abs() < 1e-6);
        let expected = [
            (Codec::Mp3, 0.02),
            (Codec::Aac, 0.04),
            (Codec::AacAt, 0.06),
            (Codec::FdkAac, 0.08),
            (Codec::Vorbis, 0.1),
            (Codec::Opus, 0.12),
            (Codec::Mp2, 0.14),
            (Codec::Wma, 0.16),
            (Codec::Musepack, 0.28),
        ];
        for ((actual_codec, actual_probability), (expected_codec, expected_probability)) in
            score.codec_probabilities().zip(expected)
        {
            assert_eq!(actual_codec, expected_codec);
            assert!((actual_probability - expected_probability).abs() < 1e-6);
        }
    }

    #[test]
    fn track_score_uses_geometric_probability_pooling() {
        let score = TrackScore::pool(&[
            WindowScore {
                transcode_probability: 0.2,
                codec_probabilities: [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            WindowScore {
                transcode_probability: 0.8,
                codec_probabilities: [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ]);

        assert!((score.transcode_probability() - 0.5).abs() < 1e-6);
        let mp3 = 0.36_f32.sqrt();
        let aac = 0.06_f32.sqrt();
        let floor = 1e-6_f32;
        let total = mp3 + aac + 7.0 * floor;
        assert!((score.codec_probability(Codec::Mp3) - mp3 / total).abs() < 1e-6);
        assert!((score.codec_probability(Codec::Aac) - aac / total).abs() < 1e-6);
        assert!((score.codec_probability(Codec::AacAt) - floor / total).abs() < 1e-8);
        assert!((score.codec_probability(Codec::Musepack) - floor / total).abs() < 1e-8);
    }

    #[test]
    fn scanner_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        assert_send_and_sync::<Scanner>();
    }
}
