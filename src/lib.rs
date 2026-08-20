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
//! let score = scanner.score_file(File::open("track.flac")?)?;
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
//!

mod audio;
mod model;
mod spectrogram;

use audio::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};
use model::{Model, WindowScore, ENCODER_COUNT};
use spectrogram::{TransformCache, WINDOW_SECONDS};
use std::fmt;
use std::fs::File;
use std::str::FromStr;
use tract_onnx::prelude::TractError;

/// A failure reported by the model runtime.
#[derive(Debug)]
pub struct ModelError(TractError);

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:#}", self.0)
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.chain().next()
    }
}

/// A failure while initializing a [`Scanner`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InitializationError {
    /// The embedded model could not be parsed, optimized, or prepared.
    #[error("model initialization failed: {0}")]
    Model(#[from] ModelError),
}

/// A failure while decoding an input audio file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DecodeError {
    /// The input could not be parsed or decoded as WAV, AIFF, or FLAC.
    ///
    /// The underlying decoder error is available through
    /// [`std::error::Error::source`].
    #[error("could not decode audio: {source}")]
    #[non_exhaustive]
    Malformed {
        /// The decoder failure that produced this error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// The file declares no usable audio track, or that track decoded to nothing.
    #[error("audio file has no usable audio track")]
    NoUsableAudio,
    /// The track is companded rather than linear PCM.
    #[error("A-law and mu-law audio are not lossless PCM")]
    NotLinearPcm,
}

/// A failure while scoring audio.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScoreError {
    /// The input could not be decoded.
    #[error("{0}")]
    Decode(#[from] DecodeError),
    /// The sample rate is outside the range the model supports.
    #[error(
        "{rate} Hz sample rate is outside the supported {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz range"
    )]
    #[non_exhaustive]
    UnsupportedSampleRate {
        /// The audio's sample rate, in hertz.
        rate: u32,
    },
    /// The audio is neither mono nor stereo.
    #[error("{channels} channels; only mono and stereo are supported")]
    #[non_exhaustive]
    UnsupportedChannelCount {
        /// The audio's channel count.
        channels: usize,
    },
    /// The audio cannot fill one analysis window.
    #[error("audio is shorter than the {WINDOW_SECONDS}-second analysis window")]
    TooShort,
    /// The model could not score prepared audio.
    #[error("model inference failed: {0}")]
    Inference(#[from] ModelError),
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
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
#[repr(usize)]
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
    #[must_use]
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
        self as usize
    }
}

impl fmt::Display for Encoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The error returned when a string does not name an [`Encoder`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("not a known encoder identifier")]
#[non_exhaustive]
pub struct UnknownEncoder;

impl FromStr for Encoder {
    type Err = UnknownEncoder;

    /// Parse an identifier produced by [`Encoder::as_str`].
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        ENCODERS
            .into_iter()
            .find(|encoder| encoder.as_str() == text)
            .ok_or(UnknownEncoder)
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
    #[must_use]
    pub fn transcode_probability(&self) -> f32 {
        self.transcode_probability
    }

    /// Return the probability for one encoder class (conditional on being a transcode).
    #[must_use]
    pub fn encoder_probability(&self, encoder: Encoder) -> f32 {
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
    #[must_use]
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

impl fmt::Debug for Scanner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Scanner").finish_non_exhaustive()
    }
}

impl Scanner {
    /// Construct a scanner using the embedded model.
    pub fn new() -> Result<Self, InitializationError> {
        let model = Model::new().map_err(ModelError)?;
        Ok(Self {
            model,
            transforms: TransformCache::default(),
        })
    }

    /// Score one WAV, AIFF, or FLAC file.
    ///
    /// Only the first 20 seconds are decoded.
    pub fn score_file(&self, file: File) -> Result<TrackScore, ScoreError> {
        self.score_clip(audio::decode_file(file)?)
    }

    /// Score decoded audio. Non-generic, so the spectrogram and inference code
    /// is generated once rather than once per reader type.
    fn score_clip(&self, clip: audio::Clip) -> Result<TrackScore, ScoreError> {
        // `decode` validated the sample rate and channel count, and truncated
        // every channel to the first 20 seconds.
        let frames = clip
            .channels
            .iter()
            .map(|channel| channel.len())
            .min()
            .unwrap_or(0);
        let cropped = clip
            .channels
            .iter()
            .map(|channel| &channel[..frames])
            .collect::<Vec<_>>();

        let offsets =
            spectrogram::window_offsets(frames, spectrogram::window_len(clip.sample_rate));
        if offsets.is_empty() {
            return Err(ScoreError::TooShort);
        }
        let windows = self
            .transforms
            .get(clip.sample_rate)
            .write_windows(&cropped, &offsets);
        let scores = self.model.run(windows).map_err(ModelError)?;
        Ok(TrackScore::pool(&scores))
    }
}

/// Reject audio the model cannot accept, whether decoded here or supplied.
pub(crate) fn validate(sample_rate: u32, channels: usize) -> Result<(), ScoreError> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(ScoreError::UnsupportedSampleRate { rate: sample_rate });
    }
    if !(1..=2).contains(&channels) {
        return Err(ScoreError::UnsupportedChannelCount { channels });
    }
    Ok(())
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

    #[test]
    fn model_errors_display_and_preserve_context() {
        let failure = ModelError(TractError::msg("backend detail").context("operation"));
        assert_eq!(failure.to_string(), "operation: backend detail");
        let context = std::error::Error::source(&failure).unwrap();
        assert_eq!(context.to_string(), "operation");
        assert_eq!(context.source().unwrap().to_string(), "backend detail");
    }

    #[test]
    fn one_model_error_serves_initialization_and_inference() {
        let failure = || ModelError(TractError::msg("backend detail").context("operation"));

        // Both paths convert from the same type and keep their own prefix.
        assert_eq!(
            InitializationError::from(failure()).to_string(),
            "model initialization failed: operation: backend detail"
        );
        assert_eq!(
            ScoreError::from(failure()).to_string(),
            "model inference failed: operation: backend detail"
        );

        // The runtime chain stays reachable without exposing the runtime.
        let score_error = ScoreError::from(failure());
        let model = std::error::Error::source(&score_error).expect("model error is the source");
        assert!(model.downcast_ref::<ModelError>().is_some());
        assert_eq!(model.source().unwrap().to_string(), "operation");
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
    fn encoder_discriminants_match_their_position() {
        for (position, encoder) in ENCODERS.into_iter().enumerate() {
            assert_eq!(encoder.index(), position);
        }
    }

    #[test]
    fn encoder_identifiers_round_trip() {
        for encoder in ENCODERS {
            assert_eq!(encoder.as_str().parse::<Encoder>(), Ok(encoder));
        }
        assert_eq!("MP3".parse::<Encoder>(), Err(UnknownEncoder));
        assert_eq!("".parse::<Encoder>(), Err(UnknownEncoder));
    }

    #[test]
    fn unsupported_audio_is_rejected_before_decoding() {
        assert!(matches!(
            validate(4_000, 2),
            Err(ScoreError::UnsupportedSampleRate { rate: 4_000 })
        ));
        assert!(matches!(
            validate(44_100, 6),
            Err(ScoreError::UnsupportedChannelCount { channels: 6 })
        ));
        assert!(matches!(
            validate(44_100, 0),
            Err(ScoreError::UnsupportedChannelCount { channels: 0 })
        ));
        assert!(validate(44_100, 2).is_ok());
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
}
