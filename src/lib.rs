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
//!     println!("transcode from about {:.0} kbps", score.bitrate_kbps());
//! }
//! # Ok(())
//! # }
//! ```
//!

mod audio;
mod model;
mod spectrogram;

use audio::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, MIN_SECONDS};
use model::{Model, WindowScore, FAMILY_COUNT};
use spectrogram::TransformCache;
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
    /// The audio is too short to analyze.
    #[error("audio is shorter than the {MIN_SECONDS}-second minimum")]
    TooShort,
    /// The model could not score prepared audio.
    #[error("model inference failed: {0}")]
    Inference(#[from] ModelError),
}

/// The model's output columns, in order.
///
/// TODO: restore the per-encoder AAC classes (`AacAt`, `FdkAac`) once the
/// model distinguishes AAC implementations again; for now every AAC source
/// is reported as `FfmpegAac`.
const MODEL_ENCODERS: [Encoder; FAMILY_COUNT] = [
    Encoder::Mp3,
    Encoder::FfmpegAac,
    Encoder::Vorbis,
    Encoder::Opus,
    Encoder::Wma,
    Encoder::Mp2,
    Encoder::Musepack,
];

/// Every encoder, in the order of its identifier.
const ENCODERS: [Encoder; 9] = [
    Encoder::AacAt,
    Encoder::FdkAac,
    Encoder::FfmpegAac,
    Encoder::Mp2,
    Encoder::Mp3,
    Encoder::Musepack,
    Encoder::Opus,
    Encoder::Vorbis,
    Encoder::Wma,
];

/// A source encoder class predicted by the model.
///
/// Ordering follows the lexicographic order of the stable identifiers returned
/// by [`Encoder::as_str`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum Encoder {
    /// Apple AudioToolbox AAC.
    AacAt,
    /// Fraunhofer FDK AAC.
    FdkAac,
    /// FFmpeg's native AAC encoder.
    FfmpegAac,
    /// MP2.
    Mp2,
    /// MP3.
    Mp3,
    /// Musepack.
    Musepack,
    /// Opus.
    Opus,
    /// Vorbis.
    Vorbis,
    /// Windows Media Audio.
    Wma,
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

    /// The model column for this encoder, if the model reports it.
    const fn model_index(self) -> Option<usize> {
        match self {
            Self::Mp3 => Some(0),
            Self::FfmpegAac => Some(1),
            Self::Vorbis => Some(2),
            Self::Opus => Some(3),
            Self::Wma => Some(4),
            Self::Mp2 => Some(5),
            Self::Musepack => Some(6),
            Self::AacAt | Self::FdkAac => None,
        }
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

/// Model outputs pooled across a track's analysis windows.
#[derive(Clone, Debug)]
pub struct TrackScore {
    transcode_probability: f32,
    encoder_probabilities: [f32; FAMILY_COUNT],
    bitrate_kbps: f32,
}

impl TrackScore {
    /// Return the probability that this track is a lossy transcode.
    #[must_use]
    pub fn transcode_probability(&self) -> f32 {
        self.transcode_probability
    }

    /// Return the probability for one encoder class (conditional on being a transcode).
    ///
    /// Encoders the model does not report have probability zero.
    #[must_use]
    pub fn encoder_probability(&self, encoder: Encoder) -> f32 {
        encoder
            .model_index()
            .map_or(0.0, |index| self.encoder_probabilities[index])
    }

    /// Iterate over every encoder the model reports and its conditional probability.
    ///
    /// The iteration order is unspecified and may change with the model.
    pub fn encoder_probabilities(&self) -> impl Iterator<Item = (Encoder, f32)> + '_ {
        MODEL_ENCODERS
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

    /// Return the estimated bitrate of the lossy source, in kilobits per second.
    ///
    /// The estimate reads how much detail survived encoding, so it is only
    /// meaningful for tracks the caller judges to be transcodes.
    #[must_use]
    pub fn bitrate_kbps(&self) -> f32 {
        self.bitrate_kbps
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
    /// Tracks up to 14.5 seconds are scored whole; longer tracks are scored
    /// on up to six evenly spaced 14.5-second windows drawn from their middle
    /// 92%, with per-window outputs averaged.
    pub fn score_file(&self, file: File) -> Result<TrackScore, ScoreError> {
        self.score_windows(&audio::decode_file(file)?)
    }

    /// Score decoded windows. Non-generic, so the spectrogram and inference
    /// code is generated once rather than once per reader type.
    fn score_windows(&self, windows: &[audio::Clip]) -> Result<TrackScore, ScoreError> {
        debug_assert!(!windows.is_empty());
        let mut scores = Vec::with_capacity(windows.len());
        for clip in windows {
            let transform = self.transforms.get(clip.sample_rate);
            let channels = clip.channels.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let (window, frames) = transform.write_window(&channels);
            scores.push(self.model.run(window, frames).map_err(ModelError)?);
        }
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
    /// Average probabilities across windows; average the bitrate in log2 space.
    fn pool(scores: &[WindowScore]) -> Self {
        debug_assert!(!scores.is_empty());
        let windows = scores.len() as f32;
        let mut transcode_sum = 0.0_f32;
        let mut encoder_sums = [0.0_f32; FAMILY_COUNT];
        let mut log2_bitrate_sum = 0.0_f32;
        for score in scores {
            transcode_sum += score.transcode_probability;
            for (sum, probability) in encoder_sums.iter_mut().zip(score.family_probabilities) {
                *sum += probability;
            }
            log2_bitrate_sum += score.bitrate_kbps.max(1.0).log2();
        }
        Self {
            transcode_probability: transcode_sum / windows,
            encoder_probabilities: encoder_sums.map(|sum| sum / windows),
            bitrate_kbps: (log2_bitrate_sum / windows).exp2(),
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
            family_probabilities: [0.02, 0.04, 0.06, 0.08, 0.1, 0.3, 0.4],
            bitrate_kbps: 192.0,
        }]);

        assert!((score.transcode_probability() - 0.75).abs() < 1e-6);
        assert!((score.bitrate_kbps() - 192.0).abs() < 1e-3);
        let expected = [
            (Encoder::Mp3, 0.02),
            (Encoder::FfmpegAac, 0.04),
            (Encoder::Vorbis, 0.06),
            (Encoder::Opus, 0.08),
            (Encoder::Wma, 0.1),
            (Encoder::Mp2, 0.3),
            (Encoder::Musepack, 0.4),
        ];
        for ((actual_encoder, actual_probability), (expected_encoder, expected_probability)) in
            score.encoder_probabilities().zip(expected)
        {
            assert_eq!(actual_encoder, expected_encoder);
            assert!((actual_probability - expected_probability).abs() < 1e-6);
            assert!(
                (score.encoder_probability(actual_encoder) - expected_probability).abs() < 1e-6
            );
        }
        assert_eq!(score.most_likely_encoder(), (Encoder::Musepack, 0.4));
        // Classes the model does not report score zero.
        assert_eq!(score.encoder_probability(Encoder::AacAt), 0.0);
        assert_eq!(score.encoder_probability(Encoder::FdkAac), 0.0);
    }

    #[test]
    fn encoder_identifiers_are_stable() {
        assert_eq!(
            ENCODERS.map(Encoder::as_str),
            [
                "aac_at",
                "fdk_aac",
                "ffmpeg_aac",
                "mp2",
                "mp3",
                "musepack",
                "opus",
                "vorbis",
                "wma",
            ]
        );
    }

    #[test]
    fn encoder_model_columns_are_explicit() {
        for (position, encoder) in MODEL_ENCODERS.into_iter().enumerate() {
            assert_eq!(encoder.model_index(), Some(position));
        }
    }

    #[test]
    fn encoder_order_follows_its_stable_identifier() {
        let mut encoders = ENCODERS;
        encoders.sort_unstable();
        assert_eq!(encoders, ENCODERS);
        let mut identifiers = ENCODERS.map(Encoder::as_str);
        identifiers.sort_unstable();
        assert_eq!(identifiers, ENCODERS.map(Encoder::as_str));
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
    fn track_score_averages_probabilities_and_log_bitrate() {
        let score = TrackScore::pool(&[
            WindowScore {
                transcode_probability: 0.2,
                family_probabilities: [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                bitrate_kbps: 128.0,
            },
            WindowScore {
                transcode_probability: 0.8,
                family_probabilities: [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
                bitrate_kbps: 512.0,
            },
        ]);

        assert!((score.transcode_probability() - 0.5).abs() < 1e-6);
        assert!((score.encoder_probability(Encoder::Mp3) - 0.65).abs() < 1e-6);
        assert!((score.encoder_probability(Encoder::FfmpegAac) - 0.35).abs() < 1e-6);
        assert_eq!(score.encoder_probability(Encoder::Opus), 0.0);
        // Geometric mean of 128 and 512.
        assert!((score.bitrate_kbps() - 256.0).abs() < 1e-3);
    }
}
