#![warn(missing_docs)]
//! Detect lossy transcodes hiding in lossless audio files.
//!
//! Reuse a [`Scanner`] when scoring multiple tracks so that its model, worker
//! pool, and spectrogram transforms stay initialized.
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
//!     score.codec_probabilities().probability(Codec::Mp3)
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
use rayon::prelude::*;
use spectrogram::TransformCache;
use std::path::Path;
use std::sync::mpsc::sync_channel;

const READY_FILES: usize = 2;

pub use audio::Error as AudioError;
pub use model::{InferenceError, InitializationError};

/// A result returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure produced while constructing or using a scanner.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The embedded model could not be initialized.
    #[error(transparent)]
    Initialization(#[from] InitializationError),
    /// The parallel audio worker pool could not be created.
    #[error("could not create audio worker pool: {0}")]
    WorkerPool(#[from] rayon::ThreadPoolBuildError),
    /// An input file could not be decoded or prepared.
    #[error(transparent)]
    Audio(#[from] AudioError),
    /// The model could not score prepared audio.
    #[error(transparent)]
    Inference(#[from] InferenceError),
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
    /// Return the stable lowercase label used by the CLI.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
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

/// Conditional source-codec probabilities returned for a track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodecProbabilities {
    values: [f32; CODEC_COUNT],
}

impl CodecProbabilities {
    /// Return the probability for one codec or encoder class.
    pub const fn probability(&self, codec: Codec) -> f32 {
        self.values[codec.index()]
    }

    /// Iterate over every codec and probability in model order.
    pub fn iter(&self) -> impl Iterator<Item = (Codec, f32)> + '_ {
        CODECS.into_iter().zip(self.values.iter().copied())
    }
}

/// Model probabilities pooled across a track's analysis windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackScore {
    transcode_probability: f32,
    codec_probabilities: CodecProbabilities,
}

impl TrackScore {
    /// Return the probability that this track is a lossy transcode.
    pub const fn transcode_probability(&self) -> f32 {
        self.transcode_probability
    }

    /// Return source-codec probabilities conditional on the track being a transcode.
    pub const fn codec_probabilities(&self) -> &CodecProbabilities {
        &self.codec_probabilities
    }
}

/// Configuration for constructing a reusable [`Scanner`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ScannerBuilder {
    jobs: usize,
}

impl ScannerBuilder {
    /// Set the number of parallel audio workers; zero uses Rayon's default.
    pub const fn jobs(mut self, jobs: usize) -> Self {
        self.jobs = jobs;
        self
    }

    /// Build a scanner using the downloaded-and-embedded model.
    pub fn build(self) -> Result<Scanner> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.jobs)
            .build()?;
        let model = Model::new()?;
        Ok(Scanner::from_parts(model, pool))
    }
}

/// A reusable audio-transcode scanner.
pub struct Scanner {
    model: Model,
    transforms: TransformCache,
    pool: rayon::ThreadPool,
}

impl Scanner {
    /// Construct a scanner using the downloaded-and-embedded model.
    pub fn new() -> Result<Self> {
        ScannerBuilder::default().build()
    }

    /// Begin configuring a scanner.
    pub fn builder() -> ScannerBuilder {
        ScannerBuilder::default()
    }

    fn from_parts(model: Model, pool: rayon::ThreadPool) -> Self {
        Self {
            model,
            transforms: TransformCache::default(),
            pool,
        }
    }

    /// Score one WAV, AIFF, or FLAC file.
    pub fn score_file(&self, path: impl AsRef<Path>) -> Result<TrackScore> {
        let windows = prepare(path.as_ref(), &self.transforms)?;
        self.score_windows(&windows)
    }

    /// Score files in parallel while preserving their input order.
    ///
    /// Each track's windows are scored together in one model call. The outer
    /// [`Result`] reports an inference failure; each inner result reports an
    /// [`AudioError`] for only the file at that same index.
    pub fn score_files<P>(
        &self,
        files: &[P],
    ) -> Result<Vec<std::result::Result<TrackScore, AudioError>>>
    where
        P: AsRef<Path> + Sync,
    {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = (0..files.len()).map(|_| None).collect::<Vec<_>>();

        std::thread::scope(|scope| -> Result<()> {
            let (sender, receiver) = sync_channel(READY_FILES);
            scope.spawn(move || {
                let _ = self.pool.install(|| {
                    files.par_iter().enumerate().try_for_each_with(
                        sender,
                        |sender, (index, path)| {
                            sender
                                .send((index, prepare(path.as_ref(), &self.transforms)))
                                .map_err(|_| ())
                        },
                    )
                });
            });

            for (index, prepared) in receiver {
                match prepared {
                    Ok(windows) => scores[index] = Some(Ok(self.score_windows(&windows)?)),
                    Err(error) => scores[index] = Some(Err(error)),
                }
            }
            Ok(())
        })?;

        Ok(scores
            .into_iter()
            .map(|score| score.expect("every input produces one result"))
            .collect())
    }

    fn score_windows(&self, windows: &[f32]) -> Result<TrackScore> {
        let scores = self.model.run(windows)?;
        Ok(TrackScore::pool(&scores))
    }
}

/// Return whether a path has a supported WAV, AIFF, or FLAC extension.
pub fn has_supported_extension(path: impl AsRef<Path>) -> bool {
    audio::has_supported_extension(path.as_ref())
}

fn prepare(path: &Path, transforms: &TransformCache) -> std::result::Result<Vec<f32>, AudioError> {
    let clip = audio::decode(path)?;
    let crop_len = clip.sample_rate as usize / 2;
    let offsets = spectrogram::window_offsets(clip.channels[0].len(), crop_len);
    if offsets.is_empty() {
        return Err(AudioError::TooShort);
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
            codec_probabilities: CodecProbabilities {
                values: geometric.map(|probability| probability / codec_total),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            score.codec_probabilities().iter().zip(expected)
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
        let probabilities = score.codec_probabilities();
        assert!((probabilities.probability(Codec::Mp3) - mp3 / total).abs() < 1e-6);
        assert!((probabilities.probability(Codec::Aac) - aac / total).abs() < 1e-6);
        assert!((probabilities.probability(Codec::AacAt) - floor / total).abs() < 1e-8);
        assert!((probabilities.probability(Codec::Musepack) - floor / total).abs() < 1e-8);
    }

    #[test]
    fn scanner_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        assert_send_and_sync::<Scanner>();
    }
}
