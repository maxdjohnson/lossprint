#![warn(missing_docs)]
//! Detect lossy transcodes hiding in lossless audio files.
//!
//! Reuse a [`Scanner`] when scoring multiple tracks so that its model, worker
//! pool, and spectrogram transforms stay initialized.
//!
//! ```no_run
//! use lossprint::Scanner;
//!
//! # #[cfg(feature = "bundled-model")]
//! # fn main() -> lossprint::Result<()> {
//! let mut scanner = Scanner::new()?;
//! let score = scanner.score_file("track.flac")?;
//!
//! println!("P(transcode) = {:.3}", score.prob_transcode);
//! println!("P(mp3 | transcode) = {:.3}", score.prob_codec.mp3);
//! if score.prob_transcode >= 0.5 {
//!     println!("transcode");
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "bundled-model"))]
//! # fn main() {}
//! ```

mod audio;
mod model;
mod spectrogram;

use anyhow::{anyhow, bail, Result as AnyResult};
use model::{Model, WindowScore, CODEC_COUNT};
use rayon::prelude::*;
use spectrogram::{TransformCache, MAX_WINDOWS, WINDOW_VALUES};
use std::fmt;
use std::path::Path;
use std::sync::mpsc::sync_channel;

const READY_FILES: usize = 2;

/// The largest supported number of tracks per model batch.
pub const MAX_BATCH_TRACKS: u8 = 8;

/// A result returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure produced while configuring or running a scan.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    inner: anyhow::Error,
}

impl Error {
    fn new(kind: ErrorKind, inner: impl Into<anyhow::Error>) -> Self {
        Self {
            kind,
            inner: inner.into(),
        }
    }

    /// Return the broad category of this failure.
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if formatter.alternate() {
            write!(formatter, "{:#}", self.inner)
        } else {
            fmt::Display::fmt(&self.inner, formatter)
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.inner.as_ref())
    }
}

/// The broad category of an [`Error`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A caller-supplied option or value is invalid.
    Configuration,
    /// The model or inference runtime could not be initialized.
    Initialization,
    /// An input file could not be decoded or transformed.
    Audio,
    /// The model could not score prepared audio.
    Inference,
    /// The parallel worker pool could not be created.
    Worker,
}

/// Conditional codec and encoder-class probabilities returned for a track.
///
/// These probabilities describe the likely source codec or encoder
/// implementation and are meaningful only when the caller considers the
/// track a transcode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodecProbabilities {
    /// MP3 probability.
    pub mp3: f32,
    /// Generic AAC probability.
    pub aac: f32,
    /// Apple AudioToolbox AAC probability.
    pub aac_at: f32,
    /// Fraunhofer FDK AAC probability.
    pub fdk_aac: f32,
    /// Vorbis probability.
    pub vorbis: f32,
    /// Opus probability.
    pub opus: f32,
    /// MP2 probability.
    pub mp2: f32,
    /// Windows Media Audio probability.
    pub wma: f32,
    /// Musepack probability.
    pub musepack: f32,
}

impl CodecProbabilities {
    fn from_array(probabilities: [f32; CODEC_COUNT]) -> Self {
        let [mp3, aac, aac_at, fdk_aac, vorbis, opus, mp2, wma, musepack] = probabilities;
        Self {
            mp3,
            aac,
            aac_at,
            fdk_aac,
            vorbis,
            opus,
            mp2,
            wma,
            musepack,
        }
    }
}

/// Model probabilities pooled across a track's analysis windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackScore {
    /// The probability that this track is a lossy transcode.
    pub prob_transcode: f32,
    /// The conditional probabilities of each codec or encoder class, given that
    /// this track is a lossy transcode.
    pub prob_codec: CodecProbabilities,
}

/// Configuration for constructing a reusable [`Scanner`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ScannerBuilder {
    jobs: usize,
    batch_size: Option<u8>,
}

impl ScannerBuilder {
    /// Set the number of parallel audio workers; zero uses Rayon's default.
    pub const fn jobs(mut self, jobs: usize) -> Self {
        self.jobs = jobs;
        self
    }

    /// Set the number of tracks represented in each model batch.
    ///
    /// Valid values are 1 through [`MAX_BATCH_TRACKS`]. When unset, scans of at
    /// least 16 tracks use 8; smaller scans use 1.
    pub const fn batch_size(mut self, batch_size: u8) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Build a scanner using the downloaded-and-embedded model.
    #[cfg(feature = "bundled-model")]
    pub fn build(self) -> Result<Scanner> {
        let (pool, batch_size) = self.prepare()?;
        let model =
            Model::bundled().map_err(|error| Error::new(ErrorKind::Initialization, error))?;
        Ok(Scanner::from_parts(model, pool, batch_size))
    }

    /// Build a scanner from caller-supplied ONNX model bytes.
    pub fn build_from_model_bytes(self, model: &[u8]) -> Result<Scanner> {
        let (pool, batch_size) = self.prepare()?;
        let model = Model::from_bytes(model)
            .map_err(|error| Error::new(ErrorKind::Initialization, error))?;
        Ok(Scanner::from_parts(model, pool, batch_size))
    }

    /// Build a scanner from a caller-supplied ONNX model file.
    pub fn build_from_model_file(self, model: impl AsRef<Path>) -> Result<Scanner> {
        let (pool, batch_size) = self.prepare()?;
        let model = Model::from_file(model.as_ref())
            .map_err(|error| Error::new(ErrorKind::Initialization, error))?;
        Ok(Scanner::from_parts(model, pool, batch_size))
    }

    fn prepare(self) -> Result<(rayon::ThreadPool, Option<u8>)> {
        if self
            .batch_size
            .is_some_and(|size| !(1..=MAX_BATCH_TRACKS).contains(&size))
        {
            return Err(Error::new(
                ErrorKind::Configuration,
                anyhow!("batch size must be between 1 and {MAX_BATCH_TRACKS}"),
            ));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.jobs)
            .build()
            .map_err(|error| Error::new(ErrorKind::Worker, error))?;
        Ok((pool, self.batch_size))
    }
}

/// A reusable audio-transcode scanner.
pub struct Scanner {
    model: Model,
    transforms: TransformCache,
    pool: rayon::ThreadPool,
    batch_size: Option<u8>,
}

impl Scanner {
    /// Construct a scanner using the downloaded-and-embedded model.
    #[cfg(feature = "bundled-model")]
    pub fn new() -> Result<Self> {
        ScannerBuilder::default().build()
    }

    /// Begin configuring a scanner.
    pub const fn builder() -> ScannerBuilder {
        ScannerBuilder {
            jobs: 0,
            batch_size: None,
        }
    }

    /// Construct a scanner from caller-supplied ONNX model bytes.
    pub fn from_model_bytes(model: &[u8]) -> Result<Self> {
        Self::builder().build_from_model_bytes(model)
    }

    /// Construct a scanner from a caller-supplied ONNX model file.
    pub fn from_model_file(model: impl AsRef<Path>) -> Result<Self> {
        Self::builder().build_from_model_file(model)
    }

    fn from_parts(model: Model, pool: rayon::ThreadPool, batch_size: Option<u8>) -> Self {
        Self {
            model,
            transforms: TransformCache::default(),
            pool,
            batch_size,
        }
    }

    /// Score one WAV, AIFF, or FLAC file.
    pub fn score_file(&mut self, path: impl AsRef<Path>) -> Result<TrackScore> {
        let paths = [path.as_ref()];
        self.score_files(&paths)?
            .pop()
            .expect("one input produces one result")
    }

    /// Score files in parallel while preserving their input order.
    ///
    /// The outer [`Result`] reports a batch-wide worker or inference failure.
    /// Each inner result reports decoding or transformation failure for only
    /// the file at that same index.
    pub fn score_files<P>(&mut self, files: &[P]) -> Result<Vec<Result<TrackScore>>>
    where
        P: AsRef<Path> + Sync,
    {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = usize::from(
            self.batch_size
                .unwrap_or_else(|| default_batch_size(files.len())),
        );
        let batch_windows = batch_size * MAX_WINDOWS;
        let (sender, receiver) = sync_channel::<(usize, AnyResult<Vec<f32>>)>(READY_FILES);
        let mut scores = (0..files.len())
            .map(|_| FileScore::default())
            .collect::<Vec<_>>();
        let mut pending = PendingBatch::new(batch_windows);

        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
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
                    Ok(windows) => pending.push(index, &windows, &mut self.model, &mut scores)?,
                    Err(error) => {
                        scores[index].error = Some(Error::new(ErrorKind::Audio, error));
                    }
                }
            }
            pending.finish(&mut self.model, &mut scores)
        })?;

        Ok(scores.into_iter().map(FileScore::finish).collect())
    }
}

/// Return whether a path has a supported WAV, AIFF, or FLAC extension.
pub fn is_supported_file(path: impl AsRef<Path>) -> bool {
    audio::is_supported(path.as_ref())
}

fn default_batch_size(files: usize) -> u8 {
    if files >= 16 {
        MAX_BATCH_TRACKS
    } else {
        1
    }
}

fn prepare(path: &Path, transforms: &TransformCache) -> AnyResult<Vec<f32>> {
    let clip = audio::decode(path)?;
    let crop_len = clip.sample_rate as usize / 2;
    let offsets = spectrogram::window_offsets(clip.channels[0].len(), crop_len);
    if offsets.is_empty() {
        bail!("audio is shorter than 0.5 seconds")
    }
    let transform = transforms.get(clip.sample_rate)?;
    Ok(transform.write_windows(&clip.channels, &offsets))
}

#[derive(Default)]
struct FileScore {
    error: Option<Error>,
    transcode_logit_sum: f32,
    codec_log_sums: [f32; CODEC_COUNT],
    windows: usize,
}

impl FileScore {
    fn add(&mut self, score: &WindowScore) {
        let probability = score.transcode_probability.clamp(1e-6, 1.0 - 1e-6);
        self.transcode_logit_sum += probability.ln() - (-probability).ln_1p();
        for (sum, probability) in self
            .codec_log_sums
            .iter_mut()
            .zip(score.codec_probabilities)
        {
            *sum += probability.clamp(1e-6, 1.0).ln();
        }
        self.windows += 1;
    }

    fn finish(self) -> Result<TrackScore> {
        if let Some(error) = self.error {
            return Err(error);
        }
        debug_assert!(self.windows > 0);
        let windows = self.windows as f32;
        let mean_logit = self.transcode_logit_sum / windows;
        let mean_logs = self.codec_log_sums.map(|sum| sum / windows);
        let max_log = mean_logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let geometric = mean_logs.map(|value| (value - max_log).exp());
        let codec_total: f32 = geometric.iter().sum();
        Ok(TrackScore {
            prob_transcode: 1.0 / (1.0 + (-mean_logit).exp()),
            prob_codec: CodecProbabilities::from_array(
                geometric.map(|probability| probability / codec_total),
            ),
        })
    }
}

struct PendingBatch {
    max_windows: usize,
    input: Vec<f32>,
    owners: Vec<usize>,
}

impl PendingBatch {
    fn new(max_windows: usize) -> Self {
        Self {
            max_windows,
            input: Vec::with_capacity(max_windows * WINDOW_VALUES),
            owners: Vec::with_capacity(max_windows),
        }
    }

    fn push(
        &mut self,
        owner: usize,
        windows: &[f32],
        model: &mut Model,
        scores: &mut [FileScore],
    ) -> Result<()> {
        debug_assert!(!windows.is_empty());
        debug_assert!(windows.len().is_multiple_of(WINDOW_VALUES));
        for window in windows.chunks_exact(WINDOW_VALUES) {
            self.input.extend_from_slice(window);
            self.owners.push(owner);
            if self.owners.len() == self.max_windows {
                self.flush(model, scores)?;
            }
        }
        Ok(())
    }

    fn finish(&mut self, model: &mut Model, scores: &mut [FileScore]) -> Result<()> {
        if self.owners.is_empty() {
            return Ok(());
        }
        self.flush(model, scores)
    }

    fn flush(&mut self, model: &mut Model, scores: &mut [FileScore]) -> Result<()> {
        debug_assert_eq!(self.input.len(), self.owners.len() * WINDOW_VALUES);
        self.input.resize(self.max_windows * WINDOW_VALUES, 0.0);
        let predictions = model
            .run(&self.input)
            .map_err(|error| Error::new(ErrorKind::Inference, error))?;
        for (&owner, prediction) in self.owners.iter().zip(&predictions) {
            scores[owner].add(prediction);
        }
        self.input.clear();
        self.owners.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_outputs_map_to_named_codec_fields() {
        let mut score = FileScore::default();
        score.add(&WindowScore {
            transcode_probability: 0.75,
            codec_probabilities: [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.28],
        });

        let score = score.finish().unwrap();
        assert!((score.prob_transcode - 0.75).abs() < 1e-6);
        let actual = [
            score.prob_codec.mp3,
            score.prob_codec.aac,
            score.prob_codec.aac_at,
            score.prob_codec.fdk_aac,
            score.prob_codec.vorbis,
            score.prob_codec.opus,
            score.prob_codec.mp2,
            score.prob_codec.wma,
            score.prob_codec.musepack,
        ];
        for (actual, expected) in actual
            .into_iter()
            .zip([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.28])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn file_score_uses_geometric_probability_pooling() {
        let mut score = FileScore::default();
        score.add(&WindowScore {
            transcode_probability: 0.2,
            codec_probabilities: [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        });
        score.add(&WindowScore {
            transcode_probability: 0.8,
            codec_probabilities: [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        });

        let score = score.finish().unwrap();
        assert!((score.prob_transcode - 0.5).abs() < 1e-6);
        let mp3 = 0.36_f32.sqrt();
        let aac = 0.06_f32.sqrt();
        let floor = 1e-6_f32;
        let total = mp3 + aac + 7.0 * floor;
        assert!((score.prob_codec.mp3 - mp3 / total).abs() < 1e-6);
        assert!((score.prob_codec.aac - aac / total).abs() < 1e-6);
        assert!((score.prob_codec.aac_at - floor / total).abs() < 1e-8);
        assert!((score.prob_codec.musepack - floor / total).abs() < 1e-8);
    }

    #[test]
    fn builder_rejects_out_of_range_batches_before_model_initialization() {
        for invalid in [0, 9] {
            let error = Scanner::builder()
                .batch_size(invalid)
                .prepare()
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Configuration);
        }
        for valid in [1, 8] {
            assert!(Scanner::builder().batch_size(valid).prepare().is_ok());
        }
    }
}
