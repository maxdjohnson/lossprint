//! Scan lossless audio containers for evidence of an earlier lossy encode.

mod audio;
mod model;
mod spectrogram;

use anyhow::{bail, Context, Result};
use clap::Parser;
use model::{Model, WindowScore, ENCODER_LABELS};
use rayon::prelude::*;
use spectrogram::{TransformCache, CROP_SECONDS, MAX_WINDOWS, WINDOW_VALUES};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use walkdir::WalkDir;

const MAX_BATCH_TRACKS: u8 = 8;
const READY_FILES: usize = 2;
const DEFAULT_THRESHOLD: f32 = 0.4;

fn default_batch_size(files: usize) -> u8 {
    // Eight tracks per model forward pass gives the best throughput for large Apple
    // Silicon libraries without imposing that memory cost on smaller scans.
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) && files >= 16 {
        MAX_BATCH_TRACKS
    } else {
        1
    }
}

#[derive(Default)]
struct FileScore {
    error: Option<anyhow::Error>,
    transcode_sum: f32,
    encoder_sums: [f32; ENCODER_LABELS.len()],
    windows: usize,
}

impl FileScore {
    fn add(&mut self, score: &WindowScore) {
        self.transcode_sum += score.transcode_probability;
        for (sum, probability) in self
            .encoder_sums
            .iter_mut()
            .zip(score.encoder_probabilities)
        {
            *sum += probability;
        }
        self.windows += 1;
    }

    fn finish(self) -> Result<TrackScore> {
        if let Some(error) = self.error {
            return Err(error);
        }
        debug_assert!(self.windows > 0);
        let windows = self.windows as f32;
        Ok(TrackScore {
            transcode_probability: self.transcode_sum / windows,
            encoder_probabilities: self.encoder_sums.map(|sum| sum / windows),
        })
    }
}

struct TrackScore {
    transcode_probability: f32,
    encoder_probabilities: [f32; ENCODER_LABELS.len()],
}

impl TrackScore {
    fn classification(&self, threshold: f32) -> (&'static str, &'static str) {
        if self.transcode_probability < threshold {
            return ("clean", "-");
        }
        let mut best = 0;
        for index in 1..ENCODER_LABELS.len() {
            if self.encoder_probabilities[index] > self.encoder_probabilities[best] {
                best = index;
            }
        }
        ("transcode", ENCODER_LABELS[best])
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
        let predictions = model.run(&self.input)?;
        for (&owner, prediction) in self.owners.iter().zip(&predictions) {
            scores[owner].add(prediction);
        }
        self.input.clear();
        self.owners.clear();
        Ok(())
    }
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Detect lossy transcodes hiding in lossless audio files",
    after_help = "Scans WAV, AIFF, and FLAC files. Output is tab-separated:\n\
                  probability, verdict, codec, path. Use --threshold 0.5 when false\n\
                  positives cost more than missed transcodes.\n\n\
                  Includes Symphonia 0.6.1 under MPL-2.0; source:\n\
                  https://github.com/pdeljanov/Symphonia/tree/v0.6.1"
)]
struct Args {
    /// Audio files or directories to scan recursively.
    #[arg(required = true, value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// P(transcode) at or above this value is classified as a transcode.
    #[arg(short, long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f32,

    /// Parallel workers; zero uses one worker per logical core.
    #[arg(short, long, default_value_t = 0)]
    jobs: usize,

    /// Tracks' worth of windows per model forward pass (1-8); defaults to 1, or 8 on Apple Silicon with at least 16 tracks.
    #[arg(
        long,
        value_name = "TRACKS",
        value_parser = clap::value_parser!(u8).range(1..=MAX_BATCH_TRACKS as i64)
    )]
    batch_size: Option<u8>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(0.0..=1.0).contains(&args.threshold) {
        bail!("--threshold must be between 0 and 1")
    }

    let files = discover(&args.paths)?;
    if files.is_empty() {
        bail!("no WAV, AIFF, or FLAC files found")
    }
    let batch_size = usize::from(
        args.batch_size
            .unwrap_or_else(|| default_batch_size(files.len())),
    );
    let batch_windows = batch_size * MAX_WINDOWS;
    let mut model = Model::new()?;
    let transforms = TransformCache::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?;
    let results = score_all(&files, &mut model, &transforms, &pool, batch_windows)?;

    let mut stdout = BufWriter::new(std::io::stdout().lock());
    let mut failures = 0_usize;
    for (path, result) in files.iter().zip(results) {
        match result {
            Ok(score) => {
                let (verdict, codec) = score.classification(args.threshold);
                writeln!(
                    stdout,
                    "{:.7}\t{}\t{}\t{}",
                    score.transcode_probability,
                    verdict,
                    codec,
                    path.display()
                )?;
            }
            Err(error) => {
                failures += 1;
                eprintln!("lossprint: {}: {error:#}", path.display());
            }
        }
    }
    stdout.flush()?;
    if failures > 0 {
        bail!("could not scan {failures} file(s)")
    }
    Ok(())
}

fn prepare(path: &Path, transforms: &TransformCache) -> Result<Vec<f32>> {
    let clip = audio::decode(path)?;
    let crop_len = clip.sample_rate as usize * CROP_SECONDS;
    let offsets = spectrogram::window_offsets(clip.channels[0].len(), crop_len);
    if offsets.is_empty() {
        bail!("audio is shorter than {CROP_SECONDS} seconds")
    }
    let transform = transforms.get(clip.sample_rate)?;
    Ok(transform.write_windows(&clip.channels, &offsets))
}

fn score_all(
    files: &[PathBuf],
    model: &mut Model,
    transforms: &TransformCache,
    pool: &rayon::ThreadPool,
    batch_windows: usize,
) -> Result<Vec<Result<TrackScore>>> {
    let (sender, receiver) = sync_channel::<(usize, Result<Vec<f32>>)>(READY_FILES);
    let mut scores = (0..files.len())
        .map(|_| FileScore::default())
        .collect::<Vec<_>>();
    let mut pending = PendingBatch::new(batch_windows);

    std::thread::scope(|scope| -> Result<()> {
        scope.spawn(|| {
            let _ = pool.install(|| {
                files
                    .par_iter()
                    .enumerate()
                    .try_for_each_with(sender, |sender, (index, path)| {
                        sender
                            .send((index, prepare(path, transforms)))
                            .map_err(|_| ())
                    })
            });
        });

        for (index, prepared) in receiver {
            match prepared {
                Ok(windows) => pending.push(index, &windows, model, &mut scores)?,
                Err(error) => scores[index].error = Some(error),
            }
        }
        pending.finish(model, &mut scores)
    })?;

    Ok(scores.into_iter().map(FileScore::finish).collect())
}

fn discover(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            if !audio::is_supported(input) {
                bail!("unsupported audio file: {}", input.display())
            }
            files.push(input.clone());
        } else if input.is_dir() {
            for entry in WalkDir::new(input) {
                let entry = entry.with_context(|| format!("could not walk {}", input.display()))?;
                if entry.file_type().is_file() && audio::is_supported(entry.path()) {
                    files.push(entry.into_path());
                }
            }
        } else {
            bail!("path is not a file or directory: {}", input.display())
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_is_bounded() {
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "1", "audio.flac"]).is_ok());
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "8", "audio.flac"]).is_ok());
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "0", "audio.flac"]).is_err());
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "9", "audio.flac"]).is_err());
    }

    #[test]
    fn file_score_averages_encoder_probabilities_across_windows() {
        let mut score = FileScore::default();
        score.add(&WindowScore {
            transcode_probability: 0.6,
            encoder_probabilities: [0.51, 0.49, 0.0, 0.0, 0.0, 0.0],
        });
        score.add(&WindowScore {
            transcode_probability: 0.8,
            encoder_probabilities: [0.51, 0.49, 0.0, 0.0, 0.0, 0.0],
        });
        score.add(&WindowScore {
            transcode_probability: 1.0,
            encoder_probabilities: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        });

        let score = score.finish().unwrap();
        assert!((score.transcode_probability - 0.8).abs() < f32::EPSILON);
        assert!((score.encoder_probabilities[0] - 0.34).abs() < f32::EPSILON);
        assert!((score.encoder_probabilities[1] - 0.66).abs() < f32::EPSILON);
        assert_eq!(score.classification(0.4), ("transcode", "aac"));
    }

    #[test]
    fn clean_tracks_hide_the_encoder_prediction() {
        let score = TrackScore {
            transcode_probability: 0.39,
            encoder_probabilities: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };

        assert_eq!(score.classification(0.4), ("clean", "-"));
    }
}
