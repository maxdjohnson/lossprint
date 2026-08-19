//! Scan lossless audio containers for evidence of an earlier lossy encode.

use anyhow::{bail, Context, Result};
use clap::Parser;
use lossprint::{has_supported_extension, Codec, Scanner, TrackScore};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use walkdir::WalkDir;

const DEFAULT_THRESHOLD: f32 = 0.5;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Detect lossy transcodes hiding in lossless audio files",
    after_help = "Scans WAV, AIFF, and FLAC files. Output is tab-separated:\n\
                  probability, verdict, codec, path. Raise --threshold to reduce\n\
                  false positives; lower it to favor recall.\n\n\
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
    let scanner = Scanner::builder().jobs(args.jobs).build()?;
    let results = scanner.score_files(&files)?;

    let mut stdout = BufWriter::new(std::io::stdout().lock());
    let mut failures = 0_usize;
    for (path, result) in files.iter().zip(results) {
        match result {
            Ok(score) => {
                let (verdict, codec) = classification(&score, args.threshold);
                writeln!(
                    stdout,
                    "{:.7}\t{}\t{}\t{}",
                    score.transcode_probability(),
                    verdict,
                    codec,
                    path.display()
                )?;
            }
            Err(error) => {
                failures += 1;
                eprintln!("lossprint: {}: {error}", path.display());
            }
        }
    }
    stdout.flush()?;
    if failures > 0 {
        bail!("could not scan {failures} file(s)")
    }
    Ok(())
}

fn classification(score: &TrackScore, threshold: f32) -> (&'static str, &'static str) {
    if score.transcode_probability() < threshold {
        return ("clean", "-");
    }
    ("transcode", most_likely_codec(score).as_str())
}

fn most_likely_codec(score: &TrackScore) -> Codec {
    score
        .codec_probabilities()
        .iter()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .expect("the fixed model has codec classes")
        .0
}

fn discover(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            if !has_supported_extension(input) {
                bail!("unsupported audio file: {}", input.display())
            }
            files.push(input.clone());
        } else if input.is_dir() {
            for entry in WalkDir::new(input) {
                let entry = entry.with_context(|| format!("could not walk {}", input.display()))?;
                if entry.file_type().is_file() && has_supported_extension(entry.path()) {
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
