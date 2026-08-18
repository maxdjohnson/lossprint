//! Scan lossless audio containers for evidence of an earlier lossy encode.

use anyhow::{bail, Context, Result};
use clap::Parser;
use lossprint::{is_supported_file, CodecProbabilities, Scanner, TrackScore, MAX_BATCH_TRACKS};
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

    /// Tracks' worth of windows per model forward pass (1-8); defaults to 8 for scans of at least 16 tracks.
    #[arg(
        long,
        value_name = "TRACKS",
        value_parser = clap::value_parser!(u8).range(1..=MAX_BATCH_TRACKS as i64)
    )]
    batch_size: Option<u8>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.threshold.is_finite() || !(0.0..=1.0).contains(&args.threshold) {
        bail!("--threshold must be between 0 and 1")
    }

    let files = discover(&args.paths)?;
    if files.is_empty() {
        bail!("no WAV, AIFF, or FLAC files found")
    }
    let mut builder = Scanner::builder().jobs(args.jobs);
    if let Some(batch_size) = args.batch_size {
        builder = builder.batch_size(batch_size);
    }
    let mut scanner = builder.build()?;
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
                    score.prob_transcode,
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

fn classification(score: &TrackScore, threshold: f32) -> (&'static str, &'static str) {
    if score.prob_transcode < threshold {
        return ("clean", "-");
    }
    ("transcode", most_likely_codec(&score.prob_codec))
}

fn most_likely_codec(probabilities: &CodecProbabilities) -> &'static str {
    let candidates = [
        ("mp3", probabilities.mp3),
        ("aac", probabilities.aac),
        ("aac_at", probabilities.aac_at),
        ("fdk_aac", probabilities.fdk_aac),
        ("vorbis", probabilities.vorbis),
        ("opus", probabilities.opus),
        ("mp2", probabilities.mp2),
        ("wma", probabilities.wma),
        ("musepack", probabilities.musepack),
    ];
    let mut best = candidates[0];
    for candidate in candidates.into_iter().skip(1) {
        if candidate.1 > best.1 {
            best = candidate;
        }
    }
    best.0
}

fn discover(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            if !is_supported_file(input) {
                bail!("unsupported audio file: {}", input.display())
            }
            files.push(input.clone());
        } else if input.is_dir() {
            for entry in WalkDir::new(input) {
                let entry = entry.with_context(|| format!("could not walk {}", input.display()))?;
                if entry.file_type().is_file() && is_supported_file(entry.path()) {
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

    fn score(prob_transcode: f32, prob_codec: CodecProbabilities) -> TrackScore {
        TrackScore {
            prob_transcode,
            prob_codec,
        }
    }

    fn codecs(values: [f32; 9]) -> CodecProbabilities {
        let [mp3, aac, aac_at, fdk_aac, vorbis, opus, mp2, wma, musepack] = values;
        CodecProbabilities {
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

    #[test]
    fn batch_size_is_bounded() {
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "1", "audio.flac"]).is_ok());
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "8", "audio.flac"]).is_ok());
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "0", "audio.flac"]).is_err());
        assert!(Args::try_parse_from(["lossprint", "--batch-size", "9", "audio.flac"]).is_err());
    }

    #[test]
    fn default_threshold_is_half() {
        let args = Args::try_parse_from(["lossprint", "audio.flac"]).unwrap();
        assert_eq!(args.threshold, 0.5);
    }

    #[test]
    fn classification_keeps_cli_threshold_and_codec_behavior() {
        let clean = score(0.49, codecs([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(classification(&clean, DEFAULT_THRESHOLD), ("clean", "-"));

        let transcode = score(0.5, codecs([0.1, 0.2, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7, 0.9]));
        assert_eq!(
            classification(&transcode, DEFAULT_THRESHOLD),
            ("transcode", "musepack")
        );

        let tie = score(0.9, codecs([0.4; 9]));
        assert_eq!(
            classification(&tie, DEFAULT_THRESHOLD),
            ("transcode", "mp3")
        );
    }
}
