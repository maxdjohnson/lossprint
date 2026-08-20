//! Scan lossless audio containers for evidence of an earlier lossy encode.

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use lossprint::Scanner;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const DEFAULT_THRESHOLD: f32 = 0.5;
const SUPPORTED_EXTENSIONS: [&str; 4] = ["aif", "aiff", "flac", "wav"];

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Detect lossy transcodes hiding in lossless audio files",
    after_help = "Scans WAV, AIFF, and FLAC files. The default output is an aligned\n\
                  table; use -o jsonl for machine-readable output. Raise --threshold\n\
                  to reduce false positives; lower it to favor recall.\n\n\
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

    /// Parallel workers; zero lets Rayon choose.
    #[arg(short, long, default_value_t = 0)]
    jobs: usize,

    /// Output format.
    #[arg(
        short = 'o',
        long = "output-format",
        value_enum,
        default_value = "table",
        value_name = "FORMAT"
    )]
    output: OutputFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    /// An aligned table for people.
    Table,
    /// One JSON object per successful file.
    Jsonl,
}

#[derive(serde::Serialize)]
struct OutputRow<'a> {
    transcode_probability: f32,
    verdict: &'static str,
    encoder: Option<&'static str>,
    path: &'a Path,
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
    let scanner = Scanner::new()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()
        .context("could not create worker pool")?;
    let results = pool.install(|| {
        files
            .par_iter()
            .map(|path| -> Result<_> {
                let source = File::open(path)?;
                Ok(scanner.score(source)?)
            })
            .collect::<Vec<_>>()
    });

    let mut rows = Vec::new();
    let mut failures = 0_usize;
    for (path, result) in files.iter().zip(results) {
        match result {
            Ok(score) => {
                let probability = score.transcode_probability();
                let (verdict, encoder) = if probability < args.threshold {
                    ("clean", None)
                } else {
                    let (encoder, _) = score.most_likely_encoder();
                    ("transcode", Some(encoder.as_str()))
                };
                if args.output == OutputFormat::Jsonl && path.to_str().is_none() {
                    failures += 1;
                    eprintln!(
                        "lossprint: {}: path is not valid UTF-8 for JSONL output",
                        path.display()
                    );
                    continue;
                }
                rows.push(OutputRow {
                    transcode_probability: probability,
                    verdict,
                    encoder,
                    path,
                });
            }
            Err(error) => {
                failures += 1;
                eprintln!("lossprint: {}: {error}", path.display());
            }
        }
    }
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    match args.output {
        OutputFormat::Table => write_table(&mut stdout, &rows)?,
        OutputFormat::Jsonl => write_jsonl(&mut stdout, &rows)?,
    }
    stdout.flush()?;
    if failures > 0 {
        bail!("could not scan {failures} file(s)")
    }
    Ok(())
}

fn write_table(writer: &mut impl Write, rows: &[OutputRow<'_>]) -> std::io::Result<()> {
    writeln!(
        writer,
        "{:<11}  {:<9}  {:<10}  PATH",
        "PROBABILITY", "VERDICT", "ENCODER"
    )?;
    for row in rows {
        let encoder = row.encoder.unwrap_or("-");
        writeln!(
            writer,
            "{probability:<11.7}  {verdict:<9}  {encoder:<10}  {path:?}",
            probability = row.transcode_probability,
            verdict = row.verdict,
            path = row.path,
        )?;
    }
    Ok(())
}

fn write_jsonl(writer: &mut impl Write, rows: &[OutputRow<'_>]) -> Result<()> {
    for row in rows {
        serde_json::to_writer(&mut *writer, row).context("could not write JSONL output")?;
        writeln!(writer)?;
    }
    Ok(())
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

fn has_supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_escapes_path_control_characters() {
        let rows = [OutputRow {
            transcode_probability: 0.25,
            verdict: "clean",
            encoder: None,
            path: Path::new("quoted\tpath\n.wav"),
        }];
        let mut output = Vec::new();

        write_table(&mut output, &rows).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert_eq!(output.lines().count(), 2);
        assert!(output.contains(r#""quoted\tpath\n.wav""#));
    }

    #[test]
    fn jsonl_escapes_paths_and_uses_null_for_a_clean_encoder() {
        let rows = [OutputRow {
            transcode_probability: 0.25,
            verdict: "clean",
            encoder: None,
            path: Path::new("quoted\t\"path.wav"),
        }];
        let mut output = Vec::new();

        write_jsonl(&mut output, &rows).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert_eq!(output.lines().count(), 1);
        let record: serde_json::Value = serde_json::from_str(output.trim_end()).unwrap();
        assert_eq!(record["transcode_probability"], 0.25);
        assert_eq!(record["verdict"], "clean");
        assert!(record["encoder"].is_null());
        assert_eq!(record["path"], "quoted\t\"path.wav");
    }
}
