# lossprint

`lossprint` is a Rust library and command-line tool that scans WAV, AIFF, and
FLAC files for evidence that they were decoded from a lossy source before being
saved in a lossless container.

```console
$ lossprint ~/Music
0.9981214	transcode	mp3	/Users/me/Music/suspect.flac
0.0042187	clean	-	/Users/me/Music/master.flac
```

Each stdout row contains the transcode probability, verdict, predicted codec,
and path separated by tabs. The codec is `-` for clean files because that
prediction is meaningful only for transcodes. Errors go to stderr.

## Install

On Apple Silicon macOS, install with Homebrew:

```bash
brew install maxdjohnson/tap/lossprint
```

Or install the Rust command-line tool from crates.io:

```bash
cargo install lossprint
```

For Linux (x86-64 or ARM64) and Windows (x86-64), download the archive for your
platform from the [latest GitHub release](https://github.com/maxdjohnson/lossprint/releases/latest).

## Rust library

Add the crate to an application:

```bash
cargo add lossprint
```

Keep a `Scanner` alive while processing multiple tracks so its model, worker
pool, and spectrogram transforms are reused.

```rust,no_run
use lossprint::Scanner;

fn main() -> lossprint::Result<()> {
    let mut scanner = Scanner::new()?;
    let score = scanner.score_file("track.flac")?;

    println!("P(transcode) = {:.3}", score.prob_transcode);
    println!("P(mp3 | transcode) = {:.3}", score.prob_codec.mp3);
    println!("P(aac | transcode) = {:.3}", score.prob_codec.aac);

    // The library returns probabilities; the application chooses its threshold.
    if score.prob_transcode >= 0.5 {
        println!("transcode");
    }

    Ok(())
}
```

`score.prob_codec` also has `aac_at`, `fdk_aac`, `vorbis`, and `opus` fields.
These codec probabilities are conditional on the track being a transcode.

`Scanner::score_files` prepares files in parallel, batches model inference, and
returns results in input order. A decoding error affects only that file's inner
result; a shared worker or inference failure is returned by the outer result.
Configure parallelism and batch size with `Scanner::builder()`.

The default `bundled-model` feature downloads the pinned v0.5 model during the
build, verifies its SHA-256 checksum, caches it under Cargo's home directory,
and embeds it in the program. Runtime scoring therefore needs no network. To
supply the model yourself and avoid that model download, disable default
features and construct the scanner with `Scanner::from_model_bytes` or
`Scanner::from_model_file`:

```toml
[dependencies]
lossprint = { version = "0.4", default-features = false }
```

The scanner requires mutable access while scoring. Put it behind your own lock
if several threads need to share one instance.

## Build

Install Rust 1.97.1 or newer and `curl`, then run:

```bash
cargo build --release
```

Set `LOSSPRINT_MODEL_PATH` to the published v0.5 ONNX file for an offline build:

```bash
LOSSPRINT_MODEL_PATH=/absolute/path/model.onnx cargo build --release
```

The supplied path is intentionally not checksum-restricted, which permits
applications to test compatible custom models.

## Use

Pass any number of files or directories. Directory scans are recursive and do
not follow symlinks.

```bash
lossprint ~/Music /Volumes/archive
lossprint --threshold 0.7 ~/Music
lossprint --jobs 4 ~/Music
lossprint --batch-size 4 ~/Music
```

The CLI's default threshold is `0.5`. Raise it to reduce false positives, or
lower it to favor recall.

`--jobs 0`, the default, uses one worker per logical core. Set a smaller value
to limit CPU and memory use.

`--batch-size N` sends `N` tracks' worth of windows through the model per
forward pass, where `N` is 1–8. It defaults to 1, except on Apple Silicon scans
of at least 16 tracks, where it defaults to 8 for the best throughput on large
libraries.

Directory scans accept `.wav`, `.aif`, `.aiff`, and `.flac`. WAV and
AIFF must contain linear integer or floating-point PCM. Input must be mono or
stereo, use a sample rate from 8 to 384 kHz, and be at least two seconds long.
Rates that do not produce the model's 173 STFT frames are rejected.

Only the first 20 seconds of each track are decoded. Scores can vary slightly
between CPU, Core ML, and DirectML inference, so callers should apply their own
threshold rather than comparing exact probability values across platforms.
