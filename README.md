# lossprint

`lossprint` is a Rust command-line tool and library that scans WAV, AIFF, and
FLAC files for evidence that they were decoded from a lossy source.

```console
$ lossprint ~/Music
PROBABILITY  VERDICT    ENCODER     KBPS  PATH
0.9981214    transcode  mp3          192  "/Users/me/Music/suspect.flac"
0.0042187    clean      -              -  "/Users/me/Music/master.flac"
```

At the default `0.5` threshold, the model's false-positive rate on untouched
masters is **0.24%**. Across the evaluated bitrate bands it detected
**99.8–100% of MP3 files** and **95.6–100% of AAC files**, and for flagged
files it estimates the source encoder family and bitrate. See the
[lossprint 1.0 model card](https://huggingface.co/maxdj/lossprint) for the
full evaluation.

## Install

### Brew

On Apple Silicon macOS, install with Homebrew:

```bash
brew install maxdjohnson/tap/lossprint
```

### Binaries

Download Apple Silicon macOS, Linux (x86-64 or ARM64), and Windows (x86-64)
binaries from the [latest GitHub release](https://github.com/maxdjohnson/lossprint/releases/latest).

### Cargo

```bash
cargo install lossprint --features cli
```

## Use

Pass any number of files or directories. Directory scans are recursive and do
not follow symlinks.

```bash
lossprint ~/Music /Volumes/archive
lossprint --threshold 0.7 ~/Music
lossprint --jobs 4 ~/Music
lossprint -o jsonl ~/Music
```

Outputs a table by default; use -o jsonl for machine-readable output:

```console
$ lossprint -o jsonl ~/Music
{"transcode_probability":0.9981214,"verdict":"transcode","encoder":"mp3","bitrate_kbps":191.6,"path":"/Users/me/Music/suspect.flac"}
{"transcode_probability":0.0042187,"verdict":"clean","encoder":null,"bitrate_kbps":null,"path":"/Users/me/Music/master.flac"}
```

Each JSON object has `transcode_probability`, `verdict`, `encoder`,
`bitrate_kbps`, and `path`. `encoder` is the most likely source encoder
(`mp3`, `ffmpeg_aac`, `vorbis`, `opus`, `wma`, `mp2`, or `musepack`; the
current model reports every AAC encoder as `ffmpeg_aac`) and `bitrate_kbps`
its estimated bitrate; both are `null` for clean files. JSONL paths must be
valid UTF-8.

The CLI's default threshold is `0.5`. Raise it to reduce false positives, or
lower it to favor recall.

The CLI scans tracks in parallel. `--jobs 0`, the default, lets Rayon choose
the worker count. Set a smaller value to limit CPU and memory use.

Directory scans accept `.wav`, `.aif`, `.aiff`, and `.flac`. WAV and AIFF must
contain linear integer or floating-point PCM. Input must be mono or stereo,
use a sample rate from 8 to 384 kHz, and be at least 2 seconds long. The model
was evaluated on 44.1–192 kHz audio.

Tracks up to 14.5 seconds are scored whole. Longer tracks are scored on up to
six evenly spaced 14.5-second windows drawn from their middle 92%, and the
per-window outputs are averaged. Inference uses the pure-Rust, CPU-only tract
runtime on every platform; a full-length track costs about three seconds of
CPU time.

## Rust library

Add the crate to an application:

```bash
cargo add lossprint
```

The library has no default features; enable `cli` only when building the
command-line executable.

Keep a `Scanner` alive while processing multiple tracks so its model and
spectrogram transforms are reused.

```rust,no_run
use lossprint::{Encoder, Scanner};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scanner = Scanner::new()?;
    let score = scanner.score_file(File::open("track.flac")?)?;

    println!("P(transcode) = {:.3}", score.transcode_probability());
    println!(
        "P(mp3 | transcode) = {:.3}",
        score.encoder_probability(Encoder::Mp3),
    );

    // The library returns probabilities; the application chooses its threshold.
    if score.transcode_probability() >= 0.5 {
        let (encoder, probability) = score.most_likely_encoder();
        println!(
            "transcode from {encoder} ({probability:.3}) at about {:.0} kbps",
            score.bitrate_kbps()
        );
    }

    Ok(())
}
```

`Scanner::score_file` takes a `File` and returns probabilities plus a bitrate
estimate; the application chooses its own threshold.

A scanner can be shared across threads, so applications control track-level
concurrency themselves.

The crate contains the pinned 1.0 model and embeds it directly in programs
that use `lossprint`. Building from a published or vendored crate and runtime
scoring therefore need no separate model download. Scoring only requires
`&Scanner`.

## Build

Install Rust 1.97.1 or newer. A published crate already contains the model, so
it builds normally (and with `--offline` when the ordinary Rust dependencies
are available locally):

```bash
cargo build --release --features cli
```

A Git checkout deliberately does not contain the model. Stage the ignored,
checksum-verified model once before building or packaging that checkout:

```bash
python tools/fetch_model.py
cargo build --release --features cli
```

Cargo treats the intentionally untracked model as a package change. After
confirming the Git worktree is clean, maintainers can create or publish the
self-contained crate with `cargo package --locked --allow-dirty` or
`cargo publish --locked --allow-dirty`.
