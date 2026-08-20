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

Keep a `Scanner` alive while processing multiple tracks so its model and
spectrogram transforms are reused.

```rust,no_run
use lossprint::{Codec, Scanner};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scanner = Scanner::new()?;
    let score = scanner.score(File::open("track.flac")?)?;

    println!("P(transcode) = {:.3}", score.transcode_probability());
    println!(
        "P(mp3 | transcode) = {:.3}",
        score.codec_probability(Codec::Mp3),
    );

    let (codec, probability) = score.most_likely_codec();
    println!("most likely source: {codec} ({probability:.3})");

    // The library returns probabilities; the application chooses its threshold.
    if score.transcode_probability() >= 0.5 {
        println!("transcode");
    }

    Ok(())
}
```

`Scanner::score` accepts any seekable `MediaSource`, including `File`, `Cursor`,
and other `Read + Seek + Send + Sync` types. The caller opens the source, so
file-opening errors remain outside lossprint's scoring API. Read or seek
failures after scoring starts are reported as audio errors.

`Scanner::new` returns `InitializationError`; `Scanner::score` returns either
`ScoreError::Audio(audio::Error)` or `ScoreError::Inference`. Decoder-specific
sources inside `audio::Error` are boxed, preserving their diagnostic chain
without making the decoder library part of lossprint's public API.
`lossprint::Error` and `lossprint::Result` combine initialization and scoring
errors for code that has already acquired its media source.

`Codec` covers all nine codec and encoder classes and implements `Display`.
Use `score.codec_probabilities()` to visit each class and probability in model
order. These probabilities are conditional on the track being a transcode.

A scanner can be shared across threads, so applications control track-level
concurrency themselves.

The build downloads the pinned v0.6 model, verifies its SHA-256 checksum,
caches it under Cargo's home directory, and embeds it in the program. Runtime
scoring therefore needs no network. Scoring only requires `&Scanner`.

## Build

Install Rust 1.97.1 or newer and `curl`, then run:

```bash
cargo build --release
```

## Use

Pass any number of files or directories. Directory scans are recursive and do
not follow symlinks.

```bash
lossprint ~/Music /Volumes/archive
lossprint --threshold 0.7 ~/Music
lossprint --jobs 4 ~/Music
```

The CLI's default threshold is `0.5`. Raise it to reduce false positives, or
lower it to favor recall.

The CLI scans tracks in parallel. `--jobs 0`, the default, lets Rayon choose
the worker count. Set a smaller value to limit CPU and memory use.

Directory scans accept `.wav`, `.aif`, `.aiff`, and `.flac`. WAV and
AIFF must contain linear integer or floating-point PCM. Input must be mono or
stereo, use a sample rate from 8 to 384 kHz, and be at least 0.5 seconds long.

Only the first 20 seconds are decoded. The scanner scores up to sixteen evenly
spaced 0.5-second windows in one model call and pools their probabilities
geometrically. Inference uses the pure-Rust, CPU-only tract runtime on every
platform.
