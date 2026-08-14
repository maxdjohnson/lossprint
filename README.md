# lossprint

`lossprint` scans WAV, AIFF, and FLAC files for evidence that they were decoded
from a lossy source before being saved in a lossless container.

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

For Linux (x86-64 or ARM64) and Windows (x86-64), download the archive for your
platform from the [latest GitHub release](https://github.com/maxdjohnson/lossprint/releases/latest).

## Build

Install Rust 1.97.1 or newer and `curl`, then run:

```bash
cargo build --release
```

Set `LOSSPRINT_MODEL_PATH` to the published v0.5 ONNX file for an offline build:

```bash
LOSSPRINT_MODEL_PATH=/absolute/path/model.onnx cargo build --release
```

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
