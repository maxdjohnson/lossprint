# Third-party software

## lossprint model

The build embeds the
[lossprint v0.7 ONNX model](https://huggingface.co/maxdj/lossprint/tree/v0.7),
released under the MIT License. Its SHA-256 digest is
`33c74bde418b8330f7e67222afb2ab53706c136281bddd19ec0870b81ddce89a`.

## Symphonia

`lossprint` statically links
[Symphonia](https://github.com/pdeljanov/Symphonia) under the
[Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/). The binary uses
unmodified source to decode WAV, AIFF, and FLAC files.

Releases are built with `--locked`, so the exact version is the one recorded in
the `Cargo.lock` distributed with this file. Its source is the matching `v<version>`
tag at <https://github.com/pdeljanov/Symphonia/tags>.

## tract

`lossprint` statically links
[`tract-onnx`](https://github.com/snipsco/tract), a CPU-only ONNX inference
engine licensed under MIT or Apache-2.0. This distribution uses the MIT option.
The complete dependency versions are recorded in `Cargo.lock`.
