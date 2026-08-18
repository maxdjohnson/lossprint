# Third-party software

## lossprint model

The default build embeds the
[lossprint v0.6 ONNX model](https://huggingface.co/maxdj/lossprint/tree/v0.6),
released under the MIT License. Its SHA-256 digest is
`1ba4997ecc1cd3379767017abc32140f883c79e74c0d2b6c1ee6628fbd4549e4`.

## Symphonia

`lossprint` statically links
[Symphonia 0.6.1](https://github.com/pdeljanov/Symphonia/tree/v0.6.1)
under the [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/).
The binary uses the unmodified source from that tag to decode WAV, AIFF, and
FLAC files.

## tract

`lossprint` statically links
[`tract-onnx` 0.23.4](https://github.com/snipsco/tract/tree/v0.23.4), a CPU-only
ONNX inference engine licensed under MIT or Apache-2.0. This distribution uses
the MIT option. The complete dependency versions are recorded in `Cargo.lock`.
