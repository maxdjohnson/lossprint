# Third-party software

## lossprint model

The default build embeds the
[lossprint v0.5 ONNX model](https://huggingface.co/maxdj/lossprint/tree/ae0526389bb0fa3d8cfed08ecfeae0bd7930a8c8),
released under the MIT License. Its SHA-256 digest is
`950836de4cf5c9265aea8c7f9cf7b0d8c417b7c1780272f1c00a5d8039e83540`.

## Symphonia

`lossprint` statically links
[Symphonia 0.6.1](https://github.com/pdeljanov/Symphonia/tree/v0.6.1)
under the [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/).
The binary uses the unmodified source from that tag to decode WAV, AIFF, and
FLAC files.

## ONNX Runtime

`lossprint` statically links
[ONNX Runtime 1.28.0](https://github.com/microsoft/onnxruntime/tree/v1.28.0)
under the MIT License through the
[`ort` 2.0.0-rc.13](https://github.com/pykeio/ort/tree/v2.0.0-rc.13) Rust
bindings, licensed under MIT or Apache-2.0. ONNX Runtime is copyright Microsoft
Corporation.

This distribution uses the MIT option for both components:

> Copyright (c) Microsoft Corporation. All rights reserved.
>
> Copyright (c) 2023-2026 pyke.io
>
> Copyright (c) 2020 Nicolas Bigaouette
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The complete dependency versions are recorded in `Cargo.lock`.

## DirectML (Windows)

The Windows release also redistributes
[Microsoft DirectML 1.15.4](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4)
for use by ONNX Runtime. DirectML is licensed under separate Microsoft Software
License Terms rather than the project MIT license. The exact terms and supplier
notices are included in the Windows archive as `DirectML_LICENSE.txt` and
`DirectML_ThirdPartyNotices.txt`.
