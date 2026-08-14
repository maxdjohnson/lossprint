These tiny files exercise the native decoder without constructing container
bytes inside the tests.

- `pcm16.{wav,aiff,flac}` contain the same four stereo frames at 44.1 kHz:
  `[-32768, 32767, -16384, 16384, -1, 1, 0, 12345]`.
- `float32-stereo.wav` contains two stereo frames at 48 kHz:
  `[1.25, -1.5, 0.25, -0.75]`.
- `float32-mono.wav` contains two mono frames at 32 kHz:
  `[0.5, -0.25]`.
