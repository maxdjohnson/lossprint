"""Stage the pinned model for builds from a Git checkout."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/maxdj/lossprint/resolve/"
    "c6ca3dd209e39c21b8ce48e235f759ac931cf914/model.onnx?download=true"
)
MODEL_SHA256 = "33c74bde418b8330f7e67222afb2ab53706c136281bddd19ec0870b81ddce89a"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.onnx"
CHUNK_SIZE = 1024 * 1024


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def fetch() -> None:
    if MODEL_PATH.is_file() and digest(MODEL_PATH) == MODEL_SHA256:
        print(f"lossprint: model is already staged at {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "lossprint-model-fetch"}
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=MODEL_PATH.parent,
            prefix=".model.",
            suffix=".onnx",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            checksum = hashlib.sha256()
            print(f"lossprint: downloading {MODEL_URL}")
            with urllib.request.urlopen(request, timeout=120) as response:
                while chunk := response.read(CHUNK_SIZE):
                    temporary.write(chunk)
                    checksum.update(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        actual = checksum.hexdigest()
        if actual != MODEL_SHA256:
            raise RuntimeError(
                f"model checksum mismatch: expected {MODEL_SHA256}, got {actual}"
            )
        os.replace(temporary_path, MODEL_PATH)
        temporary_path = None
        print(f"lossprint: staged model at {MODEL_PATH}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    try:
        fetch()
    except (OSError, RuntimeError) as error:
        print(f"lossprint: could not stage model: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
