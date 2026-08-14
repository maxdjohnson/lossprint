#!/usr/bin/env python3
"""Exercise the packaged CLI against the release-hosted integration fixtures."""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED = {
    "musdb18-hq-1.wav": ("clean", "-"),
    "musdb18-hq-1.mp3.wav": ("transcode", "mp3"),
}


def fail(message: str) -> None:
    raise SystemExit(f"CLI integration failed: {message}")


def download_fixtures(destination: Path) -> None:
    command = [
        "gh",
        "release",
        "download",
        "ci-v1",
        "--repo",
        "maxdjohnson/lossprint",
        "--pattern",
        "musdb18-hq-1.wav",
        "--pattern",
        "musdb18-hq-1.mp3.wav",
        "--dir",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        fail("gh is not installed")
    except subprocess.CalledProcessError as error:
        fail(f"fixture download exited with status {error.returncode}")


def invoke(binary: Path, fixtures: Path) -> str:
    result = subprocess.run(
        [binary, fixtures],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        fail(f"packaged binary exited with status {result.returncode}")
    return result.stdout


def assert_output(output: str, fixtures: Path) -> None:
    rows = output.splitlines()
    if len(rows) != len(EXPECTED):
        fail(f"expected {len(EXPECTED)} output rows, got {len(rows)}")

    seen = set()
    for row in rows:
        fields = row.split("\t")
        if len(fields) != 4:
            fail(f"expected four tab-separated fields: {row!r}")
        try:
            probability = float(fields[0])
        except ValueError:
            fail(f"invalid probability: {fields[0]!r}")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            fail(f"probability is outside [0, 1]: {fields[0]!r}")

        name = fields[3].replace("\\", "/").rsplit("/", 1)[-1]
        if name not in EXPECTED:
            fail(f"unexpected fixture in output: {name!r}")
        if name in seen:
            fail(f"duplicate fixture in output: {name!r}")
        if Path(fields[3]).resolve() != (fixtures / name).resolve():
            fail(f"unexpected fixture path: {fields[3]!r}")
        actual = (fields[1], fields[2])
        if actual != EXPECTED[name]:
            fail(f"unexpected classification for {name}: {actual!r}")
        seen.add(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        fail(f"binary does not exist: {binary}")

    with tempfile.TemporaryDirectory(prefix="lossprint-ci-") as directory:
        fixtures = Path(directory)
        download_fixtures(fixtures)
        assert_output(invoke(binary, fixtures), fixtures)


if __name__ == "__main__":
    main()
