#!/usr/bin/env python3
"""Exercise the packaged CLI against the release-hosted integration fixtures."""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED = {
    "musdb18-hq-1.wav": (0.0006546, "clean", "-"),
    "musdb18-hq-1.mp3.wav": (0.9997401, "transcode", "mp3"),
}
PROBABILITY_TOLERANCE = 5e-4


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


def invoke(binary: Path, arguments: list[str | Path], expected_status: int = 0) -> str:
    result = subprocess.run(
        [binary, *arguments],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != expected_status:
        fail(
            f"packaged binary exited with status {result.returncode}; "
            f"expected {expected_status}"
        )
    return result.stdout


def assert_output(output: str, fixtures: Path) -> None:
    rows = output.splitlines()
    if len(rows) != len(EXPECTED):
        fail(f"expected {len(EXPECTED)} output rows, got {len(rows)}")

    seen = set()
    names = []
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
        expected_probability, expected_verdict, expected_codec = EXPECTED[name]
        if abs(probability - expected_probability) > PROBABILITY_TOLERANCE:
            fail(
                f"unexpected probability for {name}: {probability:.7f}; "
                f"expected {expected_probability:.7f} ± {PROBABILITY_TOLERANCE}"
            )
        actual = (fields[1], fields[2])
        expected = (expected_verdict, expected_codec)
        if actual != expected:
            fail(f"unexpected classification for {name}: {actual!r}")
        seen.add(name)
        names.append(name)

    if names != sorted(EXPECTED):
        fail(f"output is not sorted by path: {names!r}")


def assert_partial_failure(binary: Path, fixtures: Path) -> None:
    short_file = Path(__file__).parent / "fixtures/audio/pcm16.wav"
    result = subprocess.run(
        [binary, fixtures, short_file],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        fail("mixed valid/invalid scan unexpectedly succeeded")
    assert_output(result.stdout, fixtures)
    if str(short_file) not in result.stderr:
        fail("per-file diagnostic did not identify the short input")
    if "audio is shorter than 0.5 seconds" not in result.stderr:
        fail("per-file diagnostic did not explain the short input")
    if "could not scan 1 file(s)" not in result.stderr:
        fail("aggregate failure did not report one failed file")


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
        assert_output(invoke(binary, [fixtures]), fixtures)
        assert_partial_failure(binary, fixtures)


if __name__ == "__main__":
    main()
