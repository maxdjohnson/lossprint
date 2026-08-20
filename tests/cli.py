"""Exercise the packaged CLI against the release-hosted integration fixtures."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED = {
    "musdb18-hq-1.wav": (0.0006546, "clean", None),
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
        check=False,
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


def assert_jsonl_output(output: str, fixtures: Path) -> None:
    lines = output.splitlines()
    if len(lines) != len(EXPECTED):
        fail(f"expected {len(EXPECTED)} JSONL records, got {len(lines)}")

    seen = set()
    names = []
    required_fields = {"transcode_probability", "verdict", "encoder", "path"}
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSONL record: {error}")
        if not isinstance(record, dict):
            fail(f"JSONL record is not an object: {record!r}")
        missing = required_fields - record.keys()
        if missing:
            fail(f"JSONL record is missing fields {sorted(missing)!r}: {record!r}")

        probability = record["transcode_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            fail(f"invalid probability: {probability!r}")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            fail(f"probability is outside [0, 1]: {probability!r}")

        path = record["path"]
        if not isinstance(path, str):
            fail(f"path is not a string: {path!r}")
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if name not in EXPECTED:
            fail(f"unexpected fixture in output: {name!r}")
        if name in seen:
            fail(f"duplicate fixture in output: {name!r}")
        if Path(path).resolve() != (fixtures / name).resolve():
            fail(f"unexpected fixture path: {path!r}")
        expected_probability, expected_verdict, expected_encoder = EXPECTED[name]
        if abs(probability - expected_probability) > PROBABILITY_TOLERANCE:
            fail(
                f"unexpected probability for {name}: {probability:.7f}; "
                f"expected {expected_probability:.7f} ± {PROBABILITY_TOLERANCE}"
            )
        actual = (record["verdict"], record["encoder"])
        expected = (expected_verdict, expected_encoder)
        if actual != expected:
            fail(f"unexpected classification for {name}: {actual!r}")
        seen.add(name)
        names.append(name)

    if names != sorted(EXPECTED):
        fail(f"output is not sorted by path: {names!r}")


def assert_table_output(output: str) -> None:
    lines = output.splitlines()
    if len(lines) != len(EXPECTED) + 1:
        fail(
            f"expected a header and {len(EXPECTED)} table rows, got {len(lines)} lines"
        )
    if lines[0].split() != ["PROBABILITY", "VERDICT", "ENCODER", "PATH"]:
        fail(f"unexpected table header: {lines[0]!r}")

    names = []
    for line in lines[1:]:
        matches = [name for name in EXPECTED if name in line]
        if len(matches) != 1:
            fail(f"table row does not identify exactly one fixture: {line!r}")
        name = matches[0]
        _, expected_verdict, expected_encoder = EXPECTED[name]
        encoder = expected_encoder or "-"
        fields = line.split(maxsplit=3)
        if len(fields) != 4 or fields[1:3] != [expected_verdict, encoder]:
            fail(f"unexpected table classification for {name}: {line!r}")
        names.append(name)

    if names != sorted(EXPECTED):
        fail(f"table output is not sorted by path: {names!r}")


def assert_partial_failure(binary: Path, fixtures: Path) -> None:
    short_file = Path(__file__).parent / "fixtures/audio/pcm16.wav"
    result = subprocess.run(
        [binary, "-o", "jsonl", fixtures, short_file],
        capture_output=True,
        check=False,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        fail("mixed valid/invalid scan unexpectedly succeeded")
    assert_jsonl_output(result.stdout, fixtures)
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
        assert_table_output(invoke(binary, [fixtures]))
        assert_jsonl_output(invoke(binary, ["-o", "jsonl", fixtures]), fixtures)
        assert_partial_failure(binary, fixtures)


if __name__ == "__main__":
    main()
