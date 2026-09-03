"""Exercise the packaged CLI against the release-hosted integration fixtures."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

# name -> (transcode probability, verdict, encoder, bitrate in kbps)
EXPECTED = {
    "musdb18-hq-1.wav": (0.0000129, "clean", None, None),
    "musdb18-hq-1.mp3.wav": (1.0000000, "transcode", "mp3", 323.9),
}
PROBABILITY_TOLERANCE = 5e-4
BITRATE_TOLERANCE = 2.0
JSON_FIELDS = {"transcode_probability", "verdict", "encoder", "bitrate_kbps", "path"}
TABLE_HEADER = "PROBABILITY  VERDICT    ENCODER     KBPS  PATH"


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


def run(binary: Path, arguments: list[str | Path]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [binary, *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result


def invoke(binary: Path, arguments: list[str | Path]) -> str:
    result = run(binary, arguments)
    if result.returncode != 0:
        fail(f"packaged binary exited with status {result.returncode}")
    return result.stdout


def assert_probability(actual: object, expected: float, name: str) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not 0.0 <= actual <= 1.0
        or not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=PROBABILITY_TOLERANCE,
        )
    ):
        fail(f"unexpected probability for {name}: {actual!r}; expected {expected:.7f}")


def assert_bitrate(actual: object, expected: float | None, name: str) -> None:
    if expected is None:
        if actual is not None:
            fail(f"unexpected bitrate for clean {name}: {actual!r}")
        return
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=BITRATE_TOLERANCE)
    ):
        fail(f"unexpected bitrate for {name}: {actual!r}; expected {expected:.1f}")


def assert_jsonl_output(output: str, fixtures: Path) -> None:
    try:
        records = [json.loads(line) for line in output.splitlines()]
    except json.JSONDecodeError as error:
        fail(f"invalid JSONL record: {error}")

    expected = sorted(EXPECTED.items())
    if len(records) != len(expected):
        fail(f"expected {len(expected)} JSONL records, got {len(records)}")

    for record, (name, values) in zip(records, expected, strict=True):
        if not isinstance(record, dict):
            fail(f"JSONL record is not an object: {record!r}")
        if set(record) != JSON_FIELDS:
            fail(f"unexpected JSONL fields: {sorted(record)!r}")

        path = record["path"]
        if (
            not isinstance(path, str)
            or Path(path).resolve() != (fixtures / name).resolve()
        ):
            fail(f"unexpected fixture path: {path!r}")

        probability, verdict, encoder, bitrate = values
        assert_probability(record["transcode_probability"], probability, name)
        if (record["verdict"], record["encoder"]) != (verdict, encoder):
            fail(f"unexpected classification for {name}: {record!r}")
        assert_bitrate(record["bitrate_kbps"], bitrate, name)


def assert_table_output(output: str, fixtures: Path) -> None:
    lines = output.splitlines()
    expected = sorted(EXPECTED.items())
    if len(lines) != len(expected) + 1:
        fail(f"expected {len(expected) + 1} table lines, got {len(lines)}")
    if lines[0] != TABLE_HEADER:
        fail(f"unexpected table header: {lines[0]!r}")

    for line, (name, values) in zip(lines[1:], expected, strict=True):
        fields = line.split(maxsplit=4)
        if len(fields) != 5:
            fail(f"unexpected table row: {line!r}")

        probability, verdict, encoder, bitrate = values
        try:
            actual_probability = float(fields[0])
        except ValueError:
            fail(f"invalid table probability for {name}: {fields[0]!r}")
        assert_probability(actual_probability, probability, name)
        if bitrate is None:
            bitrate_text = "-"
        else:
            try:
                assert_bitrate(float(fields[3]), bitrate, name)
            except ValueError:
                fail(f"invalid table bitrate for {name}: {fields[3]!r}")
            bitrate_text = fields[3]

        expected_line = (
            f"{actual_probability:<11.7f}  {verdict:<9}  {(encoder or '-'):<10}  "
            f"{bitrate_text:>4}  {json.dumps(str(fixtures / name), ensure_ascii=False)}"
        )
        if line != expected_line:
            fail(f"unexpected table row for {name}: {line!r}")


def assert_partial_failure(binary: Path, fixtures: Path) -> None:
    short_file = Path(__file__).parent / "fixtures/audio/pcm16.wav"
    result = run(binary, ["-o", "jsonl", fixtures, short_file])
    if result.returncode == 0:
        fail("mixed valid/invalid scan unexpectedly succeeded")
    assert_jsonl_output(result.stdout, fixtures)
    for message in (
        str(short_file),
        "audio is shorter than the 2-second minimum",
        "could not scan 1 file(s)",
    ):
        if message not in result.stderr:
            fail(f"missing partial-failure diagnostic: {message!r}")


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
        assert_table_output(invoke(binary, [fixtures]), fixtures)
        assert_jsonl_output(invoke(binary, ["-o", "jsonl", fixtures]), fixtures)
        assert_partial_failure(binary, fixtures)


if __name__ == "__main__":
    main()
