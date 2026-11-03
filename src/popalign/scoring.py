"""Strict IPIP-50 item scoring for release workflows, not historical preprocessing.

Polarity follows https://ipip.ori.org/newBigFive5broadKey.htm. Item identifiers
follow examples/ipip50_questions.jsonl. EST means emotional stability (higher is
more stable), not neuroticism. All fifty 1–5 answers are required.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

TRAITS = ("EXT", "AGR", "CSN", "EST", "OPN")
REVERSED = {
    "EXT": {2, 4, 6, 8, 10},
    "AGR": {1, 3, 5, 7},
    "CSN": {2, 4, 6, 8},
    "EST": {1, 3, 5, 6, 7, 8, 9, 10},
    "OPN": {2, 4, 6},
}
ID_FIELDS = ("persona_unique_id", "persona_id", "unique_id", "id", "iteration_id")


def score_row(row):
    """Return means on the 1–5 scale; invalid or partial inventories raise."""
    result = {key: row[key] for key in ID_FIELDS if key in row}
    for trait in TRAITS:
        values = []
        for i in range(1, 11):
            key = f"{trait}{i}"
            value = row.get(key)
            if isinstance(value, bool) or value is None:
                raise ValueError(f"{key}: expected a complete inventory of numeric 1–5 responses")
            try:
                value = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key}: expected a numeric 1–5 response, got {value!r}") from exc
            if not math.isfinite(value) or value not in (1., 2., 3., 4., 5.):
                raise ValueError(f"{key}: expected an integer response from 1 to 5")
            values.append(6 - value if i in REVERSED[trait] else value)
        result[trait] = sum(values) / 10
    return result


def _rows(path, fmt):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if fmt == "jsonl":
            for line, text in enumerate(handle, 1):
                if not text.strip():
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line}: expected a JSON object")
                yield line, row
        else:
            reader = csv.DictReader(handle, delimiter="\t" if fmt == "tsv" else ",")
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV/TSV header is empty or duplicated")
            for line, row in enumerate(reader, 2):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"line {line}: malformed row")
                yield line, row


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score complete IPIP-50 responses into five 1–5 trait means. New release utility, not historical paper preprocessing.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-format", choices=["jsonl", "csv", "tsv"], default="jsonl")
    parser.add_argument("--output-format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args(argv)
    temporary = None
    try:
        metadata_path = args.output.with_name(args.output.name + ".metadata.json")
        if args.output.exists() or metadata_path.exists():
            raise ValueError("Output or metadata file already exists; choose a new output path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = None
            if args.output_format == "csv":
                writer = csv.DictWriter(handle, fieldnames=[*ID_FIELDS, *TRAITS])
                writer.writeheader()
            count = 0
            for line, row in _rows(args.input, args.input_format):
                try:
                    scored = score_row(row)
                except ValueError as exc:
                    raise ValueError(f"input line {line}: {exc}") from exc
                if writer is None:
                    handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
                else:
                    writer.writerow(scored)
                count += 1
            if count == 0:
                raise ValueError("Input contains no response rows")
        os.replace(temporary, args.output)
        temporary = None
        digest = hashlib.sha256()
        with args.input.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        metadata = {
            "implementation": "new_release_ipip50_scorer_not_historical_preprocessing",
            "source": "https://ipip.ori.org/newBigFive5broadKey.htm",
            "input_sha256": digest.hexdigest(), "input_format": args.input_format,
            "output_format": args.output_format, "rows": count,
            "aggregation": "Mean of 10 keyed items per trait; reversed item = 6 - response",
            "traits": list(TRAITS), "EST_direction": "higher = emotional stability",
            "reverse_keyed_items": {t: sorted(items) for t, items in REVERSED.items()},
            "invalid_policy": "Reject incomplete/noninteger/out-of-range responses; no imputation or row filtering",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Scored {count} complete IPIP-50 inventories into {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"scoring: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
