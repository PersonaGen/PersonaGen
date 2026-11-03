"""Run real evaluation on fictional data; no model, API, or human data required."""
from pathlib import Path
import numpy as np

from popalign.evaluation.runner import run_bigfive
from popalign.io import write_jsonl


def main():
    output = Path(__file__).resolve().parents[1] / "outputs" / "demo"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    traits = ["EXT", "AGR", "CSN", "EST", "OPN"]
    reference = np.clip(rng.normal(3, 0.8, size=(64, 5)), 1, 5)
    shifted = np.clip(reference + 0.6, 1, 5)
    for name, values in [("synthetic_reference", reference), ("synthetic_shifted", shifted)]:
        write_jsonl(output / f"{name}.jsonl", [dict(zip(traits, map(float, row))) for row in values])
    print("SYNTHETIC DEMO ONLY — these are not human data or paper results.")
    run_bigfive({
        "data": {"human_path": str(output / "synthetic_reference.jsonl"),
                 "origin_path": str(output / "synthetic_shifted.jsonl")},
        "output": {"dir": str(output)},
        "quantile_warp_csn": False,
    })


if __name__ == "__main__":
    main()
