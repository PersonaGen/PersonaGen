"""Config-driven evaluation runners."""

from __future__ import annotations

from ..config import Config, require, section


def run_bigfive(config: Config) -> None:
    from popalign.evaluation.bigfive_metrics import main
    import sys

    data = section(config, "data")
    output = section(config, "output")
    args = [
        "bigfive_evaluation",
        "--human_path",
        require(data, "human_path"),
        "--origin_path",
        require(data, "origin_path"),
        "--out_dir",
        require(output, "dir"),
    ]
    if data.get("resampled_jsonl"):
        args.extend(["--resampled_jsonl", data["resampled_jsonl"]])
    if data.get("baselines"):
        args.extend(["--baselines", data["baselines"]])
    if config.get("quantile_warp_csn", False):
        print("[WARNING] CSN quantile warping uses the evaluation reference; report this as a reference-calibrated comparison.")
        args.append("--quantile_warp_csn")

    old_argv = sys.argv
    try:
        sys.argv = args
        main()
    finally:
        sys.argv = old_argv


def run_ood(config: Config) -> None:
    from popalign.evaluation.ood_metrics import main
    import sys

    data = section(config, "data")
    output = section(config, "output")
    args = [
        "other_evaluation",
        "--human_dir",
        require(data, "human_dir"),
        "--agent_dir",
        require(data, "agent_dir"),
        "--out_dir",
        require(output, "dir"),
        "--baselines",
        data.get("baselines", ""),
    ]

    old_argv = sys.argv
    try:
        sys.argv = args
        main()
    finally:
        sys.argv = old_argv


def run_wvs(config: Config) -> None:
    from popalign.evaluation.wvs_metrics import main
    import sys

    data = section(config, "data")
    output = section(config, "output")
    args = [
        "wvs_evaluation",
        "--llm_root",
        require(data, "llm_root"),
        "--human_root",
        require(data, "human_root"),
        "--output_root",
        require(output, "dir"),
    ]

    old_argv = sys.argv
    try:
        sys.argv = args
        main()
    finally:
        sys.argv = old_argv


def run_yrbss(config: Config) -> None:
    from popalign.evaluation.yrbss_metrics import main
    import sys

    data = section(config, "data")
    output = section(config, "output")
    args = [
        "yrbss_evaluation",
        "--method_dir",
        require(data, "method_dir"),
        "--human_file",
        require(data, "human_file"),
        "--save_dir",
        require(output, "dir"),
    ]

    old_argv = sys.argv
    try:
        sys.argv = args
        main()
    finally:
        sys.argv = old_argv
