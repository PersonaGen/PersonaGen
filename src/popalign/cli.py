"""Command line interface for PopAlign."""

from __future__ import annotations

import argparse
from typing import Callable

from .config import Config, load_config
from .generation.runner import run_bigfive_direct, run_bigfive_persona, run_ood_persona
from .evaluation.runner import run_bigfive as eval_bigfive
from .evaluation.runner import run_ood as eval_ood
from .evaluation.runner import run_wvs, run_yrbss
from .embeddings.runner import (
    run_filter,
    run_generate_personas,
    run_generate_responses,
    run_match,
    run_training_data,
)


Runner = Callable[[Config], None]


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML or JSON config file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="popalign", description="Population-aligned persona generation CLI")
    subparsers = parser.add_subparsers(dest="group", required=True)

    generate = subparsers.add_parser("generate", help="Generate survey responses")
    generate_sub = generate.add_subparsers(dest="command", required=True)
    for name, runner in {
        "bigfive-direct": run_bigfive_direct,
        "bigfive-persona": run_bigfive_persona,
        "ood-persona": run_ood_persona,
    }.items():
        cmd = generate_sub.add_parser(name)
        _add_config_arg(cmd)
        cmd.set_defaults(runner=runner)

    evaluate = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    evaluate_sub = evaluate.add_subparsers(dest="command", required=True)
    for name, runner in {
        "bigfive": eval_bigfive,
        "ood": eval_ood,
        "wvs": run_wvs,
        "yrbss": run_yrbss,
    }.items():
        cmd = evaluate_sub.add_parser(name)
        _add_config_arg(cmd)
        cmd.set_defaults(runner=runner)

    embeddings = subparsers.add_parser("embeddings", help="Run embedding persona adaptation")
    embeddings_sub = embeddings.add_subparsers(dest="command", required=True)
    for name, runner in {
        "training-data": run_training_data,
        "filter": run_filter,
        "match": run_match,
        "generate-personas": run_generate_personas,
        "generate-responses": run_generate_responses,
    }.items():
        cmd = embeddings_sub.add_parser(name)
        _add_config_arg(cmd)
        cmd.set_defaults(runner=runner)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    args.runner(config)


if __name__ == "__main__":
    main()
