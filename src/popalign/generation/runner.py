"""Config-driven survey generation runners."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from tqdm import tqdm

from ..config import Config, require, section
from ..io import jsonl_files, read_jsonl, read_jsonl_list, split_paths, write_jsonl
from ..llm import chat_completion, normalize_api_base
from ..paths import PROJECT_ROOT, project_path
from ..serving import maybe_vllm_server
from . import prompts


DEFAULT_BIGFIVE_QUESTION_FILE = PROJECT_ROOT / "examples" / "ipip50_questions.jsonl"


def _model_runtime(config: Config) -> tuple[Config, Config]:
    model = section(config, "model")
    runtime = section(config, "runtime")
    if "name" not in model:
        raise ValueError("Missing required config value: model.name")
    model.setdefault("port", 8000)
    model.setdefault("api_base", f"http://localhost:{model['port']}/v1")
    runtime.setdefault("threads", 8)
    runtime.setdefault("max_tokens", 512)
    runtime.setdefault("temperature", 0.7)
    return model, runtime


def _chat(model: Config, runtime: Config, messages: List[Dict[str, str]]) -> str:
    return chat_completion(
        api_base=normalize_api_base(model["api_base"]),
        model_name=model["name"],
        messages=messages,
        max_tokens=int(runtime["max_tokens"]),
        temperature=float(runtime["temperature"]),
        api_key=model.get("api_key"),
    )


def _persona_uid(item: Dict) -> str:
    for field in ("unique_id", "persona_unique_id", "persona_id", "id"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("Each persona needs a nonempty unique_id, persona_unique_id, persona_id, or id for safe resumption")


def _persona_row(item: Dict) -> Dict:
    return {
        "persona_id": item.get("persona_id") if item.get("persona_id") is not None else item.get("id"),
        "persona_source": item.get("persona_source") or item.get("source", ""),
        "persona_unique_id": _persona_uid(item),
        "persona": item["persona"],
    }


def _existing_ids(output_file: Path, field: str) -> set:
    if not output_file.exists():
        return set()
    return {row[field] for row in read_jsonl(output_file) if field in row}


def _load_bigfive_questions(data: Config) -> List[Dict]:
    question_file = project_path(data.get("question_file", DEFAULT_BIGFIVE_QUESTION_FILE))
    with question_file.open("r", encoding="utf-8") as infile:
        rows = [json.loads(line) for line in infile if line.strip()]
    print(f"[INFO] Loaded {len(rows)} Big Five questions from {question_file}")
    return rows


def _generate_row(
    items: Iterable[Dict],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    model: Config,
    runtime: Config,
    desc: str,
) -> Dict:
    row: Dict = {}
    item_list = list(items)
    with ThreadPoolExecutor(max_workers=int(runtime["threads"])) as executor:
        futures = [
            executor.submit(_chat, model, runtime, build_messages(item))
            for item in item_list
        ]
        for item, future in tqdm(list(zip(item_list, futures)), desc=desc, total=len(item_list)):
            row[item["original_id"]] = future.result()
    return row


def run_bigfive_direct(config: Config) -> None:
    model, runtime = _model_runtime(config)
    data = section(config, "data")
    output = section(config, "output")
    questions = _load_bigfive_questions(data)
    output_dir = project_path(require(output, "dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output.get(
        "file",
        f"bigfive_direct_{model['name']}_temperature_{runtime['temperature']}.jsonl",
    )

    num_repeat = int(runtime.get("num_repeat", 1))
    existing = _existing_ids(output_file, "iteration_id")
    pending = [idx for idx in range(num_repeat) if idx not in existing]
    if not pending:
        print(f"[INFO] All {num_repeat} iterations already exist in {output_file}; skipping.")
        return

    with maybe_vllm_server(model, runtime):
        rows = []
        for idx in pending:
            row = {"iteration_id": idx}
            row.update(
                _generate_row(
                    questions,
                    lambda item: prompts.bigfive_direct(item["question"]),
                    model,
                    runtime,
                    desc=f"Big Five direct iteration {idx + 1}/{num_repeat}",
                )
            )
            rows.append(row)
            if len(rows) >= int(output.get("flush_every", 100)):
                write_jsonl(output_file, rows, append=True)
                rows = []
        if rows:
            write_jsonl(output_file, rows, append=True)
    print(f"[INFO] Wrote {output_file}")


def run_bigfive_persona(config: Config) -> None:
    model, runtime = _model_runtime(config)
    data = section(config, "data")
    output = section(config, "output")
    questions = _load_bigfive_questions(data)
    persona_files = split_paths(require(data, "persona_file"))
    output_dir = project_path(require(output, "dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with maybe_vllm_server(model, runtime):
        for persona_file in persona_files:
            personas = read_jsonl_list(persona_file)
            output_file = output_dir / f"{persona_file.stem}_bigfive_{model['name']}.jsonl"
            existing = _existing_ids(output_file, "persona_unique_id")
            pending = [item for item in personas if _persona_uid(item) not in existing]
            rows = []
            for idx, persona_item in enumerate(pending):
                row = _persona_row(persona_item)
                row.update(
                    _generate_row(
                        questions,
                        lambda item, persona=row["persona"]: prompts.bigfive_with_persona(item["question"], persona),
                        model,
                        runtime,
                        desc=f"{persona_file.stem} persona {idx + 1}/{len(pending)}",
                    )
                )
                rows.append(row)
                if len(rows) >= int(output.get("flush_every", 100)):
                    write_jsonl(output_file, rows, append=True)
                    rows = []
            if rows:
                write_jsonl(output_file, rows, append=True)
            print(f"[INFO] Wrote {output_file}")


def run_ood_persona(config: Config) -> None:
    model, runtime = _model_runtime(config)
    data = section(config, "data")
    output = section(config, "output")
    survey_files = jsonl_files(require(data, "survey_path"))
    persona_files = jsonl_files(require(data, "persona_path"))
    output_dir = project_path(require(output, "dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with maybe_vllm_server(model, runtime):
        for survey_file in survey_files:
            survey_items = read_jsonl_list(survey_file)
            for persona_file in persona_files:
                personas = read_jsonl_list(persona_file)
                output_file = output_dir / f"{survey_file.stem}_{persona_file.stem}_{model['name']}.jsonl"
                existing = _existing_ids(output_file, "persona_unique_id")
                pending = [item for item in personas if _persona_uid(item) not in existing]
                rows = []
                for idx, persona_item in enumerate(pending):
                    row = _persona_row(persona_item)
                    row.update(
                        _generate_row(
                            survey_items,
                            lambda item, persona=row["persona"]: prompts.ood_with_persona(item, persona),
                            model,
                            runtime,
                            desc=f"{survey_file.stem}/{persona_file.stem} {idx + 1}/{len(pending)}",
                        )
                    )
                    rows.append(row)
                    if len(rows) >= int(output.get("flush_every", 100)):
                        write_jsonl(output_file, rows, append=True)
                        rows = []
                if rows:
                    write_jsonl(output_file, rows, append=True)
                print(f"[INFO] Wrote {output_file}")
