"""Config-driven embedding pipeline runners."""

from __future__ import annotations

from ..config import Config, require, section
from ..io import read_jsonl_list, split_paths, write_jsonl
from ..paths import project_path
from ..serving import maybe_vllm_server
from ..generation.runner import run_ood_persona


def _model_runtime(config: Config) -> tuple[Config, Config]:
    model = section(config, "model")
    runtime = section(config, "runtime")
    model.setdefault("port", 8000)
    model.setdefault("api_base", f"http://localhost:{model['port']}/v1")
    runtime.setdefault("gpu", model.get("gpu", 1))
    runtime.setdefault("threads", 16)
    runtime.setdefault("max_tokens", 100)
    runtime.setdefault("temperature", 0.7)
    return model, runtime


def run_training_data(config: Config) -> None:
    from popalign.embeddings.training_data import (
        generate_long_queries_from_personas,
        generate_short_queries_from_personas,
    )

    model, runtime = _model_runtime(config)
    data = section(config, "data")
    output = section(config, "output")
    query_type = data.get("query_type", "both")
    if query_type not in {"long", "short", "both"}:
        raise ValueError("data.query_type must be long, short, or both.")
    persona_files = split_paths(require(data, "persona_file"))
    persona_pool = []
    for persona_file in persona_files:
        persona_pool.extend(read_jsonl_list(persona_file))

    if not persona_pool:
        raise ValueError("No personas found in the training-data inputs.")

    output_file = project_path(require(output, "file"))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with maybe_vllm_server(model, runtime):
        if query_type in {"long", "both"}:
            rows = generate_long_queries_from_personas(
                persona_pool,
                model["api_base"],
                model["name"],
                int(runtime["max_tokens"]),
                float(runtime["temperature"]),
            )
            if not rows:
                raise RuntimeError("No long query pairs were generated. Check the input personas and endpoint errors above.")
            print(f"[INFO] Generated {len(rows)}/{len(persona_pool)} long query pairs.")
            path = output_file if query_type == "long" else output_file.with_name(f"{output_file.stem}_long.jsonl")
            write_jsonl(path, rows)
            print(f"[INFO] Wrote {path}")

        if query_type in {"short", "both"}:
            rows = generate_short_queries_from_personas(
                persona_pool,
                model["api_base"],
                model["name"],
                int(runtime["max_tokens"]),
                float(runtime["temperature"]),
            )
            if not rows:
                raise RuntimeError("No short query pairs were generated. Check the input personas and endpoint errors above.")
            print(f"[INFO] Generated {len(rows)}/{len(persona_pool)} short query pairs.")
            path = output_file if query_type == "short" else output_file.with_name(f"{output_file.stem}_short.jsonl")
            write_jsonl(path, rows)
            print(f"[INFO] Wrote {path}")


def run_filter(config: Config) -> None:
    from popalign.embeddings.filtering import filter_reject_personas

    model, runtime = _model_runtime(config)
    data = section(config, "data")
    output_dir = project_path(require(section(config, "output"), "dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = split_paths(require(data, "input_file"))

    with maybe_vllm_server(model, runtime):
        for input_file in input_files:
            output_file = output_dir / f"{input_file.stem}_filtered_{model['name']}.jsonl"
            filter_reject_personas(
                str(input_file),
                str(output_file),
                model["api_base"],
                model["name"],
                int(runtime.get("max_tokens", 10)),
                float(runtime.get("temperature", 0.0)),
                int(runtime["threads"]),
            )


def run_match(config: Config) -> None:
    data = section(config, "data")
    persona_files = split_paths(require(data, "persona_file"))
    # Heavy embedding dependencies are imported only when this command is executed.
    from popalign.embeddings.matching import (
        EmbeddingSimilarityCalculator,
        MultiGPUEmbeddingSimilarityCalculator,
    )

    data = section(config, "data")
    output = section(config, "output")
    embedding = section(config, "embedding")
    multi_gpu = bool(embedding.get("multi_gpu", True))
    calculator_cls = MultiGPUEmbeddingSimilarityCalculator if multi_gpu else EmbeddingSimilarityCalculator
    kwargs = {
        "model_name": embedding.get("model", "Qwen/Qwen3-Embedding-0.6B"),
        "use_flash_attention": not embedding.get("no_flash_attention", False),
    }
    if multi_gpu:
        kwargs["num_gpus"] = embedding.get("num_gpus")
    calculator = calculator_cls(**kwargs)
    calculator.process_prompt_persona_matching(
        persona_file_path=",".join(str(path) for path in persona_files),
        output_file=str(project_path(require(output, "file"))),
        embedding_batch_size=int(embedding.get("batch_size", 32)),
        similarity_chunk_size=int(embedding.get("chunk_size", 1000)),
        top_k=int(embedding.get("top_k", 100)),
        save_embeddings=not embedding.get("no_save_embeddings", False),
        survey_type=data.get("survey_type", "wvs"),
        cache_dir=str(project_path(embedding.get("cache_dir", "embeddings_cache"))),
    )


def run_generate_personas(config: Config) -> None:
    from popalign.embeddings.persona_generation import generate_personas

    model, runtime = _model_runtime(config)
    data = section(config, "data")
    output = section(config, "output")

    input_file = split_paths(require(data, "input_file"))[0]
    with maybe_vllm_server(model, runtime):
        generate_personas(
            str(input_file),
            require(output, "file"),
            model["api_base"],
            model["name"],
            int(runtime.get("max_tokens", 300)),
            float(runtime.get("temperature", 0.7)),
            int(runtime["threads"]),
            data.get("survey_type", "wvs"),
        )


def run_generate_responses(config: Config) -> None:
    model = section(config, "model")
    runtime = section(config, "runtime")
    data = section(config, "data")
    output = section(config, "output")
    run_ood_persona(
        {
            "model": model,
            "runtime": runtime,
            "data": {
                "survey_path": require(data, "survey_path"),
                "persona_path": require(data, "persona_file"),
            },
            "output": {"dir": require(output, "dir")},
        }
    )
