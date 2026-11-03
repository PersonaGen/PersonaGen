#!/usr/bin/env python3
"""
Generate training pairs for an embedding model by generating queries from personas.

This script implements the approach described in the paper:
- Generate detailed (long) queries from personas using LLM prompting
- Generate broad (short) queries from personas using LLM prompting
- Each generated query is paired with the original persona for training

Output JSONL fields (per pair):
{
  "query_unique_id": str,
  "query": str,
  "query_source": str,
  "query_file_source": str,
  "query_original_id": str,
  "true_profile": str,
  "list_profile": List[Dict]  # List of personas with their unique_ids
}
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

from popalign.io import read_jsonl_list, write_jsonl
from popalign.llm import chat_completion
from popalign.serving import start_vllm_server, stop_process


def construct_prompt_long():
    return (
        "You are a helpful assistant tasked with generating queries based on detailed persona descriptions. "
        "Your goal is to generate concise queries that describe the key traits of the persona, such as age, gender, "
        "emotional tone, and significant interests or lifestyle. The query should be specific but short, capturing the persona's "
        "core characteristics without going into excessive detail.\n\n"
        "Example:\n"
        "Persona: 'The user is a young, likely female, living in a town she affectionately calls 'peach town', working multiple jobs, "
        "including one at a gym and another at a cleaners, and values relaxation and enjoyment of simple pleasures, like good food and sunny days, "
        "while navigating life's changes and emotional ups and downs with an introspective and slightly moody tone.'\n"
        "Generated query: 'Generate a young female persona who works multiple jobs, values relaxation and simple pleasures, and has an introspective "
        "and slightly moody tone.'\n\n"
        "Your generated query should capture the persona's essential traits in a clear and concise manner, avoiding unnecessary complexity.\n\n"
        "Only return the generated query, nothing else."
    )


def construct_user_prompt_long(persona) -> str:
    return (
        f"Based on the following persona description, generate a concise query that describes a persona with similar traits and characteristics. "
        f"Focus on the persona's key traits like age, gender, emotional tone, and major activities or interests, keeping the query short and to the point.\n\n"
        f"Persona description:\n"
        f"{persona}\n\n"
        f"Generated query:\n"
        f"Only return the generated query, nothing else."
    )


def construct_long_message(persona: str):
    s_prompt = construct_prompt_long()
    prompt = construct_user_prompt_long(persona=persona)

    return [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": prompt},
    ]


def construct_prompt_short():
    return (
        "You are a helpful assistant tasked with generating broader queries based on persona descriptions. "
        "Your goal is to generate simple and concise queries that describe the basic traits of the persona, such as age, gender, "
        "and general interests or activities. The query should be high-level, focusing on the core attributes, and should be short "
        "without diving into personal or emotional details.\n\n"
        "Example:\n"
        "Persona: 'The user is a young, likely female, living in a town she affectionately calls 'peach town', working multiple jobs, "
        "including one at a gym and another at a cleaners, and values relaxation and enjoyment of simple pleasures, like good food and sunny days, "
        "while navigating life's changes and emotional ups and downs with an introspective and slightly moody tone.'\n"
        "Generated query: 'Generate a young female persona who works multiple jobs.'\n\n"
        "Your generated query should focus on broad characteristics and general interests, and should be brief without over-explaining.\n\n"
        "Only return the generated query, nothing else."
    )


def construct_user_prompt_short(persona) -> str:
    return (
        f"Based on the following persona description, generate a broad, concise query that describes a persona with similar traits and characteristics. "
        f"Focus on the basic attributes such as age, gender, and major interests or lifestyle, keeping the query short and general.\n\n"
        f"Persona description:\n"
        f"{persona}\n\n"
        "Generated query:\n"
        f"Only return the generated query, nothing else."
    )


def construct_short_message(persona: str):
    s_prompt = construct_prompt_short()
    prompt = construct_user_prompt_short(persona=persona)

    return [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": prompt},
    ]


def generate_query_from_persona(persona: str, messages: List[Dict], api_base: str, model_name: str,
                               max_tokens: int = 100, temperature: float = 0.7) -> str:
    """Generate a query from a persona using LLM prompting"""

    try:
        response = chat_completion(
            api_base=api_base,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.strip()
    except Exception as e:
        print(f"Error generating query: {e}")
        return f"Error generating query for persona: {persona[:100]}..."


def generate_long_queries_from_personas(persona_pool: List[Dict], api_base: str, model_name: str,
                                       max_tokens: int = 100, temperature: float = 0.7) -> List[Dict]:
    """Generate detailed (long) queries from each persona"""

    training_data = []

    for i, persona_item in enumerate(persona_pool):
        persona_text = persona_item.get("persona", "")
        persona_uid = persona_item.get("persona_unique_id") or persona_item.get("unique_id") or f"p_{i}"
        source = persona_item.get("source", "unknown")

        if not persona_text.strip():
            continue

        # Generate detailed (long) query
        messages = construct_long_message(persona_text)
        detailed_query = generate_query_from_persona(
            persona_text, messages, api_base, model_name, max_tokens, temperature
        )

        if detailed_query and not detailed_query.startswith("Error"):
            training_instance = {
                "query_unique_id": f"{persona_uid}_long",
                "query": detailed_query,
                "query_source": "long",
                "query_file_source": source,
                "query_original_id": persona_uid,
                "true_profile": persona_text,
                "list_profile": [persona_item]  # Single persona for this query
            }
            training_data.append(training_instance)

    return training_data


def generate_short_queries_from_personas(persona_pool: List[Dict], api_base: str, model_name: str,
                                        max_tokens: int = 100, temperature: float = 0.7) -> List[Dict]:
    """Generate broad (short) queries from each persona"""

    training_data = []

    for i, persona_item in enumerate(persona_pool):
        persona_text = persona_item.get("persona", "")
        persona_uid = persona_item.get("persona_unique_id") or persona_item.get("unique_id") or f"p_{i}"
        source = persona_item.get("source", "unknown")

        if not persona_text.strip():
            continue

        # Generate broad (short) query
        messages = construct_short_message(persona_text)
        broad_query = generate_query_from_persona(
            persona_text, messages, api_base, model_name, max_tokens, temperature
        )

        if broad_query and not broad_query.startswith("Error"):
            training_instance = {
                "query_unique_id": f"{persona_uid}_short",
                "query": broad_query,
                "query_source": "short",
                "query_file_source": source,
                "query_original_id": persona_uid,
                "true_profile": persona_text,
                "list_profile": [persona_item]  # Single persona for this query
            }
            training_data.append(training_instance)

    return training_data


def main():
    ap = argparse.ArgumentParser(description="Generate training pairs by generating queries from personas using LLM.")
    ap.add_argument("--persona_file", type=str, required=True, help="Persona JSONL file (or comma-separated list)")
    ap.add_argument("--output", type=str, required=True, help="Output training JSONL path")
    ap.add_argument("--api_base", type=str, required=True, help="Base URL for the API")
    ap.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    ap.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens for query generation")
    ap.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    ap.add_argument("--model_path", type=str, help="Path to the model for local vLLM server")
    ap.add_argument("--port", type=int, default=8000, help="Port for vLLM server")
    ap.add_argument("--gpu", type=int, default=2, help="Number of GPUs to use")
    ap.add_argument("--query_type", type=str, choices=["long", "short", "both"], default="both",
                   help="Type of queries to generate: long, short, or both")

    args = ap.parse_args()

    # Start vLLM server if model path is provided
    process_id = None
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        if process_id is None:
            print("[ERROR] Failed to start vLLM server")
            sys.exit(1)
        print(f"[INFO] vLLM server started with process ID: {process_id}")

    try:
        # Load persona pool
        persona_files = [Path(p) for p in args.persona_file.split(",")]
        pool = []
        for pf in persona_files:
            pool.extend(read_jsonl_list(pf))

        print(f"[INFO] Loaded {len(pool)} personas from {len(persona_files)} file(s)")

        # Generate queries from personas based on query type
        print(f"[INFO] Generating {args.query_type} queries using {args.model_name}...")

        if args.query_type == "long" or args.query_type == "both":
            long_training_data = generate_long_queries_from_personas(
                pool, args.api_base, args.model_name, args.max_tokens, args.temperature
            )
            print(f"[INFO] Generated {len(long_training_data)} long training instances")

            # Write long queries output
            if args.query_type == "long":
                output_path = args.output
            else:
                output_path = args.output.replace('.jsonl', '_long.jsonl')

            write_jsonl(output_path, long_training_data)
            print(f"[INFO] Long queries saved to {output_path}")

        if args.query_type == "short" or args.query_type == "both":
            short_training_data = generate_short_queries_from_personas(
                pool, args.api_base, args.model_name, args.max_tokens, args.temperature
            )
            print(f"[INFO] Generated {len(short_training_data)} short training instances")

            # Write short queries output
            if args.query_type == "short":
                output_path = args.output
            else:
                output_path = args.output.replace('.jsonl', '_short.jsonl')

            write_jsonl(output_path, short_training_data)
            print(f"[INFO] Short queries saved to {output_path}")

        if args.query_type == "both":
            print(f"[INFO] Total: {len(long_training_data) + len(short_training_data)} training instances generated")

    finally:
        if process_id:
            stop_process(process_id)
            print("[INFO] vLLM server stopped")


if __name__ == "__main__":
    main()
