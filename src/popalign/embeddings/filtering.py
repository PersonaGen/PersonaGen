import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from popalign.io import read_jsonl_list, write_jsonl
from popalign.llm import chat_completion
from popalign.serving import start_vllm_server, stop_process

# System prompt for LLM judge
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator tasked with determining whether a user persona matches a given query or description.

Your job is to carefully analyze if the persona accurately represents someone who would be described by the query. Consider:
1. Key characteristics mentioned in the query
2. Interests, behaviors, and preferences
3. Demographics if mentioned
4. Overall alignment between query and persona

Respond with ONLY "YES" if the persona matches the query, or "NO" if it doesn't match.
Be strict in your evaluation - only respond "YES" if there is a clear and strong alignment."""

JUDGE_USER_PROMPT = """Query: {query}

Persona: {persona}

Does this persona match the query? Respond with only "YES" or "NO"."""


def evaluate_persona(query, persona, api_base, model_name, max_tokens, temperature):
    """Evaluate if a persona matches a query using LLM judge"""

    user_prompt = JUDGE_USER_PROMPT.format(query=query, persona=persona)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = chat_completion(
        api_base=api_base,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Clean response and determine if it's a match
    response_clean = response.strip().upper()
    is_match = response_clean == "YES"

    return is_match, response


def process_query_item(item, api_base, model_name, max_tokens, temperature, threads):
    """Process a single query item and filter reject personas"""

    query = item["query"]
    query_unique_id = item["query_unique_id"]
    query_source = item["query_source"]
    query_file_source = item["query_file_source"]
    query_original_id = item["query_original_id"]
    true_profile = item.get("true_profile", "")
    list_profile = item["list_profile"]

    reject_personas = []

    # Evaluate each persona in parallel
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for persona_item in list_profile:
            future = executor.submit(
                evaluate_persona,
                query,
                persona_item["persona"],
                api_base,
                model_name,
                max_tokens,
                temperature
            )
            futures.append((future, persona_item))

        for future, persona_item in tqdm(futures, desc=f"Evaluating personas for {query_unique_id}", leave=False):
            is_match, response = future.result()

            if not is_match and response == "NO":  # If LLM says "NO", add to reject list
                reject_persona = {
                    "unique_id": persona_item["unique_id"],
                    "persona": persona_item["persona"]
                }
                reject_personas.append(reject_persona)

    # Create output item
    output_item = {
        "query_unique_id": query_unique_id,
        "query_source": query_source,
        "query_file_source": query_file_source,
        "query_original_id": query_original_id,
        "query": query,
        "true_profile": true_profile,
        "rejected_personas_count": len(reject_personas),
        "rejected_personas": reject_personas
    }

    return output_item


def get_existing_query_ids(output_file):
    """Get set of query unique IDs that already exist in the output file"""
    if not os.path.exists(output_file):
        return set()

    try:
        existing_data = read_jsonl_list(output_file)
        existing_query_ids = set()
        for item in existing_data:
            if "query_unique_id" in item:
                existing_query_ids.add(item["query_unique_id"])
        return existing_query_ids
    except Exception as e:
        print(f"[WARNING] Error reading existing output file {output_file}: {e}")
        return set()


def filter_reject_personas(input_file, output_file, api_base, model_name, max_tokens, temperature, threads):
    """Main function to filter reject personas"""

    input_data_list = read_jsonl_list(input_file)

    # Check which queries already exist in the output file
    existing_query_ids = get_existing_query_ids(output_file)

    # Filter out queries that already exist
    queries_to_process = []
    for item in input_data_list:
        if item["query_unique_id"] not in existing_query_ids:
            queries_to_process.append(item)

    if not queries_to_process:
        print(f'[INFO] All queries already processed in {output_file}, skipping')
        return

    print(f'[INFO] Found {len(existing_query_ids)} existing queries, will process {len(queries_to_process)} remaining queries')

    output_data_list = []

    for i, item in enumerate(queries_to_process):
        output_item = process_query_item(item, api_base, model_name, max_tokens, temperature, threads)
        output_data_list.append(output_item)

        # Save every 1000 iterations to avoid losing progress
        if (i + 1) % 1000 == 0:
            write_jsonl(output_file, output_data_list, append=True)
            print(f'[INFO] Progress saved: {i-999} - {i} queries processed')
            output_data_list = []

    # Save remaining data
    if output_data_list:
        write_jsonl(output_file, output_data_list, append=True)

    print(f'[INFO] Filtering complete. Results saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Filter reject personas using vLLM as judge")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file or comma-separated list of files")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder containing JSONL files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--api_base", type=str, required=True, help="Base URL for the API")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--max_tokens", type=int, default=10, help="Maximum number of tokens to generate (should be small for YES/NO)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (should be low for consistent judging)")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on")
    parser.add_argument("--gpu", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads to use for generation")

    args = parser.parse_args()

    # Only start vLLM server if model path is provided
    process_id = None
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        if process_id is None:
            print("[ERROR] Failed to start vLLM server")
            sys.exit(1)
        print(f"[INFO] vLLM server started with process ID: {process_id}")

    try:
        # Get input files
        if args.input_folder:
            input_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith('.jsonl')]
        elif args.input_file:
            input_files = args.input_file.split(',')
        else:
            print("[ERROR] Either --input_file or --input_folder must be provided")
            sys.exit(1)

        # Process each input file
        for input_file in input_files:
            input_file_name = os.path.basename(input_file).replace('.jsonl', '')
            output_file = os.path.join(args.output_folder, f'{input_file_name}_filtered_{args.model_name}.jsonl')

            print(f'[INFO] Processing {input_file} -> {output_file}')
            filter_reject_personas(input_file, output_file, args.api_base, args.model_name,
                                 args.max_tokens, args.temperature, args.threads)

    finally:
        if process_id:
            stop_process(process_id)
            print("[INFO] vLLM server stopped")


if __name__ == "__main__":
    main()
