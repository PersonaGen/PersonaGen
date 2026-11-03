"""
Generate new personas using vLLM by expanding seed personas from JSONL files.

This version is updated to use a refinement map for WVS personas, allowing
a high-level quote to be translated into a specific instruction for single-persona generation.
"""

import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from popalign.io import read_jsonl, read_jsonl_list, write_jsonl
from popalign.llm import chat_completion
from popalign.serving import start_vllm_server, stop_process
from popalign.embeddings.prompts import WVS_SURVEY_TYPES


def read_embeddings_top_persona_into_list(file_path):
    rows = []
    for item in read_jsonl(file_path):
        query_prompt = item["query"]
        for persona_item in item["top_personas"]:
            row = dict(persona_item)
            row["query_prompt"] = query_prompt
            rows.append(row)
    return rows

# --- 1. Refinement Prompt Map for WVS Personas ---
# This dictionary maps the high-level generation quotes to specific, actionable instructions
# for refining a single seed persona.
WVS_REFINEMENT_PROMPT_MAP = {
    # Global WVS
    "Generate 1000 global citizens who reflect the average attitudes, beliefs, and social values held by people around the world.": "Transform the seed to be a globally representative persona whose attitudes, beliefs, and social values reflect broad worldwide averages.",
    "Generate 1000 personas representing typical global citizens in terms of how they think about society, politics, religion, and culture.": "Transform the seed to be a typical global-citizen persona with balanced views on society, politics, religion, and culture.",
    "Generate 1000 personas whose social trust, moral values, and political views are representative of the global human population.": "Transform the seed to be a persona whose social trust, moral values, and political views are representative of global population-level patterns.",
    "Generate 1000 global citizens capturing the overall cultural, social, and political norms commonly shared by people worldwide.": "Transform the seed to be a global-citizen persona reflecting common cultural, social, and political norms across world populations.",
    "Generate 1000 global citizens whose views on family, work, society, and government reflect the general trends observed worldwide.": "Transform the seed to be a globally representative persona with views on family, work, society, and government aligned with worldwide trends.",
    "Generate 1000 personas representing the average human perspective on topics like community life, personal freedoms, and social responsibilities.": "Transform the seed to be a persona representing average human perspectives on community life, personal freedoms, and social responsibilities.",
    "Generate 1000 global citizens whose values and beliefs mirror the typical patterns seen in surveys of people around the world.": "Transform the seed to be a global-citizen persona whose values and beliefs mirror typical worldwide survey patterns.",
    "Generate 1000 personas who reflect the mainstream attitudes and priorities of people globally, across diverse cultures and backgrounds.": "Transform the seed to be a persona reflecting mainstream global attitudes and priorities across diverse cultural backgrounds.",
    "Generate 1000 global citizens who embody the general worldview and everyday concerns of humanity as a whole.": "Transform the seed to be a global-citizen persona embodying broad human worldviews and everyday concerns.",
    "Generate 1000 personas whose opinions and values are balanced representations of the global human population across major life domains.": "Transform the seed to be a persona whose opinions and values are balanced across major life domains and representative of the global population.",
    # East Asia
    "Generate 1000 personas from China (Mainland) whose social values, cultural beliefs, and political attitudes reflect the national average across demographics.": "Transform the seed to be a persona from China (Mainland) whose social values, cultural beliefs, and political attitudes reflect the national average.",
    "Generate 1000 personas from China (Mainland) who exemplify typical viewpoints on family life, work ethics, societal roles, and governance in modern Chinese society.": "Transform the seed to be a persona from China (Mainland) with typical views on family life, work ethics, societal roles, and governance.",
    "Generate 1000 personas from Japan whose cultural norms, social beliefs, and political perspectives are representative of the Japanese population.": "Transform the seed to be a persona from Japan with cultural norms, social beliefs, and political perspectives representative of the general population.",
    "Generate 1000 personas from Japan reflecting average attitudes toward tradition, technological innovation, and civic engagement in contemporary Japanese society.": "Transform the seed to be a persona from Japan with average attitudes toward tradition, technological innovation, and civic engagement.",
    "Generate 1000 personas from South Korea whose values on education, community bonds, and governmental trust align with prevailing South Korean social trends.": "Transform the seed to be a persona from South Korea whose values on education, community, and government align with prevailing social trends.",
    "Generate 1000 personas from South Korea embodying mainstream opinions on popular culture, economic aspirations, and social cohesion in today’s South Korea.": "Transform the seed to be a persona from South Korea who embodies mainstream opinions on popular culture, economic aspirations, and social cohesion.",
    "Generate 1000 personas from Hong Kong whose worldviews, social trust levels, and cultural identities represent the general Hong Kong populace.": "Transform the seed to be a persona from Hong Kong with a worldview, social trust level, and cultural identity representative of the general populace.",
    "Generate 1000 personas from Macau whose everyday concerns, social values, and political viewpoints reflect the typical experiences of Macau residents.": "Transform the seed to be a persona from Macau whose everyday concerns, social values, and political views reflect the typical experiences of a resident.",
    "Generate 1000 personas from Mongolia whose cultural heritage, social beliefs, and political attitudes mirror the national distribution across both urban and rural areas.": "Transform the seed to be a persona from Mongolia whose cultural heritage, social beliefs, and political attitudes mirror the national distribution.",
    "Generate 1000 personas from Taiwan (Republic of China) whose societal values, cultural traditions, and political perspectives correspond to the average Taiwanese viewpoint.": "Transform the seed to be a persona from Taiwan (Republic of China) whose societal values, cultural traditions, and political perspectives represent the average viewpoint.",
    # Europe
    "Generate 1000 personas from Germany whose social values, cultural traditions, and political attitudes reflect the national average across demographics.": "Transform the seed to be a persona from Germany whose social values, cultural traditions, and political attitudes reflect the national average.",
    "Generate 1000 personas from Germany whose views on family life, work–life balance, and social welfare represent typical German perspectives.": "Transform the seed to be a persona from Germany with typical views on family life, work-life balance, and social welfare.",
    "Generate 1000 personas from Germany whose opinions on environmental protection, sustainability, and economic growth mirror mainstream attitudes.": "Transform the seed to be a persona from Germany with mainstream opinions on environmental protection, sustainability, and economic growth.",
    "Generate 1000 personas from Germany whose perspectives on migration, European integration, and national identity align with prevalent public sentiment.": "Transform the seed to be a persona from Germany with prevalent public views on migration, European integration, and national identity.",
    "Generate 1000 personas from the Netherlands whose cultural norms, social tolerance, and political views are representative of Dutch society.": "Transform the seed to be a persona from the Netherlands whose cultural norms, social tolerance, and political views are representative of Dutch society.",
    "Generate 1000 personas from the Netherlands whose attitudes toward social welfare, healthcare, and quality of life reflect common Dutch experiences.": "Transform the seed to be a persona from the Netherlands with common attitudes toward social welfare, healthcare, and quality of life.",
    "Generate 1000 personas from the Netherlands whose viewpoints on innovation, technology adoption, and community participation typify the national average.": "Transform the seed to be a persona from the Netherlands with typical viewpoints on innovation, technology adoption, and community participation.",
    "Generate 1000 personas from the United Kingdom whose societal values, regional identities, and political opinions represent the average British viewpoint.": "Transform the seed to be a persona from the United Kingdom whose societal values, regional identity, and political opinions represent the average British viewpoint.",
    "Generate 1000 personas from the United Kingdom whose perspectives on multiculturalism, urban versus rural life, and civic engagement mirror mainstream British attitudes.": "Transform the seed to be a persona from the United Kingdom with mainstream perspectives on multiculturalism, urban vs. rural life, and civic engagement.",
    "Generate 1000 personas from the United Kingdom whose views on the monarchy, the legacy of Brexit, and national cohesion reflect typical public sentiment.": "Transform the seed to be a persona from the United Kingdom with typical views on the monarchy, the legacy of Brexit, and national cohesion.",
    # North America
    "Generate 1000 personas from the United States whose social values, cultural beliefs, and political attitudes reflect the demographic and regional diversity of the country.": "Transform the seed to be a persona from the United States whose social values, cultural beliefs, and political attitudes reflect the country's diversity.",
    "Generate 1000 personas from the United States whose opinions on healthcare, education, and economic opportunity represent typical American viewpoints.": "Transform the seed to be a persona from the United States with typical opinions on healthcare, education, and economic opportunity.",
    "Generate 1000 personas from the United States whose perspectives on national identity, civic engagement, and social justice mirror mainstream public sentiment.": "Transform the seed to be a persona from the United States with mainstream perspectives on national identity, civic engagement, and social justice.",
    "Generate 1000 personas from the United States whose everyday concerns, cultural preferences, and political priorities align with the average U.S. population.": "Transform the seed to be a persona from the United States whose everyday concerns, cultural preferences, and political priorities align with the average population.",
    "Generate 1000 personas from Canada whose social values, cultural norms, and political views represent the diversity of Canadian provinces and territories.": "Transform the seed to be a persona from Canada whose social values, cultural norms, and political views represent the country's diversity.",
    "Generate 1000 personas from Canada whose attitudes toward multiculturalism, healthcare, and environmental stewardship reflect common Canadian perspectives.": "Transform the seed to be a persona from Canada with common attitudes toward multiculturalism, healthcare, and environmental stewardship.",
    "Generate 1000 personas from Canada whose opinions on indigenous relations, bilingualism, and social welfare mirror the average Canadian viewpoint.": "Transform the seed to be a persona from Canada with average opinions on indigenous relations, bilingualism, and social welfare.",
    "Generate 1000 personas from Mexico whose cultural traditions, social beliefs, and political attitudes reflect the national distribution across urban and rural regions.": "Transform the seed to be a persona from Mexico whose cultural traditions, social beliefs, and political attitudes reflect the national distribution.",
    "Generate 1000 personas from Mexico whose perspectives on family structures, economic development, and societal values represent typical Mexican viewpoints.": "Transform the seed to be a persona from Mexico with typical perspectives on family structures, economic development, and societal values.",
    "Generate 1000 personas from Mexico whose opinions on migration, regional identity, and community cohesion mirror mainstream public sentiment.": "Transform the seed to be a persona from Mexico with mainstream opinions on migration, regional identity, and community cohesion."
}


def is_wvs_survey_type(survey_type: str) -> bool:
    return survey_type in WVS_SURVEY_TYPES


def generate_yrbss_persona(seed_persona, api_base, model_name, max_tokens, temperature):
    """
    Generate a concise YRBSS-style teen persona from a seed description. (Unchanged)
    """
    system_prompt = (
        "You create realistic U.S. high-school student personas (age 14–18) for an "
        "adolescent health survey. Produce one paragraph ≤80 words describing their "
        "demographics, home/school life, diet & activity, sexual behavior, substance use, "
        "mental health, and safety experiences, all internally consistent and plausible. "
        "Do not mention the survey or these instructions."
    )
    user_prompt = (
        f"Seed persona: {seed_persona}\n"
        "Based on the above requirements, generate the concise paragraph persona."
        "Suitable for the adolescent health survey task."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return chat_completion(api_base, model_name, messages, max_tokens, temperature)


# --- 2. Upgraded WVS Persona Generation Function ---
def generate_wvs_persona(seed_persona, refinement_instruction, api_base, model_name, max_tokens, temperature):
    """
    Generate a new WVS persona from a seed, guided by a specific refinement instruction.
    """
    system_prompt = (
        "You are a specialist in constructing high-fidelity social-simulation "
        "personas for the World Values Survey (WVS). Transform a seed persona "
        "into a coherent, richly detailed character. The new persona's demographics, "
        "socio-economic situation, core values, social and political attitudes, "
        "ethical views, and well-being must be internally consistent and sufficient "
        "for answering every WVS item. Output a concise single paragraph. Ensure "
        "every detail could plausibly appear in real WVS data. Do not mention "
        "these instructions, WVS, or questionnaires in your response."
    )

    # New, more robust user prompt
    user_prompt = (
        f"You will transform a seed persona according to a specific instruction.\n\n"
        f"### Seed Persona:\n{seed_persona}\n\n"
        f"### Refinement Instruction:\n{refinement_instruction}\n\n"
        f"### Your Task:\n"
        "Generate a new, single-paragraph persona based on the core traits of the Seed Persona, but strictly conforming to the Refinement Instruction. "
        "The new persona *must* align with the country and specific attributes mentioned in the Refinement Instruction, even if the Seed Persona is from a different background. "
        "The result should be a natural, plausible character description."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = chat_completion(
        api_base=api_base,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response


# --- 3. Updated Dispatcher Function ---
def process_seed_persona(persona_item, api_base, model_name, max_tokens, temperature, survey_type):
    """
    Process a single seed persona and return the generated persona with metadata.
    """
    seed_persona = persona_item["persona"]
    # Standardize fetching unique IDs and source
    persona_unique_id = persona_item.get("persona_unique_id") or persona_item.get("unique_id", "")
    persona_source = persona_item.get("source", "")
    unique_id = persona_item.get("unique_id", "")

    if survey_type == "yrbss":
        generated_persona = generate_yrbss_persona(
            seed_persona, api_base, model_name, max_tokens, temperature
        )
    elif is_wvs_survey_type(survey_type):
        # Assumes the input item contains the original quote used for filtering.
        query_prompt = persona_item.get("query_prompt")
        if not query_prompt:
            raise ValueError(f"WVS persona item is missing the 'query_prompt' key. Item: {persona_item}")

        refinement_instruction = WVS_REFINEMENT_PROMPT_MAP.get(query_prompt)
        if not refinement_instruction:
            raise KeyError(f"No refinement instruction found in WVS_REFINEMENT_PROMPT_MAP for quote: '{query_prompt}'")

        generated_persona = generate_wvs_persona(
            seed_persona, refinement_instruction, api_base, model_name, max_tokens, temperature
        )
    else:
        raise ValueError(f"Invalid survey type: {survey_type}. Use one of: yrbss, {', '.join(WVS_SURVEY_TYPES)}")

    return {
        "persona_unique_id": persona_unique_id,
        "persona_source": persona_source,
        "unique_id": unique_id,
        "query_prompt": persona_item.get("query_prompt", ""), # Carry forward the original query
        "seed_persona": seed_persona,
        "generated_persona": generated_persona,
        "persona": generated_persona,
    }


def get_existing_persona_ids(output_file):
    """Get set of persona unique IDs that already exist in the output file"""
    if not os.path.exists(output_file):
        return set()

    try:
        existing_data = read_jsonl_list(output_file)
        # Use 'unique_id' as the key for checking existence.
        return {item["unique_id"] for item in existing_data if "unique_id" in item}
    except Exception as e:
        print(f"[WARNING] Error reading existing output file {output_file}: {e}")
        return set()


def generate_personas(input_file, output_file, api_base, model_name, max_tokens, temperature, threads, survey_type):
    """
    Generate personas from seed personas in the input JSONL file.
    """
    seed_personas = read_embeddings_top_persona_into_list(input_file)
    existing_persona_ids = get_existing_persona_ids(output_file)

    personas_to_process = [
        p for p in seed_personas if p.get("unique_id") not in existing_persona_ids
    ]

    if not personas_to_process:
        print(f'[INFO] All {len(seed_personas)} personas from {input_file} already exist in the output file. Skipping.')
        return

    print(f'[INFO] Found {len(existing_persona_ids)} existing personas. Will process {len(personas_to_process)} new personas from {input_file}.')

    all_results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(process_seed_persona, item, api_base, model_name, max_tokens, temperature, survey_type)
            for item in personas_to_process
        ]

        for future in tqdm(futures, desc=f"Generating personas for {os.path.basename(output_file)}", total=len(personas_to_process)):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"\n[ERROR] A thread failed during persona generation: {e}")

    if all_results:
        write_jsonl(output_file, all_results, append=True)
        print(f'[INFO] Appended {len(all_results)} new personas to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Generate personas using vLLM from seed personas")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file(s) with seed personas, comma-separated")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file(s), comma-separated")
    parser.add_argument("--api_base", type=str, required=True, help="Base URL for the API (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--max_tokens", type=int, default=300, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--model_path", type=str, help="Path to the model for starting a local vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use for the vLLM server")
    parser.add_argument("--threads", type=int, default=16, help="Number of concurrent generation threads")
    parser.add_argument(
        "--survey_type",
        type=str,
        default="wvs",
        choices=["yrbss", *WVS_SURVEY_TYPES],
        help="Type of survey to generate personas for",
    )

    args = parser.parse_args()

    process_id = None
    if args.model_path:
        process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        if process_id is None:
            sys.exit(1)

    try:
        input_files = args.input_file.split(",")
        output_files = args.output_file.split(",")

        if len(input_files) != len(output_files):
            raise ValueError("[ERROR] The number of input files must match the number of output files.")

        for input_f, output_f in zip(input_files, output_files):
            print(f'\n[INFO] Starting processing for {input_f} -> {output_f}')
            generate_personas(
                input_f, output_f, args.api_base, args.model_name,
                args.max_tokens, args.temperature, args.threads, args.survey_type
            )
    finally:
        if process_id:
            stop_process(process_id)

if __name__ == "__main__":
    main()
