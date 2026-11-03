#!/usr/bin/env python3
"""
Embedding Calculation and Ranking System
Uses Qwen/Qwen3-Embedding-0.6B model to calculate similarities between queries and profiles
Efficiently processes large-scale data with GPU acceleration
Now supports multi-GPU parallel processing
Modified for specific 10 prompts and persona matching
"""

import json
import torch
import torch.multiprocessing as mp
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import gc
from typing import List, Dict, Tuple
import pickle
import os
import glob
import time
import math

from popalign.embeddings.cache import embedding_cache_paths
from popalign.embeddings.prompts import SUPPORTED_SURVEY_TYPES, get_survey_prompts

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set multiprocessing start method
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

class MultiGPUEmbeddingSimilarityCalculator:
    """
    Multi-GPU embedding similarity calculator using Qwen model
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B",
                 use_flash_attention: bool = True, num_gpus: int = None):
        """
        Initialize the multi-GPU embedding calculator

        Args:
            model_name: Hugging Face model name
            use_flash_attention: Whether to use flash attention for better performance
            num_gpus: Number of GPUs to use (None for auto-detect)
        """
        self.model_name = model_name
        self.use_flash_attention = use_flash_attention

        # Detect available GPUs
        if torch.cuda.is_available():
            self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
            logger.info(f"Found {torch.cuda.device_count()} GPUs, using {self.num_gpus}")
            for i in range(self.num_gpus):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.num_gpus = 1
            logger.warning("No GPU available, using CPU")

    def _worker_process(self, gpu_id: int, data_chunk: List[Dict],
                       embedding_batch_size: int, is_query: bool,
                       results_queue: mp.Queue, error_queue: mp.Queue):
        """
        Worker process for computing embeddings on a specific GPU

        Args:
            gpu_id: GPU device ID
            data_chunk: Data chunk to process
            embedding_batch_size: Batch size for embedding computation
            is_query: Whether processing queries or profiles
            results_queue: Queue to return results
            error_queue: Queue to return errors
        """
        try:
            # Set GPU device for this process
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)

            logger.info(f"Worker {gpu_id}: Starting on {device}")
            logger.info(f"Worker {gpu_id}: Processing {len(data_chunk)} items")
            logger.info(f"Worker {gpu_id}: Current CUDA device = {torch.cuda.current_device()}")

            # Initialize model on this specific GPU
            calculator = EmbeddingSimilarityCalculator(
                model_name=self.model_name,
                use_flash_attention=self.use_flash_attention,
                device=device
            )

            # Verify the model is on the correct device
            logger.info(f"Worker {gpu_id}: Model loaded on device = {calculator.model.device}")

            # Compute embeddings for this chunk
            embeddings = calculator.compute_embeddings_batch(
                data_chunk,
                batch_size=embedding_batch_size,
                is_query=is_query,
                save_path=None
            )

            # Return results
            logger.info(f"Worker {gpu_id}: Putting results in queue...")
            try:
                results_queue.put({
                    'gpu_id': gpu_id,
                    'embeddings': embeddings,
                    'chunk_size': len(data_chunk)
                })
                logger.info(f"Worker {gpu_id}: Results successfully put in queue")
            except Exception as e:
                logger.error(f"Worker {gpu_id}: Failed to put results in queue: {e}")
                error_queue.put({
                    'gpu_id': gpu_id,
                    'error': f"Failed to put results in queue: {e}"
                })
                return

            logger.info(f"Worker {gpu_id}: Completed processing {len(data_chunk)} items")

            # Clean up model and memory
            logger.info(f"Worker {gpu_id}: Cleaning up...")
            del calculator
            del embeddings  # Also delete embeddings to free memory
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Worker {gpu_id}: Worker process finished successfully")
            return

        except Exception as e:
            error_queue.put({
                'gpu_id': gpu_id,
                'error': str(e)
            })
            logger.error(f"Worker {gpu_id}: Error - {e}")

    def _split_data_for_gpus(self, data: List[Dict]) -> List[List[Dict]]:
        """
        Split data into chunks for each GPU

        Args:
            data: Input data list

        Returns:
            List of data chunks for each GPU
        """
        chunk_size = math.ceil(len(data) / self.num_gpus)
        chunks = []

        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))

            if start_idx < len(data):
                chunk = data[start_idx:end_idx]
                chunks.append(chunk)
                logger.info(f"GPU {i}: Will process {len(chunk)} items (indices {start_idx}-{end_idx-1})")

        return chunks

    def compute_embeddings_multi_gpu(self, data: List[Dict],
                                    embedding_batch_size: int = 32,
                                    is_query: bool = False) -> np.ndarray:
        """
        Compute embeddings using multiple GPUs

        Args:
            data: List of data items with metadata
            embedding_batch_size: Batch size for processing
            is_query: Whether processing queries or profiles

        Returns:
            Combined embeddings from all GPUs
        """
        logger.info(f"Computing embeddings for {len(data)} items using {self.num_gpus} GPUs")

        # Split data for GPUs
        data_chunks = self._split_data_for_gpus(data)

        # Create queues for inter-process communication
        results_queue = mp.Queue()
        error_queue = mp.Queue()

        # Start worker processes
        processes = []
        for i, chunk in enumerate(data_chunks):
            if len(chunk) > 0:  # Only start process if chunk is not empty
                p = mp.Process(
                    target=self._worker_process,
                    args=(i, chunk, embedding_batch_size, is_query, results_queue, error_queue)
                )
                p.start()
                processes.append(p)

        # Collect results
        results = []
        errors = []

        # Wait for all processes to complete and collect results
        logger.info(f"Waiting for {len(processes)} GPU workers to complete...")
        results = []
        errors = []

        # Collect results as processes complete
        active_processes = set(range(len(processes)))
        collected_results = set()

        while active_processes:
            # Check for completed processes
            for i in list(active_processes):
                if not processes[i].is_alive():
                    logger.info(f"Process {i} has finished")
                    active_processes.remove(i)

            # Try to collect results from queue
            try:
                if not results_queue.empty():
                    result = results_queue.get_nowait()
                    results.append(result)
                    collected_results.add(result['gpu_id'])
                    logger.info(f"✓ Collected result from GPU {result['gpu_id']} ({len(results)}/{len(processes)})")
            except:
                pass

            # Try to collect errors from queue
            try:
                if not error_queue.empty():
                    error = error_queue.get_nowait()
                    errors.append(error)
                    logger.error(f"✗ Collected error from GPU {error['gpu_id']}: {error['error']}")
            except:
                pass

            # If we have collected all expected results, break early
            if len(results) == len(processes):
                logger.info("All results collected!")
                break

            # Small sleep to prevent busy waiting
            time.sleep(0.1)

        # Final cleanup - force join all processes
        logger.info("Final cleanup of worker processes...")
        for i, p in enumerate(processes):
            if p.is_alive():
                logger.warning(f"Process {i} still alive, forcing join...")
                p.join(timeout=30)
                if p.is_alive():
                    logger.warning(f"Process {i} still alive after join, terminating...")
                    p.terminate()
                    p.join(timeout=10)
                    if p.is_alive():
                        logger.error(f"Process {i} could not be terminated, killing...")
                        p.kill()

        # Collect any remaining results
        logger.info("Collecting any remaining results...")
        remaining_collected = 0
        while not results_queue.empty():
            try:
                result = results_queue.get_nowait()
                if result['gpu_id'] not in collected_results:
                    results.append(result)
                    remaining_collected += 1
                    logger.info(f"✓ Collected remaining result from GPU {result['gpu_id']}")
            except:
                break

        # Collect any remaining errors
        while not error_queue.empty():
            try:
                error = error_queue.get_nowait()
                errors.append(error)
                logger.error(f"✗ Collected remaining error from GPU {error['gpu_id']}: {error['error']}")
            except:
                break

        logger.info(f"Final count: {len(results)} results, {len(errors)} errors")
        if remaining_collected > 0:
            logger.info(f"Collected {remaining_collected} additional results in final sweep")

        # Check if we have the expected number of results
        expected_results = len(data_chunks)
        if len(results) != expected_results:
            logger.warning(f"Expected {expected_results} results but got {len(results)}")
            received_gpu_ids = {r['gpu_id'] for r in results}
            expected_gpu_ids = set(range(expected_results))
            missing_gpu_ids = expected_gpu_ids - received_gpu_ids
            logger.warning(f"Missing results from GPUs: {missing_gpu_ids}")

        # Check for errors
        if errors:
            error_msg = f"Errors occurred in {len(errors)} GPU workers: {errors}"
            logger.error(error_msg)
            if len(results) == 0:  # Only raise if we have no results at all
                raise RuntimeError(error_msg)
            else:
                logger.warning("Some workers failed but continuing with available results")

        if len(results) == 0:
            raise RuntimeError("No results collected from any GPU worker")

        # Sort results by GPU ID to maintain order
        results.sort(key=lambda x: x['gpu_id'])

        # Combine embeddings
        logger.info(f"Starting to combine embeddings from {len(results)} GPUs...")
        all_embeddings = []
        total_processed = 0

        for result in results:
            embeddings = result['embeddings']
            chunk_size = result['chunk_size']
            gpu_id = result['gpu_id']

            logger.info(f"Processing result from GPU {gpu_id}: {chunk_size} items, embedding shape: {embeddings.shape}")

            all_embeddings.append(embeddings)
            total_processed += chunk_size

            # Check memory usage
            try:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    logger.info(f"Current GPU memory used: {memory_used:.2f} GB")
            except:
                pass

        # Concatenate all embeddings with memory check
        logger.info(f"Concatenating {len(all_embeddings)} embedding arrays...")
        try:
            combined_embeddings = np.vstack(all_embeddings)
            logger.info(f"✓ Successfully combined embeddings: {combined_embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to combine embeddings: {e}")
            logger.error(f"Embedding shapes: {[emb.shape for emb in all_embeddings]}")
            raise

        # Clear intermediate results to free memory
        logger.info("Cleaning up intermediate results...")
        del all_embeddings
        for result in results:
            if 'embeddings' in result:
                del result['embeddings']  # Free memory from results
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"✓ Final combined embeddings shape: {combined_embeddings.shape}")
        logger.info(f"✓ Total items processed: {total_processed}")

        return combined_embeddings

    def load_persona_data(self, persona_file_paths: str) -> List[Dict]:
        """Load persona data from multiple JSONL files"""
        # Split the comma-separated file paths
        file_paths = [path.strip() for path in persona_file_paths.split(',')]
        logger.info(f"Loading persona data from {len(file_paths)} files: {file_paths}")

        persona_data = []
        total_items = 0

        for file_path in file_paths:
            file_path = file_path.strip()
            if not file_path:
                continue

            logger.info(f"Loading persona data from: {file_path}")
            file_items = 0

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            item = json.loads(line.strip())
                            persona_data.append(item)
                            file_items += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                            continue

                logger.info(f"Loaded {file_items} persona items from {file_path}")
                total_items += file_items

            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue

        logger.info(f"Total loaded {len(persona_data)} persona items from {len(file_paths)} files")
        return persona_data

    def create_query_items(self, survey_type: str = "wvs") -> List[Dict]:
        """Create query items from the 10 specific prompts, based on survey_type"""
        logger.info(f"Creating query items for survey_type: {survey_type}")
        prompts = get_survey_prompts(survey_type)
        query_items = []
        for i, prompt in enumerate(prompts):
            query_item = {
                'text': prompt,
                'query_id': f"prompt_{i+1}",
                'prompt_index': i
            }
            query_items.append(query_item)
        logger.info(f"Created {len(query_items)} query items")
        return query_items

    def create_persona_items(self, persona_data: List[Dict]) -> List[Dict]:
        """Create persona items from persona data"""
        logger.info("Creating persona items from persona data")

        persona_items = []
        for item in persona_data:
            persona_item = {
                'text': item.get('persona', ''),
                'persona_unique_id': item.get('persona_unique_id', ''),
                'source': item.get('source', ''),
                'unique_id': item.get('unique_id', ''),
                'full_item': item
            }
            persona_items.append(persona_item)

        logger.info(f"Created {len(persona_items)} persona items")
        return persona_items

    def find_top_similar_personas(self, query_embeddings: np.ndarray,
                                 persona_embeddings: np.ndarray,
                                 query_items: List[Dict], persona_items: List[Dict],
                                 top_k: int = 100, chunk_size: int = 1000) -> List[Dict]:
        """
        Find top-k similar personas for each query

        Args:
            query_embeddings: Query embeddings array
            persona_embeddings: Persona embeddings array
            query_items: List of query items with metadata
            persona_items: List of persona items with metadata
            top_k: Number of top similar personas to return
            chunk_size: Chunk size for processing similarities
        """
        logger.info(f"Finding top-{top_k} similar personas for {len(query_items)} queries")

        results = []

        # Process in chunks to manage memory
        for i in tqdm(range(0, len(query_embeddings), chunk_size), desc="Computing similarities"):
            end_idx = min(i + chunk_size, len(query_embeddings))
            chunk_query_embeddings = query_embeddings[i:end_idx]

            # Compute cosine similarity for this chunk
            similarities = cosine_similarity(chunk_query_embeddings, persona_embeddings)

            # Process each query in the chunk
            for j, query_idx in enumerate(range(i, end_idx)):
                query_item = query_items[query_idx]
                query_id = query_item['query_id']

                # Get similarity scores for this query
                query_similarities = similarities[j]

                # Get indices sorted by similarity (descending)
                sorted_indices = np.argsort(query_similarities)[::-1]

                # Find top personas
                top_personas = []
                for persona_idx in sorted_indices[:top_k]:
                    persona_item = persona_items[persona_idx]

                    top_personas.append({
                        'persona_unique_id': persona_item['persona_unique_id'],
                        'persona': persona_item['text'],
                        'source': persona_item['source'],
                        'unique_id': persona_item['unique_id'],
                        'similarity_score': float(query_similarities[persona_idx])
                    })

                # Create result entry
                result = {
                    "query_id": query_id,
                    "query": query_item['text'],
                    "prompt_index": query_item['prompt_index'],
                    "top_personas": top_personas
                }
                results.append(result)

            # Clear memory
            del similarities
            torch.cuda.empty_cache()
            gc.collect()

        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSONL file"""
        logger.info(f"Saving {len(results)} results to {output_path}")

        # create all parent folders in the output path
        output_dir = Path(output_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logger.info("Results saved successfully")

    def process_prompt_persona_matching(self, persona_file_path: str, output_file: str,
                                      embedding_batch_size: int = 32,
                                      similarity_chunk_size: int = 1000,
                                      top_k: int = 100,
                                      save_embeddings: bool = True,
                                      survey_type: str = "wvs",
                                      cache_dir: str = "embeddings_cache"):
        """
        Run the complete pipeline for matching prompts with persona data
        """
        try:
            start_time = time.time()
            # Load persona data
            persona_data = self.load_persona_data(persona_file_path)
            # Create query and persona items
            query_items = self.create_query_items(survey_type=survey_type)
            persona_items = self.create_persona_items(persona_data)

            # Compute embeddings using multi-GPU
            cache_paths = embedding_cache_paths(cache_dir, self.model_name, survey_type, persona_file_path)
            cache_paths.query_embeddings.parent.mkdir(parents=True, exist_ok=True)
            query_emb_path = cache_paths.query_embeddings
            persona_emb_path = cache_paths.persona_embeddings

            # Check if embeddings already exist
            if save_embeddings and query_emb_path.exists():
                logger.info("Loading existing query embeddings")
                query_embeddings = np.load(query_emb_path)
            else:
                logger.info("Computing query embeddings using multi-GPU")
                query_embeddings = self.compute_embeddings_multi_gpu(
                    query_items,
                    embedding_batch_size=embedding_batch_size,
                    is_query=True
                )
                if save_embeddings:
                    np.save(query_emb_path, query_embeddings)

            if save_embeddings and persona_emb_path.exists():
                logger.info("Loading existing persona embeddings")
                persona_embeddings = np.load(persona_emb_path)
            else:
                logger.info("Computing persona embeddings using multi-GPU")
                persona_embeddings = self.compute_embeddings_multi_gpu(
                    persona_items,
                    embedding_batch_size=embedding_batch_size,
                    is_query=False
                )
                if save_embeddings:
                    np.save(persona_emb_path, persona_embeddings)

            # Find top similar personas
            logger.info("Starting similarity calculation...")
            logger.info(f"Query embeddings shape: {query_embeddings.shape}")
            logger.info(f"Persona embeddings shape: {persona_embeddings.shape}")

            # Check available memory before similarity calculation
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                        logger.info(f"GPU {i} free memory: {free_memory / 1e9:.2f} GB")
                    except:
                        pass

            try:
                results = self.find_top_similar_personas(
                    query_embeddings, persona_embeddings,
                    query_items, persona_items,
                    top_k=top_k, chunk_size=similarity_chunk_size
                )
                logger.info(f"✓ Similarity calculation completed, found {len(results)} results")
            except Exception as e:
                logger.error(f"Similarity calculation failed: {e}")
                raise

            # Save results
            logger.info("Saving results...")
            self.save_results(results, output_file)
            logger.info("✓ Results saved successfully")

            total_time = time.time() - start_time
            logger.info(f"Multi-GPU pipeline completed successfully in {total_time:.2f} seconds!")

        except Exception as e:
            logger.error(f"Multi-GPU pipeline failed: {e}")
            raise

class EmbeddingSimilarityCalculator:
    """
    Efficient embedding similarity calculator using Qwen model with GPU acceleration
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B",
                 use_flash_attention: bool = True, device: str = "auto"):
        """
        Initialize the embedding model

        Args:
            model_name: Hugging Face model name
            use_flash_attention: Whether to use flash attention for better performance
            device: Device to use for computation
        """
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name

        # Set device first
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        if use_flash_attention:
            try:
                # Set model to use fp16 for flash attention compatibility
                self.model = SentenceTransformer(
                    model_name,
                    model_kwargs={
                        "attn_implementation": "flash_attention_2",
                        "torch_dtype": torch.float16  # Add explicit dtype for flash attention
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
                logger.info("Using flash attention with fp16 for better performance")
                self.use_flash_attention = True
            except Exception as e:
                logger.warning(f"Flash attention initialization failed: {e}. Using standard attention.")
                self.model = SentenceTransformer(model_name)
                self.use_flash_attention = False
        else:
            self.model = SentenceTransformer(model_name)
            self.use_flash_attention = False

        # Move model to specified device explicitly
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        if torch.cuda.is_available() and self.device.type == "cuda":
            device_id = self.device.index if self.device.index is not None else 0
            logger.info(f"GPU: {torch.cuda.get_device_name(device_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f} GB")

    def load_persona_data(self, persona_file_paths: str) -> List[Dict]:
        """Load persona data from multiple JSONL files"""
        # Split the comma-separated file paths
        file_paths = [path.strip() for path in persona_file_paths.split(',')]
        logger.info(f"Loading persona data from {len(file_paths)} files: {file_paths}")

        persona_data = []
        total_items = 0

        for file_path in file_paths:
            file_path = file_path.strip()
            if not file_path:
                continue

            logger.info(f"Loading persona data from: {file_path}")
            file_items = 0

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            item = json.loads(line.strip())
                            persona_data.append(item)
                            file_items += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                            continue

                logger.info(f"Loaded {file_items} persona items from {file_path}")
                total_items += file_items

            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue

        logger.info(f"Total loaded {len(persona_data)} persona items from {len(file_paths)} files")
        return persona_data

    def create_query_items(self, survey_type: str = "wvs") -> List[Dict]:
        """Create query items from the 10 specific prompts, based on survey_type"""
        logger.info(f"Creating query items for survey_type: {survey_type}")
        prompts = get_survey_prompts(survey_type)
        query_items = []
        for i, prompt in enumerate(prompts):
            query_item = {
                'text': prompt,
                'query_id': f"prompt_{i+1}",
                'prompt_index': i
            }
            query_items.append(query_item)
        logger.info(f"Created {len(query_items)} query items")
        return query_items

    def create_persona_items(self, persona_data: List[Dict]) -> List[Dict]:
        """Create persona items from persona data"""
        logger.info("Creating persona items from persona data")

        persona_items = []
        for item in persona_data:
            persona_item = {
                'text': item.get('persona', ''),
                'persona_unique_id': item.get('persona_unique_id', ''),
                'source': item.get('source', ''),
                'unique_id': item.get('unique_id', ''),
                'full_item': item
            }
            persona_items.append(persona_item)

        logger.info(f"Created {len(persona_items)} persona items")
        return persona_items



    def compute_embeddings_batch(self, text_items: List[Dict], batch_size: int = 32,
                                is_query: bool = False, save_path: str = None) -> np.ndarray:
        """
        Compute embeddings for text items in batches with improved error handling

        Args:
            text_items: List of text items with metadata
            batch_size: Batch size for processing
            is_query: Whether texts are queries (affects prompt)
            save_path: Path to save embeddings (optional)
        """
        texts = [item['text'] for item in text_items]
        logger.info(f"Computing embeddings for {len(texts)} texts with batch size {batch_size}")

        all_embeddings = []
        failed_batches = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]

            try:
                # Ensure model is on the correct device
                if hasattr(self.model, 'device') and self.model.device != self.device:
                    self.model = self.model.to(self.device)

                if is_query:
                    # Use query prompt for better query embeddings
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        prompt_name="query",
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        device=self.device
                    )
                else:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        device=self.device
                    )

                all_embeddings.append(batch_embeddings)

                # Clear GPU cache periodically
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                failed_batches += 1
                logger.error(f"Error processing batch {i//batch_size}: {e}")

                # If this is a flash attention error and we're using flash attention, disable it
                if "FlashAttention" in str(e) and self.use_flash_attention:
                    logger.warning("FlashAttention error detected. Falling back to standard attention...")
                    self._fallback_to_standard_attention()

                    # Retry this batch with standard attention
                    try:
                        if is_query:
                            batch_embeddings = self.model.encode(
                                batch_texts,
                                prompt_name="query",
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                        else:
                            batch_embeddings = self.model.encode(
                                batch_texts,
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                        all_embeddings.append(batch_embeddings)
                        failed_batches -= 1  # This batch actually succeeded
                    except Exception as retry_e:
                        logger.error(f"Retry failed for batch {i//batch_size}: {retry_e}")

                continue

        if failed_batches > 0:
            logger.warning(f"{failed_batches} batches failed during processing")

        if not all_embeddings:
            raise RuntimeError("All batches failed. No embeddings were computed successfully.")

        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)

        # Save embeddings if path provided
        if save_path:
            logger.info(f"Saving embeddings to {save_path}")
            np.save(save_path, embeddings)

        logger.info(f"Computed embeddings shape: {embeddings.shape}")
        return embeddings

    def _fallback_to_standard_attention(self):
        """Fallback to standard attention by reinitializing the model"""
        logger.info("Reinitializing model with standard attention...")
        model_name = self.model._modules.get('0').__class__.__name__
        # Get the original model name - this is a simplified approach
        # In practice, you might want to store the original model name
        try:
            self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
            self.use_flash_attention = False
            logger.info("Successfully switched to standard attention")
        except Exception as e:
            logger.error(f"Failed to reinitialize model: {e}")



    def find_top_similar_personas(self, query_embeddings: np.ndarray,
                                 persona_embeddings: np.ndarray,
                                 query_items: List[Dict], persona_items: List[Dict],
                                 top_k: int = 100, chunk_size: int = 1000) -> List[Dict]:
        """
        Find top-k similar personas for each query

        Args:
            query_embeddings: Query embeddings array
            persona_embeddings: Persona embeddings array
            query_items: List of query items with metadata
            persona_items: List of persona items with metadata
            top_k: Number of top similar personas to return
            chunk_size: Chunk size for processing similarities
        """
        logger.info(f"Finding top-{top_k} similar personas for {len(query_items)} queries")

        results = []

        # Process in chunks to manage memory
        for i in tqdm(range(0, len(query_embeddings), chunk_size), desc="Computing similarities"):
            end_idx = min(i + chunk_size, len(query_embeddings))
            chunk_query_embeddings = query_embeddings[i:end_idx]

            # Compute cosine similarity for this chunk
            similarities = cosine_similarity(chunk_query_embeddings, persona_embeddings)

            # Process each query in the chunk
            for j, query_idx in enumerate(range(i, end_idx)):
                query_item = query_items[query_idx]
                query_id = query_item['query_id']

                # Get similarity scores for this query
                query_similarities = similarities[j]

                # Get indices sorted by similarity (descending)
                sorted_indices = np.argsort(query_similarities)[::-1]

                # Find top personas
                top_personas = []
                for persona_idx in sorted_indices[:top_k]:
                    persona_item = persona_items[persona_idx]

                    top_personas.append({
                        'persona_unique_id': persona_item['persona_unique_id'],
                        'persona': persona_item['text'],
                        'source': persona_item['source'],
                        'unique_id': persona_item['unique_id'],
                        'similarity_score': float(query_similarities[persona_idx])
                    })

                # Create result entry
                result = {
                    "query_id": query_id,
                    "query": query_item['text'],
                    "prompt_index": query_item['prompt_index'],
                    "top_personas": top_personas
                }
                results.append(result)

            # Clear memory
            del similarities
            torch.cuda.empty_cache()
            gc.collect()

        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSONL file"""
        logger.info(f"Saving {len(results)} results to {output_path}")

        # create all parent folders in the output path
        output_dir = Path(output_path)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logger.info("Results saved successfully")



    def process_prompt_persona_matching(self, persona_file_path: str, output_file: str,
                                      embedding_batch_size: int = 32,
                                      similarity_chunk_size: int = 1000,
                                      top_k: int = 100,
                                      save_embeddings: bool = True,
                                      survey_type: str = "wvs",
                                      cache_dir: str = "embeddings_cache"):
        """
        Run the complete pipeline for matching prompts with persona data
        """
        try:
            # Load persona data
            persona_data = self.load_persona_data(persona_file_path)
            # Create query and persona items
            query_items = self.create_query_items(survey_type=survey_type)
            persona_items = self.create_persona_items(persona_data)

            # Compute embeddings
            cache_paths = embedding_cache_paths(cache_dir, self.model_name, survey_type, persona_file_path)
            cache_paths.query_embeddings.parent.mkdir(parents=True, exist_ok=True)
            query_emb_path = cache_paths.query_embeddings
            persona_emb_path = cache_paths.persona_embeddings

            # Check if embeddings already exist
            if save_embeddings and query_emb_path.exists():
                logger.info("Loading existing query embeddings")
                query_embeddings = np.load(query_emb_path)
            else:
                query_embeddings = self.compute_embeddings_batch(
                    query_items,
                    batch_size=embedding_batch_size,
                    is_query=True,
                    save_path=query_emb_path if save_embeddings else None
                )

            if save_embeddings and persona_emb_path.exists():
                logger.info("Loading existing persona embeddings")
                persona_embeddings = np.load(persona_emb_path)
            else:
                persona_embeddings = self.compute_embeddings_batch(
                    persona_items,
                    batch_size=embedding_batch_size,
                    is_query=False,
                    save_path=persona_emb_path if save_embeddings else None
                )

            # Find top similar personas
            results = self.find_top_similar_personas(
                query_embeddings, persona_embeddings,
                query_items, persona_items,
                top_k=top_k, chunk_size=similarity_chunk_size
            )

            # Save results
            self.save_results(results, output_file)

            logger.info("Pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise




def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Embedding Similarity Calculator for Prompts and Persona Matching")
    parser.add_argument("--persona_file", "-p", required=True, help="Comma-separated list of persona JSONL file paths (e.g. file1.jsonl,file2.jsonl)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU processing")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embeddings")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for similarities")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top similar personas")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--no_save_embeddings", action="store_true", help="Don't save embeddings")
    parser.add_argument("--survey_type", type=str, default="wvs", choices=list(SUPPORTED_SURVEY_TYPES), help="Survey type")
    parser.add_argument("--cache_dir", type=str, default="embeddings_cache", help="Directory for cached embeddings")
    args = parser.parse_args()
    # Choose calculator based on multi-GPU flag
    if args.multi_gpu:
        logger.info("Using Multi-GPU mode")
        calculator = MultiGPUEmbeddingSimilarityCalculator(
            model_name=args.model,
            use_flash_attention=not args.no_flash_attention,
            num_gpus=args.num_gpus
        )
        calculator.process_prompt_persona_matching(
            persona_file_path=args.persona_file,
            output_file=args.output,
            embedding_batch_size=args.batch_size,
            similarity_chunk_size=args.chunk_size,
            top_k=args.top_k,
            save_embeddings=not args.no_save_embeddings,
            survey_type=args.survey_type,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Using Single-GPU mode")
        calculator = EmbeddingSimilarityCalculator(
            model_name=args.model,
            use_flash_attention=not args.no_flash_attention
        )
        calculator.process_prompt_persona_matching(
            persona_file_path=args.persona_file,
            output_file=args.output,
            embedding_batch_size=args.batch_size,
            similarity_chunk_size=args.chunk_size,
            top_k=args.top_k,
            save_embeddings=not args.no_save_embeddings,
            survey_type=args.survey_type,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
