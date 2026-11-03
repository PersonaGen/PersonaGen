import tempfile
import unittest
from pathlib import Path

from popalign.config import load_config
from popalign.generation import prompts
from popalign.io import read_jsonl_list, write_jsonl
from popalign.paths import PROJECT_ROOT, project_path


class SmokeTests(unittest.TestCase):
    def test_config_loads(self):
        config = load_config("configs/bigfive_persona.yaml")
        self.assertEqual(config["data"]["question_file"], "examples/ipip50_questions.jsonl")

    def test_project_paths_resolve_from_repo_root(self):
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").exists())
        self.assertTrue((project_path("configs") / "bigfive_persona.yaml").exists())

    def test_prompt_builders(self):
        direct = prompts.bigfive_direct("I am the life of the party.")
        persona = prompts.bigfive_with_persona("I am organized.", "A careful student.")
        self.assertEqual(direct[0]["role"], "system")
        self.assertIn("A careful student.", persona[0]["content"])

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": "x"}])
            self.assertEqual(read_jsonl_list(path), [{"a": 1}, {"b": "x"}])

    def test_wvs_refinement_map_covers_supported_wvs_prompts(self):
        from popalign.embeddings.persona_generation import WVS_REFINEMENT_PROMPT_MAP
        from popalign.embeddings.prompts import (
            SPECIFIC_PROMPTS,
            SPECIFIC_PROMPTS_WVS_EAST,
            SPECIFIC_PROMPTS_WVS_EUROPE,
            SPECIFIC_PROMPTS_WVS_NORTH_AMERICA,
        )

        prompts = (
            SPECIFIC_PROMPTS
            + SPECIFIC_PROMPTS_WVS_EAST
            + SPECIFIC_PROMPTS_WVS_EUROPE
            + SPECIFIC_PROMPTS_WVS_NORTH_AMERICA
        )
        missing = [prompt for prompt in prompts if prompt not in WVS_REFINEMENT_PROMPT_MAP]

        self.assertEqual(missing, [])

    def test_survey_prompt_selector_supports_all_public_survey_types(self):
        from popalign.embeddings.prompts import get_survey_prompts

        expected_counts = {
            "wvs": 10,
            "wvs_east": 10,
            "wvs_west": 10,
            "wvs_north_america": 10,
            "yrbss": 10,
        }

        for survey_type, count in expected_counts.items():
            self.assertEqual(len(get_survey_prompts(survey_type)), count)

        with self.assertRaises(ValueError):
            get_survey_prompts("unknown")

    def test_embedding_cache_paths_are_scoped_by_model_survey_and_persona(self):
        from popalign.embeddings.cache import embedding_cache_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            persona_a = Path(tmpdir) / "persona_a.jsonl"
            persona_b = Path(tmpdir) / "persona_b.jsonl"
            persona_a.write_text('{"persona": "a"}\n', encoding="utf-8")
            persona_b.write_text('{"persona": "b"}\n', encoding="utf-8")

            base = embedding_cache_paths(tmpdir, "Qwen/Qwen3-Embedding-0.6B", "wvs", persona_a)
            other_survey = embedding_cache_paths(tmpdir, "Qwen/Qwen3-Embedding-0.6B", "yrbss", persona_a)
            other_model = embedding_cache_paths(tmpdir, "Other/Embedding", "wvs", persona_a)
            other_persona = embedding_cache_paths(tmpdir, "Qwen/Qwen3-Embedding-0.6B", "wvs", persona_b)

            self.assertEqual(base.query_embeddings.parent, Path(tmpdir))
            self.assertNotEqual(base.query_embeddings, other_survey.query_embeddings)
            self.assertNotEqual(base.persona_embeddings, other_model.persona_embeddings)
            self.assertNotEqual(base.persona_embeddings, other_persona.persona_embeddings)


if __name__ == "__main__":
    unittest.main()
