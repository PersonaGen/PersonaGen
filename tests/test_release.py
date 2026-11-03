"""Regression checks for release fixes, with no external services."""
import json
import subprocess
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from scipy.stats import entropy

from popalign.config import load_config
from popalign.evaluation.bigfive_metrics import js_div, tv_dist, hellinger_dist, _joint_hist
from popalign.generation.runner import run_bigfive_persona
from popalign.io import read_jsonl_list, write_jsonl
from popalign.serving import start_vllm_server, stop_process, wait_for_server


class ReleaseTests(unittest.TestCase):
    def test_sparse_metrics_match_dense_histograms_including_edges(self):
        rng = np.random.default_rng(7)
        a = np.vstack([rng.uniform(1, 5, (40, 3)), [1, 5, 3], [5, 5, 5]])
        b = np.vstack([rng.uniform(1, 5, (50, 3)), [1, 1, 1], [0, 3, 3]])
        for bins in (2, 4, 8):
            p = np.histogramdd(a, bins=bins, range=[(1, 5)] * 3)[0].ravel()
            q = np.histogramdd(b, bins=bins, range=[(1, 5)] * 3)[0].ravel()
            p, q = p / p.sum(), q / q.sum()
            m = (p + q) / 2
            self.assertAlmostEqual(js_div(a, b, bins), (entropy(p, m) + entropy(q, m)) / 2)
            self.assertAlmostEqual(tv_dist(a, b, bins), np.abs(p - q).sum() / 2)
            self.assertAlmostEqual(hellinger_dist(a, b, bins), np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2))
        p, q = _joint_hist(np.ones((2, 5)), np.full((2, 5), 5))
        self.assertEqual(len(p), 2)  # Default 40^5 grid is never allocated.
        with self.assertRaises(ValueError):
            _joint_hist(np.zeros((2, 5)), np.ones((2, 5)))

    def test_reference_warp_is_opt_in(self):
        from popalign.evaluation.runner import run_bigfive
        config = {"data": {"human_path": "human.jsonl", "origin_path": "origin.jsonl"},
                  "output": {"dir": "out"}}
        seen = []
        with patch("popalign.evaluation.bigfive_metrics.main", side_effect=lambda: seen.append(list(sys.argv))):
            run_bigfive(config)
            run_bigfive({**config, "quantile_warp_csn": True})
        self.assertNotIn("--quantile_warp_csn", seen[0])
        self.assertIn("--quantile_warp_csn", seen[1])

    def test_json_configs_reject_non_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "mapping"):
                load_config(path)

    def test_yaml_configs_reject_empty_non_mappings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            for value in ("[]", "false", "0", "''"):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "mapping"):
                    load_config(path)

    def test_persona_id_zero_is_preserved_and_missing_id_rejected(self):
        from popalign.generation.runner import _persona_uid, _persona_row
        self.assertEqual(_persona_uid({"id": 0}), "0")
        self.assertEqual(_persona_row({"id": 0, "persona": "Fictional"})['persona_id'], 0)
        self.assertEqual(_persona_uid({"unique_id": "", "id": 1}), "1")
        with self.assertRaisesRegex(ValueError, "safe resumption"):
            _persona_uid({"persona": "Fictional without ID"})

    def test_missing_generation_inputs_fail_before_launching_model(self):
        from popalign.generation.runner import run_ood_persona
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"model": {"name": "fake", "path": "fake-model"},
                      "data": {"survey_path": str(root/'missing'), "persona_path": str(root/'missing')},
                      "output": {"dir": str(root/'out')}}
            with patch('popalign.generation.runner.maybe_vllm_server') as server:
                with self.assertRaises(FileNotFoundError):
                    run_ood_persona(config)
                server.assert_not_called()
            empty = root/'empty'; empty.mkdir()
            config['data']['survey_path'] = str(empty)
            with patch('popalign.generation.runner.maybe_vllm_server') as server:
                with self.assertRaisesRegex(ValueError, "No JSONL"):
                    run_ood_persona(config)
                server.assert_not_called()

    def test_startup_failure_stops_child_and_uses_current_python(self):
        child = Mock()
        with patch("popalign.serving._ensure_port_available"), \
             patch("popalign.serving.subprocess.Popen", return_value=child) as spawn, \
             patch("popalign.serving.wait_for_server", side_effect=RuntimeError("timeout")):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                start_vllm_server("fake-model", "fake", 8000, 1)
            self.assertEqual(spawn.call_args.args[0][0], sys.executable)
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=30)

    def test_existing_listener_is_rejected_before_launch(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with patch("popalign.serving.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    start_vllm_server("fake", "fake", port, 1)
                spawn.assert_not_called()

    def test_dead_child_cannot_pass_readiness(self):
        child = Mock(returncode=1)
        child.poll.return_value = 1
        with patch("popalign.serving.requests.get") as request:
            with self.assertRaisesRegex(RuntimeError, "exited before"):
                wait_for_server(8000, process=child)
            request.assert_not_called()

    def test_zero_training_pairs_fail_without_writing_empty_success(self):
        from popalign.embeddings.runner import run_training_data
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / "personas.jsonl", [{"persona": "Fictional persona.", "unique_id": "p1"}])
            config = {"model": {"name": "test"},
                      "data": {"persona_file": str(root / "personas.jsonl"), "query_type": "long"},
                      "output": {"file": str(root / "pairs.jsonl")}}
            with patch("popalign.embeddings.training_data.generate_long_queries_from_personas", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "No long query pairs"):
                    run_training_data(config)
            self.assertFalse((root / "pairs.jsonl").exists())

    def test_termination_timeout_kills_child(self):
        child = Mock()
        child.wait.side_effect = [subprocess.TimeoutExpired("server", 30), 0]
        stop_process(child)
        child.kill.assert_called_once()

    def test_generation_resumes_without_duplicate_api_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / "questions.jsonl", [{"original_id": "EXT1", "question": "I enjoy company."}])
            write_jsonl(root / "personas.jsonl", [{"unique_id": "p1", "persona": "Fictional example."}])
            config = {"model": {"name": "test"}, "runtime": {"threads": 1},
                      "data": {"question_file": str(root / "questions.jsonl"), "persona_file": str(root / "personas.jsonl")},
                      "output": {"dir": str(root / "out"), "flush_every": 1}}
            with patch("popalign.generation.runner._chat", return_value="4") as chat:
                run_bigfive_persona(config)
                run_bigfive_persona(config)
                self.assertEqual(chat.call_count, 1)
            result = read_jsonl_list(root / "out/personas_bigfive_test.jsonl")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["EXT1"], "4")


if __name__ == "__main__":
    unittest.main()
