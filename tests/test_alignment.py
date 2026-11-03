"""Numerical checks for the new reference sampler; no APIs or model weights."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp

from popalign.alignment import align, sinkhorn
from popalign.alignment.__main__ import main, read_csv


class AlignmentTests(unittest.TestCase):
    def test_sinkhorn_matches_analytic_two_point_solution(self):
        epsilon = 0.2
        cost = np.array([[0., 1.], [1., 0.]])
        plan, _, error = sinkhorn(cost, epsilon=epsilon)
        diagonal = 0.5 / (1 + np.exp(-1 / epsilon))
        assert_allclose(plan, [[diagonal, 0.5 - diagonal], [0.5 - diagonal, diagonal]], atol=1e-12)
        assert_allclose(plan.sum(axis=0), [.5, .5], atol=1e-9)
        assert_allclose(plan.sum(axis=1), [.5, .5], atol=1e-9)
        self.assertLessEqual(error, 1e-9)

    def test_importance_density_ratio_matches_explicit_gaussian(self):
        persona = np.array([[-1.], [0.], [1.]])
        human = np.array([[.5], [1.]])
        h = .6
        result = align(persona, human, bandwidth=h, n_output=12)
        log_human = logsumexp(-((persona[:, None] - human[None]) ** 2).sum(2) / (2 * h*h), axis=1) - np.log(len(human))
        log_persona = logsumexp(-((persona[:, None] - persona[None]) ** 2).sum(2) / (2 * h*h), axis=1) - np.log(len(persona))
        ratio = log_human - log_persona
        expected = np.exp(ratio - logsumexp(ratio))
        assert_allclose(result.importance_probabilities, expected, rtol=1e-12)

    def test_legacy_default_is_mass_weighted_cost(self):
        persona = np.array([[0., 0.], [1., 2.], [2., 1.]])
        human = np.array([[0., 0.], [1., 1.]])
        temperature = .7
        r = align(persona, human, n_output=20,
                  feature_weights=[2., 3.], temperature=temperature, epsilon=2., seed=12)
        expected_cost = ((persona[r.candidate_rows, None] - human[None]) ** 2 * [2., 3.]).sum(2)
        assert_allclose(r.cost, expected_cost)
        expected_c = (r.plan * expected_cost).sum(axis=1)
        assert_allclose(r.transport_costs, expected_c)
        logits = -expected_c / temperature
        expected_p = np.exp(logits - logsumexp(logits))
        assert_allclose(r.transport_probabilities, expected_p)
        conditional_logits = -(expected_c * len(r.candidate_rows)) / temperature
        self.assertGreater(np.max(np.abs(expected_p - np.exp(conditional_logits - logsumexp(conditional_logits)))), .01)
        assert_allclose(r.plan.sum(1), np.full(2, 1/2), atol=1e-9)
        assert_allclose(r.plan.sum(0), [.5, .5], atol=1e-9)
        assert_array_equal(r.selected_rows, r.candidate_rows[r.selected_candidates])

    def test_appendix_cost_is_explicit_and_rescales_by_candidate_mass(self):
        persona = [[0.], [1.], [3.]]
        human = [[1.], [2.]]
        kwargs = dict(n_output=20, seed=9, temperature=.4, epsilon=.4)
        main_text = align(persona, human, **kwargs)
        appendix = align(persona, human, **kwargs, transport_cost_convention="appendix")
        assert_array_equal(main_text.candidate_rows, appendix.candidate_rows)
        assert_allclose(main_text.plan, appendix.plan)
        assert_allclose(appendix.transport_costs, main_text.transport_costs * 2)
        expected = -appendix.transport_costs / kwargs['temperature']
        assert_allclose(appendix.transport_probabilities, np.exp(expected - logsumexp(expected)))
        with self.assertRaisesRegex(ValueError, "convention"):
            align(persona, human, transport_cost_convention="unknown")

    def test_top_seventy_percent_matches_known_density_ranking(self):
        persona = np.arange(10, dtype=float).reshape(-1, 1)
        human = np.array([[0.]])
        # For this symmetric, equally spaced pool and human point at zero,
        # the first seven rows have the highest human/persona density ratios.
        a = align(persona, human, bandwidth=1., epsilon=1., seed=1)
        b = align(persona, human, bandwidth=1., epsilon=1., seed=99)
        assert_array_equal(a.candidate_rows, np.arange(7))
        assert_array_equal(b.candidate_rows, a.candidate_rows)
        self.assertEqual(len(np.unique(a.candidate_rows)), 7)

    def test_ranking_survives_probability_underflow(self):
        persona = np.arange(9, -1, -1, dtype=float).reshape(-1, 1)
        result = align(persona, [[0.]], bandwidth=.1, epsilon=1.)
        self.assertTrue(np.any(result.importance_probabilities == 0))
        assert_array_equal(result.candidate_rows, np.arange(9, 2, -1))

    def test_fraction_rounding_ties_and_final_replacement(self):
        # Exact ties preserve input order. 0.7 * 90 rounds below 63 in binary
        # floating point, but the paper's floor(0.7 N) must still give 63.
        for n, expected in [(3, 2), (10, 7), (90, 63)]:
            x = np.ones((n, 1))
            a = align(x, [[1.]], n_output=100, seed=123)
            b = align(x, [[1.]], n_output=100, seed=123)
            assert_array_equal(a.candidate_rows, np.arange(expected))
            assert_array_equal(a.selected_rows, b.selected_rows)
            self.assertLess(len(np.unique(a.selected_rows)), len(a.selected_rows))
        single = align([[1.]], [[2.]], retain_fraction=1., n_output=5)
        assert_array_equal(single.selected_rows, np.zeros(5, dtype=int))
        assert_allclose(single.transport_costs, [1.])
        assert_allclose(single.transport_probabilities, [1.])
        with self.assertRaisesRegex(ValueError, 'zero candidates'):
            align([[1.]], [[2.]])

    def test_invalid_inputs_and_dense_memory_guard(self):
        for persona, human, kwargs in [
            ([], [[1.]], {}),
            ([[np.nan]], [[1.]], {}),
            ([[1., 2.]], [[1.]], {}),
            ([[1.], [2.]], [[1.]], {'bandwidth': 0}),
            ([[1.], [2.]], [[1.]], {'temperature': -1}),
            ([[1.], [2.]], [[1.]], {'feature_weights': [0]}),
            ([[1.], [2.]], [[1.]], {'feature_weights': [1, 2]}),
            ([[1.], [2.]], [[1.]], {'retain_fraction': 1.1}),
            ([[1.], [2.]], [[1.]], {'retain_fraction': 0}),
            ([[1.], [2.]], [[1.]], {'retain_fraction': float('nan')}),
            ([[1.], [2.]], [[1.]], {'n_output': 0}),
            ([[1.], [2.], [3.]], [[1.], [2.]], {'max_cost_entries': 3}),
        ]:
            with self.subTest(kwargs=kwargs, persona=persona):
                with self.assertRaises(ValueError):
                    align(persona, human, **kwargs)

    def test_nonconvergence_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            sinkhorn([[0., 10.], [1., 2.], [2., 1.]], epsilon=.1,
                     max_iterations=1, tolerance=1e-14)

    def test_cli_preserves_ids_hashes_and_feature_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'persona.csv'
            h = root / 'human.csv'
            p.write_text('id,EXT,EST\na,1,2\nb,2,1\nc,2,2\n')
            h.write_text('id,EST,EXT\nh1,1,1\nh2,2,2\n')
            output = root / 'out'
            main(['--persona-csv',str(p),'--human-csv',str(h),'--output',str(output),
                  '--id-column','id','--columns','EXT','EST',
                  '--n-output','10','--epsilon','0.5','--save-plan','--transport-cost-convention','appendix'])
            meta = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(meta['columns'], ['EXT','EST'])
            self.assertEqual(meta['n_candidates'], 2)
            self.assertEqual(meta['parameters']['retain_fraction'], 0.7)
            with (output / 'candidates.csv').open() as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(len({row['persona_row'] for row in candidates}), 2)
            self.assertEqual(meta['parameters']['transport_cost_convention'], 'appendix')
            self.assertIn('/ a_i', meta['transport_cost_definition'])
            self.assertEqual(len(meta['input_files']['persona']['sha256']), 64)
            self.assertEqual(meta['replacement'], {'importance_sampling':False,'ot_sampling':True})
            with (output / 'selection.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows),10)
            for row in rows:
                self.assertEqual(row['persona_id'], ['a','b','c'][int(row['persona_row'])])
            self.assertTrue((output / 'transport.npz').exists())
            with self.assertRaises(SystemExit) as raised:
                main(['--persona-csv',str(p),'--human-csv',str(h),'--output',str(output)])
            self.assertEqual(raised.exception.code, 2)

    def test_csv_rejects_bad_header_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            for text in ['x,x\n1,2\n', 'x,y\n1\n', 'x\nhello\n', 'x\n']:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_csv(path)


if __name__ == '__main__':
    unittest.main()
