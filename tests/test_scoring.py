"""Test known keyed response patterns and atomic failure, without APIs."""
import json
from pathlib import Path
import tempfile
import unittest

from popalign.scoring import REVERSED, TRAITS, main, score_row


class ScoringTests(unittest.TestCase):
    def test_known_positive_negative_and_neutral_profiles(self):
        high = {f'{t}{i}': 1 if i in REVERSED[t] else 5 for t in TRAITS for i in range(1, 11)}
        low = {key: 6 - value for key, value in high.items()}
        neutral = {key: '3' for key in high}
        self.assertEqual(score_row(high), dict.fromkeys(TRAITS, 5.0))
        self.assertEqual(score_row(low), dict.fromkeys(TRAITS, 1.0))
        self.assertEqual(score_row(neutral), dict.fromkeys(TRAITS, 3.0))
        all_fives = {f'{t}{i}': 5 for t in TRAITS for i in range(1, 11)}
        self.assertEqual(score_row(all_fives), {'EXT': 3.0, 'AGR': 3.4, 'CSN': 3.4, 'EST': 1.8, 'OPN': 3.8})
        # EST1 (stress) is negative; EST2 (relaxed) is positive.
        neutral.update(EST1=1, EST2=5, id=0)
        self.assertEqual(score_row(neutral)['EST'], 3.4)
        self.assertEqual(score_row(neutral)['id'], 0)

    def test_invalid_or_incomplete_responses_are_not_silently_imputed(self):
        neutral = {f'{t}{i}': 3 for t in TRAITS for i in range(1, 11)}
        for value in (None, True, 'Answer: 4', 0, 6, 2.5, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_row({**neutral, 'AGR3': value})
        del neutral['EXT1']
        with self.assertRaises(ValueError):
            score_row(neutral)

    def test_bad_later_row_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = {f'{t}{i}': 3 for t in TRAITS for i in range(1, 11)}
            inp, out = root/'input.jsonl', root/'scores.jsonl'
            inp.write_text(json.dumps(data) + '\n{}\n')
            with self.assertRaises(SystemExit):
                main(['--input', str(inp), '--output', str(out)])
            self.assertFalse(out.exists())
            self.assertEqual(list(root.iterdir()), [inp])
            inp.write_text(json.dumps(data) + '\n')
            main(['--input', str(inp), '--output', str(out)])
            self.assertEqual(json.loads(out.read_text())['EST'], 3)
            metadata = json.loads(out.with_name(out.name+'.metadata.json').read_text())
            self.assertEqual(metadata['rows'], 1)
            self.assertEqual(len(metadata['input_sha256']), 64)
