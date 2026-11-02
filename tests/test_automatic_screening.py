import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automatic_screening import automatic_invalidity


class AutomaticScreeningTests(unittest.TestCase):
    def test_stage1_timeout_boundary(self):
        rows = [{"stage": "stage1_timeout", "timeout_flag": True}] * 3
        self.assertEqual(automatic_invalidity(rows[:2]), [])
        self.assertIn("automatic_stage1_repeated_timeout", automatic_invalidity(rows))

    def test_single_stage2_timeout_stays_for_review(self):
        for q, a in [(True, False), (False, True)]:
            self.assertEqual(automatic_invalidity([{"stage": "stage2_final_guess", "query_timeout_flag": q,
                "answer_timeout_flag": a, "timeout_flag": True}]), [])
        self.assertIn("automatic_stage2_both_phases_timeout", automatic_invalidity([
            {"stage": "stage2_final_guess", "query_timeout_flag": True, "answer_timeout_flag": True}]))

    def test_fast_streak_requires_complete_trials(self):
        fast = {"stage": "stage1_passive_update", "stage1_trial_elapsed_sec": 4.9}
        normal = {"stage": "stage1_passive_update", "stage1_trial_elapsed_sec": 5}
        self.assertEqual(automatic_invalidity([fast]*4), [])
        self.assertTrue(automatic_invalidity([fast]*5))
        self.assertEqual(automatic_invalidity([fast]*4+[normal]+[fast]*4), [])
        self.assertEqual(automatic_invalidity([{ "stage": "stage1_passive_update"}]*20), [])

    def test_stage2_fast_streak_uses_trial_duration_not_query_count_or_aha(self):
        fast = {"stage": "stage2_final_guess", "trial_elapsed_sec": 9.9, "aha": 0, "n_queries": 0}
        self.assertEqual(automatic_invalidity([fast]*2), [])
        self.assertTrue(automatic_invalidity([fast]*3))
        self.assertEqual(automatic_invalidity([dict(fast, trial_elapsed_sec=30)]*5), [])

    def test_practice_and_query_events_do_not_trigger(self):
        self.assertEqual(automatic_invalidity([{"stage": "practice_stage1", "stage1_trial_elapsed_sec": 0}]*10), [])
        self.assertEqual(automatic_invalidity([{"stage": "stage2_active_query", "trial_elapsed_sec": 0}]*10), [])
