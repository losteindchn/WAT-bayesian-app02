"""Offline checks of response-to-query alignment and required explicit responses."""
import json
import unittest
from test_natural_stopping import ready, button, ROOT
from aha_location import aha_location_options, AHA_LOCATION_VERSION


class AhaLocationTests(unittest.TestCase):
    def radio(self, at):
        return next(r for r in at.radio if r.label.startswith("如果本题出现过"))

    def final(self, at):
        return next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_final_guess")

    def test_zero_queries_no_default_and_missing_response_blocks_submission(self):
        at, _ = ready()
        button(at, "暂时无法解答").click().run()
        radio = self.radio(at)
        self.assertIsNone(radio.value)
        self.assertEqual(len(radio.options), 4)
        self.assertFalse(any("第 1 次" in x for x in radio.options))
        next(b for b in at.button if b.label == "提交并进入下一题").click().run()
        self.assertEqual(at.session_state["stage2_phase"], "answer")
        self.assertFalse(any(r.get("stage") == "stage2_final_guess" for r in at.session_state["responses"]))
        self.assertTrue(at.warning)
        self.radio(at).set_value("before_search")
        button(at, "提交并进入下一题").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(self.final(at)["aha_location"], "before_search")
        self.assertIsNone(self.final(at)["aha_query_index"])

    def test_actual_feedback_index_is_saved_with_history(self):
        at, item = ready()
        lookup = json.loads((ROOT / "data/lookup_full_FH.json").read_text())
        words = [w for w in lookup[item["item_id"]]["scores"] if w != item["network_target_word"]][:2]
        for word in words:
            at.text_input[0].set_value(word).run()
            button(at, "查询这个词").click().run()
        button(at, "暂时无法解答").click().run()
        self.assertEqual(len(self.radio(at).options), 6)
        self.radio(at).set_value("after_feedback_2")
        button(at, "提交并进入下一题").click().run()
        self.assertFalse(at.exception)
        final = self.final(at)
        self.assertEqual(final["aha_location"], "after_feedback")
        self.assertEqual(final["aha_query_index"], 2)
        self.assertEqual(final["query_history"][1]["query_raw"], words[1])
        self.assertEqual(final["aha_location_measure_version"], AHA_LOCATION_VERSION)

    def test_unknown_and_absent_are_distinct_and_not_exclusion_flags(self):
        for code in ("unknown", "none", "answer_phase"):
            with self.subTest(code=code):
                at, _ = ready()
                button(at, "暂时无法解答").click().run()
                self.radio(at).set_value(code)
                button(at, "提交并进入下一题").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(self.final(at)["aha_location"], code)
                self.assertIsNone(self.final(at)["aha_query_index"])
                self.assertEqual(at.session_state["responses"][-1]["stage"], "completion_valid")

    def test_all_eight_queries_available_without_invented_timestamp(self):
        options = aha_location_options(8)
        self.assertEqual(len(options), 12)
        self.assertEqual(options["after_feedback_8"]["query_index"], 8)
        self.assertNotIn("after_feedback_9", options)
        self.assertTrue(all(set(v) == {"category", "query_index", "label"} for v in options.values()))


if __name__ == "__main__":
    unittest.main()
