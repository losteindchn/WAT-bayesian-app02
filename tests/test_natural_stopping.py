"""Synthetic offline regression checks; never writes to the production Google Sheet."""
import json
import sys
import time
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Route every synthetic local event away from real experiment logs.
import tempfile
import common
_TEST_LOGS = tempfile.TemporaryDirectory(prefix="wat_synthetic_tests_")
_original_append_jsonl = common.append_jsonl
common.append_jsonl = lambda path, row: _original_append_jsonl(Path(_TEST_LOGS.name) / Path(path).name, row)


def button(at, prefix):
    return next(b for b in at.button if b.label.startswith(prefix))


def ready():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    at.secrets.clear()
    # Explicit test-only constants; no service-account credentials.
    at.secrets["materials_file"] = "materials_for_app_balanced_stage1.json"
    at.secrets["stage1_lookup_prefix"] = "lookup_stage1_balanced"
    at.secrets["min_queries"] = 3  # The old deployed secret must not reintroduce a gate.
    at.query_params.update({"group": "FH", "source": "test"})
    at.run(timeout=30)
    button(at, "开始实验").click().run(timeout=30)
    assert at.session_state["min_queries"] == 0
    materials = json.loads((ROOT / "data/materials_for_app_balanced_stage1.json").read_text())
    index = next(i for i, x in enumerate(materials) if x.get("suggested_stage") == "stage2")
    state = at.session_state
    state["pid"] = "TEST-NATURAL-STOPPING"
    state["page"] = "stage2"
    state["order_stage2"] = [index]
    state["stage2_idx"] = 0
    state["stage2_n"] = 1
    state["stage1_n"] = 1
    state["responses"] = [{"stage": "stage1_passive_update"}]
    state["stage2_phase"] = "query"
    state["experiment_started_at"] = time.time() - 1000
    state["stage2_query_timeout_sec"] = 480
    state["stage2_answer_timeout_sec"] = 240
    at.run(timeout=30)
    assert not at.exception, [e.message for e in at.exception]
    return at, materials[index]


class NaturalStoppingTests(unittest.TestCase):
    def assert_clean(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_old_sheet_columns_remain_at_the_same_indices(self):
        import ast
        src = ast.parse((ROOT / "streamlit_app.py").read_text())
        cols = next(ast.literal_eval(n.value) for n in src.body if isinstance(n, ast.Assign)
                    and any(isinstance(x, ast.Name) and x.id == "SHEET_COLUMNS" for x in n.targets))
        expected_prefix = ["timestamp", "app_version", "materials_file", "stage1_lookup_prefix", "full_lookup_prefix",
            "stage1_design_version", "session_id", "participant_id", "platform_participant_id", "platform_source",
            "condition", "desktop_confirmed", "screen_timeout_sec", "timeout_flag", "invalid_reason",
            "quality_status", "quality_pass", "quality_flags", "quality_summary", "payment_eligible",
            "completion_code", "completion_code_hash", "review_code", "group", "age", "native_chinese", "page",
            "stage", "trial_set", "item_id", "title", "query_index", "query_raw", "query_normalized", "matched_word",
            "match_type", "match_confidence", "feedback_score", "feedback_prob", "prior", "updated", "confidence",
            "aha", "aha_suddenness", "aha_surprise", "known_story", "n_queries", "guess", "total_duration_sec",
            "stage1_valid_count", "stage1_timeout_count", "stage2_query_count", "stage2_final_count",
            "stage2_timeout_count", "empty_query_attempts", "duplicate_query_attempts", "no_feedback_query_attempts",
            "invalid_query_attempts", "elapsed_on_screen_sec", "payload_json"]
        self.assertEqual(cols[:len(expected_prefix)], expected_prefix)
        self.assertEqual(len(cols), len(set(cols)))

    def test_sheet_migration_preserves_an_unexpected_old_header(self):
        import ast
        import types
        src = ast.parse((ROOT / "streamlit_app.py").read_text())
        cols = next(ast.literal_eval(n.value) for n in src.body if isinstance(n, ast.Assign)
                    and any(isinstance(x, ast.Name) and x.id == "SHEET_COLUMNS" for x in n.targets))
        func = next(n for n in src.body if isinstance(n, ast.FunctionDef) and n.name == "init_gsheet")
        func.decorator_list = []
        class MissingSheet(Exception): pass
        class Sheet:
            def __init__(self, header):
                self.header = header[:]
                self.col_count = len(header)
                self.updated = False
            def row_values(self, row): return self.header
            def resize(self, cols): self.col_count = cols
            def update(self, range_name, values):
                self.updated = True
                self.header = values[0]
        for header, expect_new in [(cols[:cols.index("payload_json")+1], False), (["an_unrelated_header"], True)]:
            old = Sheet(header)
            sheets = {"human_events_v2": old}
            class Book:
                def worksheet(self, name):
                    if name not in sheets: raise MissingSheet()
                    return sheets[name]
                def add_worksheet(self, title, rows, cols):
                    sheets[title] = Sheet([])
                    return sheets[title]
            book = Book()
            client = types.SimpleNamespace(open_by_url=lambda url: book)
            fake_gspread = types.SimpleNamespace(service_account_from_dict=lambda account: client)
            fake_st = types.SimpleNamespace(secrets={"gcp_service_account": {}, "gsheet_url": "test"})
            ns = {"st": fake_st, "gspread": fake_gspread, "WorksheetNotFound": MissingSheet,
                  "get_secret": lambda name, default=None: default, "SHEET_COLUMNS": cols}
            exec(compile(ast.Module(body=[func], type_ignores=[]), "test_sheet", "exec"), ns)
            result = ns["init_gsheet"]()
            self.assertEqual(result.header, cols)
            self.assertGreaterEqual(result.col_count, len(cols))
            if expect_new:
                self.assertIsNot(result, old)
                self.assertFalse(old.updated)
            else:
                self.assertIs(result, old)

    def test_query_limit_keeps_both_stopping_routes_available(self):
        at, item = ready()
        at.session_state["max_queries"] = 0
        at.run()
        self.assert_clean(at)
        self.assertTrue(button(at, "我形成了一个解释"))
        self.assertTrue(button(at, "暂时无法解答"))
        self.assertFalse(any(b.label == "查询这个词" for b in at.button))

    def test_zero_query_solution_and_monotonic_clocks(self):
        at, _ = ready()
        trial = at.session_state["stage2_trial"]
        started = trial["started_at"]
        trial["started_at"] -= 12
        trial["query_started_at"] -= 12
        at.session_state["stage2_trial"] = trial
        button(at, "我形成了一个解释").click().run(timeout=30)
        self.assert_clean(at)
        self.assertEqual(at.session_state["stage2_phase"], "answer")
        stop = next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_search_stop")
        self.assertEqual(stop["n_queries"], 0)
        self.assertEqual(stop["stop_reason"], "explanation_formed")
        self.assertGreaterEqual(stop["query_phase_elapsed_sec"], 12)
        frozen_search = stop["query_phase_elapsed_sec"]
        at.text_area[0].set_value("一个测试解释").run()
        button(at, "提交并进入下一题").click().run(timeout=30)
        self.assert_clean(at)
        final = next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_final_guess")
        self.assertEqual(final["query_phase_elapsed_sec"], frozen_search)
        self.assertGreaterEqual(final["trial_elapsed_sec"], final["query_phase_elapsed_sec"])
        self.assertEqual(final["trial_id"], stop["trial_id"])
        self.assertTrue(final["has_written_guess"])
        completion = at.session_state["responses"][-1]
        self.assertEqual(completion["stage"], "completion_valid")

    def test_unsolved_can_finish_without_a_fabricated_answer(self):
        at, _ = ready()
        button(at, "暂时无法解答").click().run(timeout=30)
        button(at, "提交并进入下一题").click().run(timeout=30)
        self.assert_clean(at)
        final = next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_final_guess")
        self.assertEqual(final["self_report_status"], "not_solved")
        self.assertEqual(final["guess"], "")
        self.assertFalse(final["timeout_flag"])
        self.assertEqual(at.session_state["responses"][-1]["stage"], "completion_valid")

    def test_formed_explanation_requires_text(self):
        at, _ = ready()
        button(at, "我形成了一个解释").click().run(timeout=30)
        button(at, "提交并进入下一题").click().run(timeout=30)
        self.assert_clean(at)
        self.assertTrue(at.warning)
        self.assertFalse(any(r.get("stage") == "stage2_final_guess" for r in at.session_state["responses"]))

    def test_queries_and_rerenders_do_not_reset_trial_clock(self):
        at, item = ready()
        trial_start = at.session_state["stage2_trial"]["started_at"]
        lookup = json.loads((ROOT / "data/lookup_full_FH.json").read_text())
        word = next(w for w in lookup[item["item_id"]]["scores"] if w != item["network_target_word"])
        at.text_input[0].set_value(word).run()
        button(at, "查询这个词").click().run(timeout=30)
        self.assert_clean(at)
        self.assertEqual(at.session_state["stage2_trial"]["started_at"], trial_start)
        records = at.session_state["responses"]
        q = next(r for r in records if r.get("stage") == "stage2_active_query")
        fb = next(r for r in records if r.get("stage") == "stage2_feedback_ready")
        self.assertLessEqual(q["query_submitted_at_unix"], q["feedback_computed_at_unix"])
        self.assertLessEqual(q["feedback_computed_at_unix"], fb["feedback_render_ready_at_unix"])
        self.assertEqual(q["feedback_score"], max(0, min(100, round(100 * (1 + __import__("math").log10(q["feedback_prob"]) / 6)))))
        at.run()
        self.assertEqual(sum(r.get("stage") == "stage2_feedback_ready" for r in at.session_state["responses"]), 1)
        button(at, "我形成了一个解释").click().run(timeout=30)
        stop = next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_search_stop")
        self.assertEqual(stop["n_queries"], 1)

    def test_timeout_blocks_new_queries_but_retains_final_response(self):
        at, _ = ready()
        trial = at.session_state["stage2_trial"]
        trial["started_at"] -= 500
        trial["query_started_at"] -= 500
        at.session_state["stage2_trial"] = trial
        at.run(timeout=30)
        self.assert_clean(at)
        self.assertEqual(at.session_state["stage2_phase"], "answer")
        stop = next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_search_stop")
        self.assertEqual(stop["stop_reason"], "query_timeout")
        trial = at.session_state["stage2_trial"]
        trial["answer_started_at"] -= 250
        at.session_state["stage2_trial"] = trial
        button(at, "提交并进入下一题").click().run(timeout=30)
        self.assert_clean(at)
        final = next(r for r in at.session_state["responses"] if r.get("stage") == "stage2_final_guess")
        self.assertTrue(final["query_timeout_flag"] and final["answer_timeout_flag"])
        self.assertEqual(at.session_state["responses"][-1]["stage2_timeout_count"], 1)
        self.assertEqual(at.session_state["responses"][-1]["stage"], "completion_review")

    def test_next_trial_has_new_clock_and_id(self):
        at, item = ready()
        old_id = at.session_state["stage2_trial"]["trial_id"]
        at.session_state["order_stage2"] *= 2
        at.session_state["stage2_n"] = 2
        button(at, "暂时无法解答").click().run()
        button(at, "提交并进入下一题").click().run(timeout=30)
        self.assert_clean(at)
        self.assertEqual(at.session_state["stage2_phase"], "query")
        self.assertNotEqual(at.session_state["stage2_trial"]["trial_id"], old_id)
        self.assertIsNone(at.session_state["stage2_trial"]["search_stopped_at"])
        self.assertEqual(at.session_state["explore_history"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
