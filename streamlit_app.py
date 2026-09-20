#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage Streamlit app for human behavioural experiment."""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from common import GROUP_PROFILES, append_jsonl
from feedback_resolver import load_grounding_config, load_semantic_grounding_index, resolve_query_feedback

try:
    import gspread
    from gspread.exceptions import WorksheetNotFound
except Exception:  # pragma: no cover - keeps local dry-runs usable without gspread.
    gspread = None
    WorksheetNotFound = Exception


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
LOG_DIR = APP_DIR / "logs"


APP_VERSION = "human-two-stage-v2.7-participant-friendly"


DEFAULT_STAGE1_ITEM_IDS_20 = [
    "spx_0034",
    "spx_0771",
    "spx_0150",
    "spx_0384",
    "spx_0901",
    "spx_0026",
    "spx_0628",
    "spx_0823",
    "spx_0531",
    "spx_0797",
    "spx_0819",
    "spx_0209",
    "spx_0240",
    "spx_0013",
    "spx_0049",
    "spx_0788",
    "spx_0151",
    "spx_0283",
    "spx_0561",
    "spx_0195",
]


st.set_page_config(page_title="文字谜题联想实验", layout="centered")

st.markdown(
    """
<style>
section.main > div.block-container {
    max-width: 920px;
    padding-top: 2.2rem;
}
html, body, [class*="css"] {
    font-size: 20px;
}
h1 {
    font-size: 3.05rem !important;
    line-height: 1.18 !important;
    margin-bottom: 1.2rem !important;
}
h2, h3 {
    line-height: 1.3 !important;
}
p, li, label, .stMarkdown, .stCaption {
    line-height: 1.75 !important;
}
div[data-testid="stCaptionContainer"] {
    font-size: 1rem !important;
}
div.stButton > button {
    font-size: 1.15rem;
    padding: 0.55rem 1.05rem;
}
.friendly-note {
    border-left: 7px solid #2f80ed;
    background: #eef6ff;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin: 1rem 0;
    font-size: 1.08rem;
    line-height: 1.7;
}
.friendly-warning {
    border-left: 7px solid #f2994a;
    background: #fff7ed;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin: 1rem 0;
    font-size: 1.08rem;
    line-height: 1.7;
}
.friendly-card {
    background: #f7f8fb;
    border: 1px solid #e4e7ee;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin: 0.9rem 0;
    font-size: 1.05rem;
    line-height: 1.7;
}
.big-word {
    font-size: 1.35rem;
    font-weight: 700;
}
</style>
    """,
    unsafe_allow_html=True,
)


SHEET_COLUMNS = [
    "timestamp",
    "app_version",
    "materials_file",
    "stage1_lookup_prefix",
    "full_lookup_prefix",
    "stage1_design_version",
    "session_id",
    "participant_id",
    "platform_participant_id",
    "platform_source",
    "condition",
    "desktop_confirmed",
    "screen_timeout_sec",
    "timeout_flag",
    "invalid_reason",
    "quality_status",
    "quality_pass",
    "quality_flags",
    "quality_summary",
    "payment_eligible",
    "completion_code",
    "completion_code_hash",
    "review_code",
    "group",
    "age",
    "native_chinese",
    "page",
    "stage",
    "trial_set",
    "item_id",
    "title",
    "query_index",
    "query_raw",
    "query_normalized",
    "matched_word",
    "match_type",
    "match_confidence",
    "feedback_score",
    "feedback_prob",
    "prior",
    "updated",
    "confidence",
    "aha",
    "aha_suddenness",
    "aha_surprise",
    "known_story",
    "n_queries",
    "guess",
    "total_duration_sec",
    "stage1_valid_count",
    "stage1_timeout_count",
    "stage2_query_count",
    "stage2_final_count",
    "stage2_timeout_count",
    "empty_query_attempts",
    "duplicate_query_attempts",
    "no_feedback_query_attempts",
    "invalid_query_attempts",
    "elapsed_on_screen_sec",
    "payload_json",
]


def get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def int_secret(name: str, default: int) -> int:
    try:
        return int(get_secret(name, default))
    except Exception:
        return default


def bool_secret(name: str, default: bool = False) -> bool:
    value = get_secret(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def list_secret(name: str, default: List[str]) -> List[str]:
    value = get_secret(name, default)
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    try:
        return [str(part).strip() for part in value if str(part).strip()]
    except TypeError:
        return list(default)


def query_param(*names: str, default: str = "") -> str:
    try:
        qp = st.query_params
        for name in names:
            value = qp.get(name)
            if isinstance(value, list):
                value = value[0] if value else ""
            if value not in (None, ""):
                return str(value)
    except Exception:
        pass
    try:
        qp = st.experimental_get_query_params()
        for name in names:
            value = qp.get(name)
            if isinstance(value, list):
                value = value[0] if value else ""
            if value not in (None, ""):
                return str(value)
    except Exception:
        pass
    return default


def stable_int(*parts: Any) -> int:
    text = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def completion_code(pid: str, session_id: str, group: str) -> str:
    salt = str(get_secret("completion_salt", "human-two-stage-v2"))
    digest = hashlib.sha256(f"{salt}::{pid}::{session_id}::{group}".encode("utf-8")).hexdigest()
    return f"INS-{digest[:10].upper()}"


def review_code(pid: str, session_id: str, group: str) -> str:
    salt = str(get_secret("completion_salt", "human-two-stage-v2"))
    digest = hashlib.sha256(f"{salt}::review::{pid}::{session_id}::{group}".encode("utf-8")).hexdigest()
    return f"REVIEW-{digest[:10].upper()}"


def completion_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def build_return_url(base_url: str, pid: str, code: str) -> str:
    if not base_url:
        return ""
    if "{completion_code}" in base_url or "{pid}" in base_url:
        return base_url.replace("{completion_code}", code).replace("{pid}", pid)
    try:
        parsed = urlparse(base_url)
        query = parse_qs(parsed.query)
        query["pid"] = [pid]
        query["completion_code"] = [code]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
    except Exception:
        return base_url


def materials_file_name() -> str:
    return str(get_secret("materials_file", "materials_for_app.json"))


def stage1_lookup_prefix() -> str:
    return str(get_secret("stage1_lookup_prefix", "lookup_stage1"))


def full_lookup_prefix() -> str:
    return str(get_secret("full_lookup_prefix", "lookup_full"))


def stage1_item_ids() -> List[str]:
    return list_secret("stage1_item_ids", DEFAULT_STAGE1_ITEM_IDS_20)


@st.cache_data
def load_materials() -> List[Dict[str, Any]]:
    path = DATA_DIR / materials_file_name()
    if not path.exists():
        st.error(f"Missing materials file: {path}")
        st.stop()
    return json.load(open(path, "r", encoding="utf-8"))


@st.cache_data
def load_stage1_lookup(group: str) -> Dict[str, Any]:
    path = DATA_DIR / f"{stage1_lookup_prefix()}_{group}.json"
    if not path.exists():
        st.error(f"Missing lookup file: {path}")
        st.stop()
    return json.load(open(path, "r", encoding="utf-8"))


@st.cache_data
def load_full_lookup(group: str) -> Dict[str, Any]:
    path = DATA_DIR / f"{full_lookup_prefix()}_{group}.json"
    if not path.exists():
        st.error(f"Missing lookup file: {path}")
        st.stop()
    return json.load(open(path, "r", encoding="utf-8"))


@st.cache_data
def load_query_grounding_config() -> Dict[str, Any]:
    path = DATA_DIR / "query_grounding_aliases.json"
    if not path.exists():
        return {}
    return load_grounding_config(path)


@st.cache_resource
def load_semantic_index() -> Optional[Dict[str, Any]]:
    path = DATA_DIR / "semantic_grounding_index.npz"
    if not path.exists():
        return None
    return load_semantic_grounding_index(path)


@st.cache_resource
def init_gsheet():
    if gspread is None:
        return None
    if "gcp_service_account" not in st.secrets or "gsheet_url" not in st.secrets:
        return None

    client = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
    spreadsheet = client.open_by_url(str(st.secrets["gsheet_url"]))
    worksheet_name = str(get_secret("worksheet_name", "human_events_v2"))
    try:
        sheet = spreadsheet.worksheet(worksheet_name)
    except WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=worksheet_name, rows=10000, cols=len(SHEET_COLUMNS))

    first_row = sheet.row_values(1)
    if not first_row or first_row[: len(SHEET_COLUMNS)] != SHEET_COLUMNS:
        sheet.update("A1", [SHEET_COLUMNS])
    return sheet


def infer_group(pid: str, selected: str) -> Optional[str]:
    pid_u = pid.upper()
    for group in GROUP_PROFILES:
        if group in pid_u:
            return group
    return selected if selected in GROUP_PROFILES else None


def init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "intro"
        st.session_state.session_id = hashlib.sha256(f"{datetime.now().isoformat()}::{random.random()}".encode("utf-8")).hexdigest()[:16]
        st.session_state.stage1_idx = 0
        st.session_state.stage2_idx = 0
        st.session_state.stage1_phase = "prior"
        st.session_state.stage2_phase = "query"
        st.session_state.practice1_phase = "prior"
        st.session_state.practice2_history = []
        st.session_state.responses = []
        st.session_state.explore_history = []
        st.session_state.screen_started_at = time.time()
        st.session_state.experiment_started_at = time.time()
        st.session_state.empty_query_attempts = 0
        st.session_state.duplicate_query_attempts = 0
        st.session_state.no_feedback_query_attempts = 0
        st.session_state.invalid_query_attempts = 0


def log_event(row: Dict[str, Any]) -> None:
    row = dict(row)
    row["timestamp"] = datetime.now().isoformat()
    row["app_version"] = APP_VERSION
    row.setdefault("materials_file", materials_file_name())
    row.setdefault("stage1_lookup_prefix", stage1_lookup_prefix())
    row.setdefault("full_lookup_prefix", full_lookup_prefix())
    row["session_id"] = st.session_state.get("session_id", "")
    row.setdefault("platform_participant_id", st.session_state.get("platform_participant_id", ""))
    row.setdefault("platform_source", st.session_state.get("platform_source", ""))
    row.setdefault("condition", st.session_state.get("condition", ""))
    row.setdefault("desktop_confirmed", st.session_state.get("desktop_confirmed", ""))
    row.setdefault("screen_timeout_sec", "")
    row.setdefault("timeout_flag", False)
    row.setdefault("invalid_reason", "")
    row.setdefault("quality_status", st.session_state.get("quality_status", ""))
    row.setdefault("quality_pass", st.session_state.get("quality_pass", ""))
    row.setdefault("quality_flags", st.session_state.get("quality_flags", ""))
    row.setdefault("quality_summary", st.session_state.get("quality_summary", ""))
    row.setdefault("payment_eligible", st.session_state.get("payment_eligible", ""))
    row.setdefault("completion_code", st.session_state.get("completion_code", ""))
    row.setdefault("completion_code_hash", st.session_state.get("completion_code_hash", ""))
    row.setdefault("review_code", st.session_state.get("review_code", ""))
    row.setdefault("age", st.session_state.get("age", ""))
    row.setdefault("native_chinese", st.session_state.get("native_chinese", ""))
    row.setdefault("page", st.session_state.get("page", ""))
    row["elapsed_on_screen_sec"] = round(time.time() - float(st.session_state.get("screen_started_at", time.time())), 3)
    st.session_state.responses.append(row)
    append_jsonl(LOG_DIR / "human_events.jsonl", row)
    append_gsheet(row)
    st.session_state.screen_started_at = time.time()


def append_gsheet(row: Dict[str, Any]) -> None:
    sheet = init_gsheet()
    if sheet is None:
        st.session_state.sheet_status = "local_only"
        return

    payload = json.dumps(row, ensure_ascii=False, default=str)
    values = []
    for key in SHEET_COLUMNS:
        if key == "payload_json":
            values.append(payload)
            continue
        value = row.get(key, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        values.append(value)
    last_error = ""
    for _ in range(3):
        try:
            sheet.append_row(values, value_input_option="RAW")
            st.session_state.sheet_status = "ok"
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.8)
    st.session_state.sheet_status = "failed"
    st.session_state.sheet_error = last_error
    append_jsonl(LOG_DIR / "human_events_unsynced.jsonl", {"error": last_error, "row": row})


def add_researcher_fields(event: Dict[str, Any], item: Dict[str, Any], fb: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    event.update(
        {
            "stage1_design_version": item.get("stage1_design_version", ""),
            "target_word_research_only": item.get("target_word", ""),
            "network_target_word_research_only": item.get("network_target_word", ""),
            "anchor_word_design": item.get("anchor_word", ""),
            "cue_word_design": item.get("cue_word", ""),
            "target_anchor_bin": item.get("target_anchor_bin", ""),
            "target_cue_bin": item.get("target_cue_bin", ""),
            "target_anchor_distance": item.get("target_anchor_distance", ""),
            "target_cue_distance": item.get("target_cue_distance", ""),
        }
    )
    if fb:
        event.update(
            {
                "target_anchor_prob": fb.get("target_anchor_prob", ""),
                "target_cue_prob": fb.get("target_cue_prob", ""),
                "anchor_cue_prob": fb.get("anchor_cue_prob", ""),
                "target_anchor_score": fb.get("target_anchor_score", ""),
                "target_cue_score": fb.get("target_cue_score", ""),
                "anchor_cue_score": fb.get("anchor_cue_score", ""),
            }
        )
    return event


def show_progress(stage: str, idx: int, total: int) -> None:
    st.caption(f"{stage}: {idx + 1} / {total}")
    st.progress(min(1.0, (idx + 1) / max(1, total)))


def reset_screen_timer() -> None:
    st.session_state.screen_started_at = time.time()


def elapsed_screen_sec() -> float:
    return time.time() - float(st.session_state.get("screen_started_at", time.time()))


def is_timed_out(limit_sec: int) -> bool:
    return bool(limit_sec and limit_sec > 0 and elapsed_screen_sec() > float(limit_sec))


def show_time_rule(limit_sec: int) -> None:
    if limit_sec and limit_sec > 0:
        minutes = max(1, round(limit_sec / 60))
        st.caption(f"本页最多约 {minutes} 分钟。请看懂后及时作答。")


def rating_slider(label: str, key: str) -> int:
    return st.slider(f"{label}（0=完全没有，100=非常强）", 0, 100, 50, key=key)


def note(text: str) -> None:
    st.markdown(f'<div class="friendly-note">{text}</div>', unsafe_allow_html=True)


def warning_note(text: str) -> None:
    st.markdown(f'<div class="friendly-warning">{text}</div>', unsafe_allow_html=True)


def card(text: str) -> None:
    st.markdown(f'<div class="friendly-card">{text}</div>', unsafe_allow_html=True)


def clean_query_word(text: Any) -> str:
    return str(text or "").strip()


def query_key(text: Any) -> str:
    return "".join(clean_query_word(text).split()).lower()


def query_already_used(history: List[Dict[str, Any]], query_raw: Any = "", matched_word: Any = "") -> bool:
    current = {query_key(query_raw), query_key(matched_word)}
    current.discard("")
    if not current:
        return False
    for row in history:
        previous = {
            query_key(row.get("query_raw", "")),
            query_key(row.get("query_word", "")),
            query_key(row.get("matched_word", "")),
            query_key(row.get("查询词", "")),
        }
        previous.discard("")
        if current & previous:
            return True
    return False


def increment_counter(name: str) -> None:
    st.session_state[name] = int(st.session_state.get(name, 0)) + 1


def reset_formal_quality_state() -> None:
    st.session_state.experiment_started_at = time.time()
    for name in [
        "empty_query_attempts",
        "duplicate_query_attempts",
        "no_feedback_query_attempts",
        "invalid_query_attempts",
    ]:
        st.session_state[name] = 0


def quality_thresholds() -> Dict[str, int]:
    return {
        "min_total_duration_sec": int_secret("min_total_duration_sec", 900),
        "max_stage1_timeout_count": int_secret("max_stage1_timeout_count", 2),
        "max_stage2_timeout_count": int_secret("max_stage2_timeout_count", 0),
        "max_invalid_query_attempts": int_secret("max_invalid_query_attempts", 8),
        "max_duplicate_query_attempts": int_secret("max_duplicate_query_attempts", 6),
        "max_empty_query_attempts": int_secret("max_empty_query_attempts", 5),
        "max_no_feedback_query_attempts": int_secret("max_no_feedback_query_attempts", 8),
    }


def evaluate_completion_quality() -> Dict[str, Any]:
    responses = list(st.session_state.get("responses", []))
    expected_stage1 = int(st.session_state.get("stage1_n", 0))
    expected_stage2 = int(st.session_state.get("stage2_n", 0))
    total_duration = round(time.time() - float(st.session_state.get("experiment_started_at", time.time())), 3)

    stage1_valid = [r for r in responses if r.get("stage") == "stage1_passive_update"]
    stage1_timeouts = [r for r in responses if r.get("stage") == "stage1_timeout"]
    stage2_queries = [r for r in responses if r.get("stage") == "stage2_active_query"]
    stage2_finals = [r for r in responses if r.get("stage") == "stage2_final_guess"]
    stage2_timeouts = [r for r in responses if r.get("stage") == "stage2_timeout"]
    known_story_yes = [
        r for r in stage2_finals
        if str(r.get("known_story", "")).strip() == "是"
    ]

    counters = {
        "total_duration_sec": total_duration,
        "stage1_valid_count": len(stage1_valid),
        "stage1_timeout_count": len(stage1_timeouts),
        "stage2_query_count": len(stage2_queries),
        "stage2_final_count": len(stage2_finals),
        "stage2_timeout_count": len(stage2_timeouts),
        "empty_query_attempts": int(st.session_state.get("empty_query_attempts", 0)),
        "duplicate_query_attempts": int(st.session_state.get("duplicate_query_attempts", 0)),
        "no_feedback_query_attempts": int(st.session_state.get("no_feedback_query_attempts", 0)),
    }
    counters["invalid_query_attempts"] = (
        counters["empty_query_attempts"]
        + counters["duplicate_query_attempts"]
        + counters["no_feedback_query_attempts"]
        + int(st.session_state.get("invalid_query_attempts", 0))
    )
    thresholds = quality_thresholds()

    flags = []
    if len(stage1_valid) < expected_stage1:
        flags.append("incomplete_stage1")
    if len(stage2_finals) < expected_stage2:
        flags.append("incomplete_stage2")
    if total_duration < thresholds["min_total_duration_sec"]:
        flags.append("too_fast")
    if len(stage1_timeouts) > thresholds["max_stage1_timeout_count"]:
        flags.append("too_many_stage1_timeouts")
    if len(stage2_timeouts) > thresholds["max_stage2_timeout_count"]:
        flags.append("stage2_timeout")
    if counters["invalid_query_attempts"] > thresholds["max_invalid_query_attempts"]:
        flags.append("too_many_invalid_queries")
    if counters["duplicate_query_attempts"] > thresholds["max_duplicate_query_attempts"]:
        flags.append("too_many_duplicate_queries")
    if counters["empty_query_attempts"] > thresholds["max_empty_query_attempts"]:
        flags.append("too_many_empty_queries")
    if counters["no_feedback_query_attempts"] > thresholds["max_no_feedback_query_attempts"]:
        flags.append("too_many_no_feedback_queries")
    if not bool(st.session_state.get("desktop_confirmed", False)):
        flags.append("desktop_not_confirmed")
    if expected_stage2 and len(known_story_yes) >= max(2, expected_stage2):
        flags.append("known_story_all_or_most")

    status = "valid" if not flags else "review"
    summary = dict(counters)
    summary.update(
        {
            "expected_stage1": expected_stage1,
            "expected_stage2": expected_stage2,
            "known_story_yes_count": len(known_story_yes),
            "thresholds": thresholds,
        }
    )
    return {
        "quality_status": status,
        "quality_pass": not flags,
        "payment_eligible": not flags,
        "quality_flags": flags,
        "quality_summary": summary,
        **counters,
    }


init_state()


if st.session_state.page == "intro":
    st.title("文字谜题联想实验")
    video_url = str(get_secret("instruction_video_url", "") or "").strip()
    if video_url:
        note("请先看完下面的讲解视频。视频会演示每一步怎么操作。看完后，再填写页面下方的信息。")
        st.video(video_url)
    else:
        note("这个实验会先带你做两个练习。请按页面提示一步一步完成。")
    st.caption("你的回答将匿名用于科研分析；你可以随时停止实验。")

    pid_from_url = query_param("pid", "participant_id", "participantId", "uid", "id")
    platform_pid = query_param("platform_pid", "external_id", "respondent_id", "wjx_id", "credamo_id", default=pid_from_url)
    platform_source = query_param("source", "platform", "recruitment_source", default="direct")
    condition = query_param("condition", "cond", default=str(get_secret("condition", "balanced_stage1")))
    group_from_url = query_param("group", default="").upper()
    return_url = query_param("return_url", "redirect", default="")

    st.subheader("开始前，请填写基本信息")
    pid = st.text_input("受试者编号", value=pid_from_url, help="请填写问卷平台或研究人员提供的编号。")
    default_group = group_from_url if group_from_url in GROUP_PROFILES else str(get_secret("default_group", "FH")).upper()
    if default_group not in GROUP_PROFILES:
        default_group = "FH"
    default_group_idx = list(GROUP_PROFILES.keys()).index(default_group)
    if bool_secret("show_group_selector", False) or bool_secret("show_admin_controls", False):
        group_choice = st.selectbox("实验分组（请保持默认选项）", list(GROUP_PROFILES.keys()), index=default_group_idx)
    else:
        group_choice = default_group
    age = st.number_input("年龄", min_value=10, max_value=99, value=20)
    native_chinese = st.selectbox("中文熟练程度", ["母语/近似母语", "熟练", "一般"])
    require_desktop = bool_secret("require_desktop", True)
    desktop_confirmed = st.checkbox("我正在使用电脑或笔记本电脑完成实验。")
    if require_desktop:
        st.caption("请不要用手机作答。手机屏幕会影响阅读、输入和查询。")
    else:
        st.caption("当前允许非电脑设备进入，但设备信息会用于后续数据质量审核。")
    consent = st.checkbox("我已了解实验说明，并自愿参加。")

    stage1_n_limit = int_secret("stage1_n_max", len(DEFAULT_STAGE1_ITEM_IDS_20))
    stage1_n = min(int_secret("stage1_n", len(DEFAULT_STAGE1_ITEM_IDS_20)), stage1_n_limit)
    stage2_n = int_secret("stage2_n", 5)
    min_queries = int_secret("min_queries", 3)
    max_queries = int_secret("max_queries", 8)
    stage1_timeout_limit = int_secret("stage1_timeout_max_sec", 120)
    stage1_prior_timeout_sec = min(int_secret("stage1_prior_timeout_sec", 120), stage1_timeout_limit)
    stage1_update_timeout_sec = min(int_secret("stage1_update_timeout_sec", 120), stage1_timeout_limit)
    stage2_query_timeout_sec = int_secret("stage2_query_timeout_sec", 480)
    stage2_answer_timeout_sec = int_secret("stage2_answer_timeout_sec", 240)
    if bool_secret("show_admin_controls", False):
        with st.expander("实验员设置"):
            stage1_n = st.number_input("Stage 1 题数", min_value=1, max_value=max(1, min(80, stage1_n_limit)), value=stage1_n)
            stage2_n = st.number_input("Stage 2 题数", min_value=1, max_value=80, value=stage2_n)
            min_queries = st.number_input("Stage 2 每题最少查询次数", min_value=0, max_value=20, value=min_queries)
            max_queries = st.number_input("Stage 2 每题最多查询次数", min_value=1, max_value=20, value=max_queries)
            stage1_prior_timeout_sec = st.number_input("Stage 1 初始判断时间上限（秒）", min_value=0, max_value=max(0, stage1_timeout_limit), value=stage1_prior_timeout_sec)
            stage1_update_timeout_sec = st.number_input("Stage 1 更新判断时间上限（秒）", min_value=0, max_value=max(0, stage1_timeout_limit), value=stage1_update_timeout_sec)
            stage2_query_timeout_sec = st.number_input("Stage 2 查询页时间上限（秒）", min_value=0, max_value=3600, value=stage2_query_timeout_sec)
            stage2_answer_timeout_sec = st.number_input("Stage 2 答案页时间上限（秒）", min_value=0, max_value=3600, value=stage2_answer_timeout_sec)
    card("点击开始后，你会先完成两个练习题。练习题不计入正式数据。")

    if get_secret("gsheet_url") is None:
        st.warning("当前为本地测试模式：未配置 Google Sheet secrets。")

    if st.button("开始"):
        group = infer_group(pid, group_choice)
        if not pid.strip() or not group or not consent or (require_desktop and not desktop_confirmed):
            st.warning("请填写受试者编号，并勾选电脑作答和知情同意。")
            st.stop()
        if int(min_queries) > int(max_queries):
            st.warning("最少查询次数不能大于最多查询次数。")
            st.stop()

        materials = load_materials()
        seed = stable_int(pid.strip(), platform_pid, group, condition, get_secret("randomization_salt", "human-balanced-stage1"))
        rng = random.Random(seed)
        order = list(range(len(materials)))
        rng.shuffle(order)
        selected_stage1_ids = set(stage1_item_ids())
        labelled_stage1 = [
            i for i, item in enumerate(materials)
            if str(item.get("suggested_stage", "")).strip().lower() == "stage1"
            and (not selected_stage1_ids or str(item.get("item_id", "")).strip() in selected_stage1_ids)
        ]
        labelled_stage2 = [
            i for i, item in enumerate(materials)
            if str(item.get("suggested_stage", "")).strip().lower() == "stage2"
        ]
        if labelled_stage1 and labelled_stage2:
            rng.shuffle(labelled_stage1)
            rng.shuffle(labelled_stage2)
            if int(stage1_n) > len(labelled_stage1) or int(stage2_n) > len(labelled_stage2):
                st.warning("当前 Stage 1 或 Stage 2 标记材料数量不足，请减少题数或检查材料文件。")
                st.stop()
            order_stage1 = labelled_stage1[: int(stage1_n)]
            order_stage2 = labelled_stage2[: int(stage2_n)]
        else:
            if int(stage1_n) + int(stage2_n) > len(order):
                st.warning("当前材料数量不足以保证第一部分和第二部分不复用题目，请减少题数或先扩充材料库。")
                st.stop()
            order_stage1 = order[: int(stage1_n)]
            order_stage2 = order[int(stage1_n) : int(stage1_n) + int(stage2_n)]
        st.session_state.pid = pid.strip()
        st.session_state.platform_participant_id = platform_pid.strip()
        st.session_state.platform_source = platform_source.strip()
        st.session_state.condition = condition.strip()
        st.session_state.return_url = return_url.strip()
        st.session_state.desktop_confirmed = bool(desktop_confirmed)
        st.session_state.group = group
        st.session_state.age = int(age)
        st.session_state.native_chinese = native_chinese
        st.session_state.stage1_n = len(order_stage1)
        st.session_state.stage2_n = len(order_stage2)
        st.session_state.min_queries = int(min_queries)
        st.session_state.max_queries = int(max_queries)
        st.session_state.stage1_prior_timeout_sec = int(stage1_prior_timeout_sec)
        st.session_state.stage1_update_timeout_sec = int(stage1_update_timeout_sec)
        st.session_state.stage2_query_timeout_sec = int(stage2_query_timeout_sec)
        st.session_state.stage2_answer_timeout_sec = int(stage2_answer_timeout_sec)
        st.session_state.order_stage1 = order_stage1
        st.session_state.order_stage2 = order_stage2
        st.session_state.page = "training"
        reset_screen_timer()
        st.rerun()


elif st.session_state.page == "training":
    st.title("正式开始前，请记住三件事")
    video_url = str(get_secret("instruction_video_url", "") or "").strip()
    if video_url:
        with st.expander("如果需要，可以重新观看讲解视频"):
            st.video(video_url)
    card("<b>1. 不要搜索答案，也不要和别人讨论。</b><br>请只根据你自己的想法作答。")
    card("<b>2. 第一部分：先看一个词，再看一个新提示。</b><br>你要判断“第一个词”和答案有多相关。看到新提示和分数后，再判断一次。")
    card("<b>3. 第二部分：自己输入想查的词。</b><br>系统会告诉你这个词和答案有多接近。分数越高，越接近答案。")
    warning_note("所有滑块都是 0-100 分。请拖动滑块，不要一直使用默认的 50。")
    if st.button("进入练习 1"):
        st.session_state.page = "practice_stage1"
        st.session_state.practice1_phase = "prior"
        reset_screen_timer()
        st.rerun()


elif st.session_state.page == "practice_stage1":
    st.title("练习 1：看完新提示后，再判断一次")
    st.caption("这是练习题，不记录为正式数据。")
    st.subheader("练习题：雨中的门口")
    st.write("一个人站在门口，外面正在下雨。他看了一眼手里的东西，突然决定不出门了。")
    st.markdown('<span class="big-word">第一个词：雨伞</span>', unsafe_allow_html=True)
    if st.session_state.practice1_phase == "prior":
        note("这一步只做一件事：先判断“雨伞”和答案有多相关。")
        rating_slider("现在看，你觉得“雨伞”和答案有多相关？", "practice_prior")
        st.caption("请拖动滑块。0 表示完全无关，100 表示非常相关。")
        if st.button("下一步：查看新提示"):
            st.session_state.practice1_phase = "update"
            reset_screen_timer()
            st.rerun()
    else:
        warning_note("重要：82 分是“钥匙”和答案的接近程度，不是“雨伞”的分数。请用这个新信息，再判断一次“雨伞”。")
        st.markdown('<span class="big-word">新提示：钥匙</span>', unsafe_allow_html=True)
        st.markdown('<span class="big-word">新提示和答案的接近程度：82 / 100</span>', unsafe_allow_html=True)
        rating_slider("看过“钥匙 82分”后，你觉得“雨伞”和答案有多相关？", "practice_updated")
        rating_slider("你对这次判断有多确定？", "practice_conf")
        if st.button("进入练习 2"):
            st.session_state.page = "practice_stage2"
            st.session_state.practice2_history = []
            reset_screen_timer()
            st.rerun()


elif st.session_state.page == "practice_stage2":
    st.title("练习 2：自己输入想查的词")
    st.caption("这是练习题，不记录为正式数据。")
    st.subheader("练习题：打不开的门")
    st.write("一个人回到家门口，却没有立刻进门。他在口袋里找了很久，然后笑了。")
    note("这一步只做一件事：输入一个你想查的词，看看它和答案有多接近。")
    practice_scores = {"钥匙": 90, "门": 72, "口袋": 61, "手机": 21, "蛋糕": 5}
    history = st.session_state.practice2_history
    if history:
        st.write("练习查询结果：")
        st.table(pd.DataFrame(history))
    q = st.text_input("想查的词", key="practice_query", help="例如：钥匙、门、手机。一次只输入一个词。")
    st.caption("请换用新的词；同一个词不要重复查。")
    if st.button("查询这个词"):
        query_clean = clean_query_word(q)
        score = practice_scores.get(query_clean, 30 if query_clean else 0)
        if not query_clean:
            st.warning("请输入一个词。")
        elif query_already_used(history, query_raw=query_clean):
            st.warning("这个词已经查询过了，请换一个新词；本次不计入查询次数。")
        else:
            history.append({"第几次": len(history) + 1, "你查的词": query_clean, "这个词和答案的接近程度": f"{score} / 100"})
            st.session_state.practice2_history = history
            st.rerun()
    st.caption("正式实验中，每道题要先查询几次；查够后会出现进入答案页的按钮。")
    if len(history) >= 1 and st.button("我已理解，开始正式实验"):
        reset_formal_quality_state()
        st.session_state.page = "stage1"
        reset_screen_timer()
        st.rerun()


elif st.session_state.page == "stage1":
    materials = load_materials()
    lookup = load_stage1_lookup(st.session_state.group)
    idx = st.session_state.stage1_idx
    if idx >= len(st.session_state.order_stage1):
        st.session_state.page = "stage2_intro"
        st.rerun()

    item = materials[st.session_state.order_stage1[idx]]
    item_id = item["item_id"]
    fb = lookup[item_id]
    show_progress("第一部分", idx, len(st.session_state.order_stage1))
    st.subheader(item.get("title") or f"题目 {idx + 1}")
    st.write(item["riddle_text"])
    st.markdown(f'<span class="big-word">第一个词：{fb["anchor_word"]}</span>', unsafe_allow_html=True)

    if st.session_state.stage1_phase == "prior":
        prior_limit = int(st.session_state.get("stage1_prior_timeout_sec", 0))
        show_time_rule(prior_limit)
        note("这页只做一件事：先判断“第一个词”和答案有多相关。")
        prior = rating_slider(f"现在看，你觉得“{fb['anchor_word']}”和答案有多相关？", f"prior_{item_id}")
        if st.button("下一步：查看新提示", key=f"show_{item_id}"):
            if is_timed_out(prior_limit):
                event = {
                    "participant_id": st.session_state.pid,
                    "group": st.session_state.group,
                    "stage": "stage1_timeout",
                    "trial_set": "stage1",
                    "item_id": item_id,
                    "title": item.get("title", ""),
                    "anchor_word": fb["anchor_word"],
                    "cue_word": fb["cue_word"],
                    "prior": prior / 100,
                    "screen_timeout_sec": prior_limit,
                    "timeout_flag": True,
                    "invalid_reason": "stage1_prior_timeout",
                }
                add_researcher_fields(event, item, fb)
                log_event(event)
                st.session_state.stage1_idx += 1
                st.session_state.stage1_phase = "prior"
                st.warning("本轮已超时，系统已记录为无效并进入下一题。")
                st.rerun()
            st.session_state.temp_prior = prior
            st.session_state.stage1_phase = "update"
            reset_screen_timer()
            st.rerun()
    else:
        update_limit = int(st.session_state.get("stage1_update_timeout_sec", 0))
        show_time_rule(update_limit)
        warning_note(f"重要：{fb['target_cue_score']} 分是“{fb['cue_word']}”和答案的接近程度，不是“{fb['anchor_word']}”的分数。请用这个新信息，再判断一次“{fb['anchor_word']}”。")
        st.markdown(f'<span class="big-word">新提示：{fb["cue_word"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="big-word">新提示和答案的接近程度：{fb["target_cue_score"]} / 100</span>', unsafe_allow_html=True)
        updated = rating_slider(f"看过“{fb['cue_word']} {fb['target_cue_score']}分”后，你觉得“{fb['anchor_word']}”和答案有多相关？", f"updated_{item_id}")
        confidence = rating_slider("你对这次判断有多确定？", f"conf_{item_id}")
        if st.button("提交，进入下一题", key=f"submit_stage1_{item_id}"):
            timeout = is_timed_out(update_limit)
            event = {
                    "participant_id": st.session_state.pid,
                    "group": st.session_state.group,
                    "stage": "stage1_timeout" if timeout else "stage1_passive_update",
                    "trial_set": "stage1",
                    "item_id": item_id,
                    "title": item.get("title", ""),
                    "anchor_word": fb["anchor_word"],
                    "cue_word": fb["cue_word"],
                    "target_cue_score": fb["target_cue_score"],
                    "prior": st.session_state.temp_prior / 100,
                    "updated": updated / 100,
                    "confidence": confidence / 100,
                    "screen_timeout_sec": update_limit,
                    "timeout_flag": timeout,
                    "invalid_reason": "stage1_update_timeout" if timeout else "",
                }
            add_researcher_fields(event, item, fb)
            log_event(event)
            st.session_state.stage1_idx += 1
            st.session_state.stage1_phase = "prior"
            st.rerun()


elif st.session_state.page == "stage2_intro":
    st.title("第二部分：自己查词")
    card("你会看到新的谜题。先不要写答案。")
    card("你要输入自己想查的词。系统会告诉你：这个词和答案有多接近。")
    card("分数越高，表示这个词越接近答案。同一道题不要重复查同一个词。")
    st.caption(f"每题至少查询 {st.session_state.get('min_queries', 0)} 次，最多查询 {st.session_state.max_queries} 次。")
    if st.button("进入第二部分"):
        st.session_state.page = "stage2"
        st.session_state.stage2_phase = "query"
        st.session_state.explore_history = []
        reset_screen_timer()
        st.rerun()


elif st.session_state.page == "stage2":
    materials = load_materials()
    full_lookup = load_full_lookup(st.session_state.group)
    idx = st.session_state.stage2_idx
    if idx >= len(st.session_state.order_stage2):
        st.session_state.page = "done"
        st.rerun()

    item = materials[st.session_state.order_stage2[idx]]
    item_id = item["item_id"]
    scores = full_lookup[item_id]["scores"]
    grounding_config = load_query_grounding_config()
    semantic_index = load_semantic_index()
    show_progress("第二部分", idx, len(st.session_state.order_stage2))
    st.subheader(item.get("title") or f"探索题 {idx + 1}")
    st.write(item["riddle_text"])

    history = st.session_state.explore_history
    min_queries = int(st.session_state.get("min_queries", 0))
    if st.session_state.get("stage2_phase", "query") == "query":
        query_limit = int(st.session_state.get("stage2_query_timeout_sec", 0))
        show_time_rule(query_limit)
        if history:
            visible_history = [
                {"第几次": h["query_index"], "你查的词": h["query_raw"], "这个词和答案的接近程度": f"{h['score']} / 100"}
                for h in history
            ]
            st.write("已查询结果：")
            st.table(pd.DataFrame(visible_history))

        note(f"这页只做一件事：输入一个想查的词。已查询 {len(history)} 次；至少 {min_queries} 次后可以进入答案页。")
        if len(history) < st.session_state.max_queries:
            query = st.text_input("想查的词", key=f"query_{item_id}_{len(history)}", help="一次只输入一个常见词，不要输入一句话。")
            st.caption("请换用新的词；已经查过的词不会重复计数。")
            if st.button("查询这个词", key=f"do_query_{item_id}_{len(history)}"):
                if is_timed_out(query_limit):
                    event = {
                        "participant_id": st.session_state.pid,
                        "group": st.session_state.group,
                        "stage": "stage2_timeout",
                        "trial_set": "stage2",
                        "item_id": item_id,
                        "title": item.get("title", ""),
                        "n_queries": len(history),
                        "query_history": history,
                        "screen_timeout_sec": query_limit,
                        "timeout_flag": True,
                        "invalid_reason": "stage2_query_timeout",
                    }
                    add_researcher_fields(event, item)
                    log_event(event)
                    st.session_state.stage2_idx += 1
                    st.session_state.stage2_phase = "query"
                    st.session_state.explore_history = []
                    st.warning("本题查询阶段已超时，系统已记录为无效并进入下一题。")
                    st.rerun()
                query_clean = clean_query_word(query)
                if not query_clean:
                    increment_counter("empty_query_attempts")
                    st.warning("请输入一个词。")
                    st.stop()
                if query_already_used(history, query_raw=query_clean):
                    increment_counter("duplicate_query_attempts")
                    st.warning("这个词已经查询过了，请换一个新词；本次不计入查询次数。")
                    st.stop()
                resolution = resolve_query_feedback(
                    query_clean,
                    target_word=item.get("network_target_word", ""),
                    scores=scores,
                    grounding_config=grounding_config,
                    semantic_index=semantic_index,
                )
                if not resolution.has_feedback:
                    increment_counter("no_feedback_query_attempts")
                    st.warning("词表中没有找到该词或近似词，请换一个更常见的词。")
                    st.stop()
                if query_already_used(history, query_raw=query_clean, matched_word=resolution.matched_word):
                    increment_counter("duplicate_query_attempts")
                    st.warning("这个词对应的查询结果已经出现过了，请换一个新词；本次不计入查询次数。")
                    st.stop()
                event = {
                    "participant_id": st.session_state.pid,
                    "group": st.session_state.group,
                    "stage": "stage2_active_query",
                    "trial_set": "stage2",
                    "item_id": item_id,
                    "title": item.get("title", ""),
                    "query_index": len(history) + 1,
                    "query_raw": query_clean,
                }
                event.update(resolution.as_event_fields())
                event["query_word"] = resolution.matched_word
                add_researcher_fields(event, item)
                log_event(event)
                history.append(
                    {
                        "query_index": len(history) + 1,
                        "query_raw": query_clean,
                        "query_word": resolution.matched_word,
                        "matched_word": resolution.matched_word,
                        "match_type": resolution.match_type,
                        "match_confidence": round(float(resolution.match_confidence), 4),
                        "resolver_source": resolution.resolver_source,
                        "feedback_prob": resolution.feedback_prob,
                        "score": resolution.feedback_score,
                    }
                )
                st.session_state.explore_history = history
                st.rerun()

        can_answer = len(history) >= min_queries
        if can_answer:
            if st.button("我知道答案了，去填写答案", key=f"know_{item_id}"):
                st.session_state.stage2_phase = "answer"
                reset_screen_timer()
                st.rerun()
        else:
            st.caption(f"还需要至少查询 {min_queries - len(history)} 次。")
    else:
        answer_limit = int(st.session_state.get("stage2_answer_timeout_sec", 0))
        show_time_rule(answer_limit)
        if history:
            visible_history = [
                {"第几次": h["query_index"], "你查的词": h["query_raw"], "这个词和答案的接近程度": f"{h['score']} / 100"}
                for h in history
            ]
            st.write("你的查询结果：")
            st.table(pd.DataFrame(visible_history))
        note("请根据题目和你查到的分数，写出你认为的答案或解释。")
        guess = st.text_area("你的答案或解释", key=f"guess_{item_id}")
        aha = rating_slider("你是否有突然明白的感觉", f"aha_{item_id}")
        aha_suddenness = rating_slider("这个想法出现得有多突然", f"aha_sudden_{item_id}")
        aha_surprise = rating_slider("这个答案让你有多惊讶", f"aha_surprise_{item_id}")
        confidence = rating_slider("你对这个答案有多确定？", f"final_conf_{item_id}")
        known_story = st.radio("你以前是否看过这个谜题或知道答案？", ["否", "不确定", "是"], horizontal=True, key=f"known_{item_id}")
        if st.button("提交并进入下一题", key=f"finish_{item_id}"):
            timeout = is_timed_out(answer_limit)
            event = {
                    "participant_id": st.session_state.pid,
                    "group": st.session_state.group,
                    "stage": "stage2_timeout" if timeout else "stage2_final_guess",
                    "trial_set": "stage2",
                    "item_id": item_id,
                    "title": item.get("title", ""),
                    "guess": guess,
                    "aha": aha / 100,
                    "aha_suddenness": aha_suddenness / 100,
                    "aha_surprise": aha_surprise / 100,
                    "confidence": confidence / 100,
                    "known_story": known_story,
                    "n_queries": len(history),
                    "query_history": history,
                    "screen_timeout_sec": answer_limit,
                    "timeout_flag": timeout,
                    "invalid_reason": "stage2_answer_timeout" if timeout else "",
                }
            add_researcher_fields(event, item)
            log_event(event)
            st.session_state.stage2_idx += 1
            st.session_state.stage2_phase = "query"
            st.session_state.explore_history = []
            st.rerun()


elif st.session_state.page == "done":
    if "quality_evaluated" not in st.session_state:
        quality = evaluate_completion_quality()
        st.session_state.quality_evaluated = True
        st.session_state.quality_status = quality["quality_status"]
        st.session_state.quality_pass = bool(quality["quality_pass"])
        st.session_state.payment_eligible = bool(quality["payment_eligible"])
        st.session_state.quality_flags = ",".join(quality["quality_flags"])
        st.session_state.quality_summary = json.dumps(quality["quality_summary"], ensure_ascii=False, default=str)
        for key in [
            "total_duration_sec",
            "stage1_valid_count",
            "stage1_timeout_count",
            "stage2_query_count",
            "stage2_final_count",
            "stage2_timeout_count",
            "empty_query_attempts",
            "duplicate_query_attempts",
            "no_feedback_query_attempts",
            "invalid_query_attempts",
        ]:
            st.session_state[key] = quality[key]

        if quality["quality_pass"]:
            code = completion_code(st.session_state.get("pid", ""), st.session_state.get("session_id", ""), st.session_state.get("group", ""))
            st.session_state.completion_code = code
            st.session_state.completion_code_hash = completion_hash(code)
            st.session_state.review_code = ""
            stage_name = "completion_valid"
        else:
            code = review_code(st.session_state.get("pid", ""), st.session_state.get("session_id", ""), st.session_state.get("group", ""))
            st.session_state.completion_code = ""
            st.session_state.completion_code_hash = ""
            st.session_state.review_code = code
            stage_name = "completion_review"

        log_event(
            {
                "participant_id": st.session_state.get("pid", ""),
                "group": st.session_state.get("group", ""),
                "stage": stage_name,
                "trial_set": "completion",
                "completion_code": st.session_state.get("completion_code", ""),
                "completion_code_hash": st.session_state.get("completion_code_hash", ""),
                "review_code": st.session_state.get("review_code", ""),
                "quality_status": st.session_state.get("quality_status", ""),
                "quality_pass": st.session_state.get("quality_pass", ""),
                "quality_flags": st.session_state.get("quality_flags", ""),
                "quality_summary": st.session_state.get("quality_summary", ""),
                "payment_eligible": st.session_state.get("payment_eligible", ""),
                "n_stage1_trials": st.session_state.get("stage1_n", ""),
                "n_stage2_trials": st.session_state.get("stage2_n", ""),
                "total_duration_sec": st.session_state.get("total_duration_sec", ""),
                "stage1_valid_count": st.session_state.get("stage1_valid_count", ""),
                "stage1_timeout_count": st.session_state.get("stage1_timeout_count", ""),
                "stage2_query_count": st.session_state.get("stage2_query_count", ""),
                "stage2_final_count": st.session_state.get("stage2_final_count", ""),
                "stage2_timeout_count": st.session_state.get("stage2_timeout_count", ""),
                "empty_query_attempts": st.session_state.get("empty_query_attempts", ""),
                "duplicate_query_attempts": st.session_state.get("duplicate_query_attempts", ""),
                "no_feedback_query_attempts": st.session_state.get("no_feedback_query_attempts", ""),
                "invalid_query_attempts": st.session_state.get("invalid_query_attempts", ""),
            }
        )
    st.success("实验完成，感谢你完成本次任务。")
    if st.session_state.get("quality_pass"):
        st.subheader("完成码")
        st.code(st.session_state.completion_code)
        st.caption("请返回问卷平台填写该完成码，以便核验完成状态和发放报酬。")
        return_code = st.session_state.completion_code
    else:
        st.warning("你的实验记录需要人工审核。当前不会生成正式完成码。")
        st.subheader("人工审核编号")
        st.code(st.session_state.review_code)
        st.caption("请返回问卷平台填写该人工审核编号。研究者会根据完整记录决定是否采纳和发放报酬。")
        with st.expander("为什么需要人工审核？"):
            flags = st.session_state.get("quality_flags", "")
            st.write(flags or "系统质量检查发现需要人工确认。")
        return_code = st.session_state.review_code
    final_return_url = build_return_url(st.session_state.get("return_url", ""), st.session_state.get("pid", ""), return_code)
    if final_return_url:
        if hasattr(st, "link_button"):
            st.link_button("返回问卷平台", final_return_url)
        else:
            st.markdown(f"[返回问卷平台]({final_return_url})")
    if st.session_state.get("sheet_status") == "failed":
        st.warning("云端保存可能失败；请联系实验人员。")
    df = pd.DataFrame(st.session_state.responses)
    st.download_button("下载本次数据 CSV", df.to_csv(index=False), "human_experiment_events.csv", "text/csv")
