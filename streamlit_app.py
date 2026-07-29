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


APP_VERSION = "human-two-stage-v2.3-platform-ready"


st.set_page_config(page_title="文字谜题联想实验", layout="centered")


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
    "completion_code",
    "completion_code_hash",
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
        st.session_state.responses = []
        st.session_state.explore_history = []
        st.session_state.screen_started_at = time.time()


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
    row.setdefault("completion_code", st.session_state.get("completion_code", ""))
    row.setdefault("completion_code_hash", st.session_state.get("completion_code_hash", ""))
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


init_state()


if st.session_state.page == "intro":
    st.title("文字谜题联想实验")
    st.write("本实验包含两个部分：先根据提示更新判断，再主动查询线索。")
    st.caption("你的回答将匿名用于科研分析；你可以随时停止参与。")

    pid_from_url = query_param("pid", "participant_id", "participantId", "uid", "id")
    platform_pid = query_param("platform_pid", "external_id", "respondent_id", "wjx_id", "credamo_id", default=pid_from_url)
    platform_source = query_param("source", "platform", "recruitment_source", default="direct")
    condition = query_param("condition", "cond", default=str(get_secret("condition", "balanced_stage1")))
    group_from_url = query_param("group", default="").upper()
    return_url = query_param("return_url", "redirect", default="")

    pid = st.text_input("参与者 ID", value=pid_from_url)
    default_group_idx = list(GROUP_PROFILES.keys()).index(group_from_url) if group_from_url in GROUP_PROFILES else 0
    group_choice = st.selectbox("分组（若 ID 中包含 FH/FN/MH/MN，将自动使用 ID 中的分组）", list(GROUP_PROFILES.keys()), index=default_group_idx)
    age = st.number_input("年龄", min_value=10, max_value=99, value=20)
    native_chinese = st.selectbox("中文熟练程度", ["母语/近似母语", "熟练", "一般"])
    desktop_confirmed = st.checkbox("我正在使用电脑或笔记本电脑完成实验。")
    consent = st.checkbox("我已了解实验说明，并自愿参加。")
    st.caption(f"招募来源：{platform_source}；实验条件：{condition}")

    stage1_n = int_secret("stage1_n", 12)
    stage2_n = int_secret("stage2_n", 8)
    min_queries = int_secret("min_queries", 3)
    max_queries = int_secret("max_queries", 8)
    if bool_secret("show_admin_controls", False):
        with st.expander("实验员设置"):
            stage1_n = st.number_input("Stage 1 题数", min_value=1, max_value=80, value=stage1_n)
            stage2_n = st.number_input("Stage 2 题数", min_value=1, max_value=80, value=stage2_n)
            min_queries = st.number_input("Stage 2 每题最少查询次数", min_value=0, max_value=20, value=min_queries)
            max_queries = st.number_input("Stage 2 每题最多查询次数", min_value=1, max_value=20, value=max_queries)
    st.caption(f"预计第一部分 {stage1_n} 题；第二部分 {stage2_n} 题，每题查询 {min_queries}-{max_queries} 次。")

    if get_secret("gsheet_url") is None:
        st.warning("当前为本地测试模式：未配置 Google Sheet secrets。")

    if st.button("开始"):
        group = infer_group(pid, group_choice)
        if not pid.strip() or not group or not consent or not desktop_confirmed:
            st.warning("请输入参与者 ID、确认分组，并确认知情同意和电脑端作答。")
            st.stop()
        if int(min_queries) > int(max_queries):
            st.warning("最少查询次数不能大于最多查询次数。")
            st.stop()

        materials = load_materials()
        seed = stable_int(pid.strip(), platform_pid, group, condition, get_secret("randomization_salt", "human-balanced-stage1"))
        rng = random.Random(seed)
        order = list(range(len(materials)))
        rng.shuffle(order)
        labelled_stage1 = [
            i for i, item in enumerate(materials)
            if str(item.get("suggested_stage", "")).strip().lower() == "stage1"
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
        st.session_state.order_stage1 = order_stage1
        st.session_state.order_stage2 = order_stage2
        st.session_state.page = "training"
        st.rerun()


elif st.session_state.page == "training":
    st.title("练习说明")
    st.write("每道题会给出一个简短谜面。你不会看到真正答案。")
    st.write("第一部分中，你会看到一个候选词，并先判断它是否可能解释谜面；随后看到一个提示词及其与真实答案的关联分数，再更新判断。")
    st.write("第二部分中，你可以主动输入想查询的词，系统会返回该词与真实答案的关联分数。")
    if st.button("进入第一部分"):
        st.session_state.page = "stage1"
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
    st.markdown(f"候选词：**{fb['anchor_word']}**")

    if st.session_state.stage1_phase == "prior":
        prior = st.slider("你认为这个候选词能解释谜面的可能性", 0, 100, 50, key=f"prior_{item_id}")
        if st.button("查看提示", key=f"show_{item_id}"):
            st.session_state.temp_prior = prior
            st.session_state.stage1_phase = "update"
            st.rerun()
    else:
        st.info("分数表示提示词与真实答案的语义关联强度。")
        st.markdown(f"提示词：**{fb['cue_word']}**")
        st.markdown(f"关联分数：**{fb['target_cue_score']} / 100**")
        updated = st.slider("看到提示后，你现在的判断", 0, 100, 50, key=f"updated_{item_id}")
        confidence = st.slider("你对当前判断的信心", 0, 100, 50, key=f"conf_{item_id}")
        if st.button("提交本题", key=f"submit_stage1_{item_id}"):
            event = {
                    "participant_id": st.session_state.pid,
                    "group": st.session_state.group,
                    "stage": "stage1_passive_update",
                    "trial_set": "stage1",
                    "item_id": item_id,
                    "title": item.get("title", ""),
                    "anchor_word": fb["anchor_word"],
                    "cue_word": fb["cue_word"],
                    "target_cue_score": fb["target_cue_score"],
                    "prior": st.session_state.temp_prior / 100,
                    "updated": updated / 100,
                    "confidence": confidence / 100,
                }
            add_researcher_fields(event, item, fb)
            log_event(event)
            st.session_state.stage1_idx += 1
            st.session_state.stage1_phase = "prior"
            st.rerun()


elif st.session_state.page == "stage2_intro":
    st.title("第二部分：主动探索")
    st.write("你可以输入词语来查询它与真实答案的关联强度。查询次数有限；准备好后进入第二部分。")
    st.caption(f"每题至少查询 {st.session_state.get('min_queries', 0)} 次，最多查询 {st.session_state.max_queries} 次。")
    if st.button("进入第二部分"):
        st.session_state.page = "stage2"
        st.session_state.explore_history = []
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
    if history:
        st.write("已查询：")
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

    if len(history) < st.session_state.max_queries:
        query = st.text_input("输入一个你想查询的词", key=f"query_{item_id}_{len(history)}")
        if st.button("查询", key=f"do_query_{item_id}_{len(history)}"):
            resolution = resolve_query_feedback(
                query,
                target_word=item.get("network_target_word", ""),
                scores=scores,
                grounding_config=grounding_config,
                semantic_index=semantic_index,
            )
            if not resolution.has_feedback:
                st.warning("词表中没有找到该词或近似词，请换一个更常见的词。")
                st.stop()
            event = {
                "participant_id": st.session_state.pid,
                "group": st.session_state.group,
                "stage": "stage2_active_query",
                "trial_set": "stage2",
                "item_id": item_id,
                "title": item.get("title", ""),
                "query_index": len(history) + 1,
            }
            event.update(resolution.as_event_fields())
            event["query_word"] = resolution.matched_word
            add_researcher_fields(event, item)
            log_event(event)
            history.append(
                {
                    "query_word": resolution.matched_word,
                    "matched_word": resolution.matched_word,
                    "match_type": resolution.match_type,
                    "match_confidence": round(float(resolution.match_confidence), 4),
                    "grounded_words": resolution.as_event_fields().get("grounded_words", ""),
                    "grounding_weights": resolution.as_event_fields().get("grounding_weights", ""),
                    "grounding_method": resolution.as_event_fields().get("grounding_method", ""),
                    "semantic_candidates": resolution.as_event_fields().get("semantic_candidates", ""),
                    "semantic_similarities": resolution.as_event_fields().get("semantic_similarities", ""),
                    "resolver_source": resolution.resolver_source,
                    "feedback_prob": resolution.feedback_prob,
                    "score": resolution.feedback_score,
                }
            )
            st.session_state.explore_history = history
            st.rerun()

    guess = st.text_input("最终答案/解释", key=f"guess_{item_id}")
    aha = st.slider("你是否有突然明白的感觉", 0, 100, 50, key=f"aha_{item_id}")
    aha_suddenness = st.slider("这个想法出现得有多突然", 0, 100, 50, key=f"aha_sudden_{item_id}")
    aha_surprise = st.slider("这个答案让你有多惊讶", 0, 100, 50, key=f"aha_surprise_{item_id}")
    confidence = st.slider("你对最终答案的信心", 0, 100, 50, key=f"final_conf_{item_id}")
    known_story = st.radio("你之前是否知道这个谜题或答案？", ["否", "不确定", "是"], horizontal=True, key=f"known_{item_id}")
    min_queries = int(st.session_state.get("min_queries", 0))
    if len(history) < min_queries:
        st.caption(f"请至少查询 {min_queries} 次后再提交。")
    if st.button("提交并进入下一题", key=f"finish_{item_id}", disabled=len(history) < min_queries):
        event = {
                "participant_id": st.session_state.pid,
                "group": st.session_state.group,
                "stage": "stage2_final_guess",
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
            }
        add_researcher_fields(event, item)
        log_event(event)
        st.session_state.stage2_idx += 1
        st.session_state.explore_history = []
        st.rerun()


elif st.session_state.page == "done":
    if "completion_code" not in st.session_state:
        code = completion_code(st.session_state.get("pid", ""), st.session_state.get("session_id", ""), st.session_state.get("group", ""))
        st.session_state.completion_code = code
        st.session_state.completion_code_hash = completion_hash(code)
        log_event(
            {
                "participant_id": st.session_state.get("pid", ""),
                "group": st.session_state.get("group", ""),
                "stage": "completion",
                "trial_set": "completion",
                "completion_code": code,
                "completion_code_hash": st.session_state.completion_code_hash,
                "n_stage1_trials": st.session_state.get("stage1_n", ""),
                "n_stage2_trials": st.session_state.get("stage2_n", ""),
            }
        )
    st.success("实验完成，感谢参与。")
    st.subheader("完成码")
    st.code(st.session_state.completion_code)
    st.caption("请返回问卷平台填写该完成码，以便核验完成状态和发放报酬。")
    final_return_url = build_return_url(st.session_state.get("return_url", ""), st.session_state.get("pid", ""), st.session_state.completion_code)
    if final_return_url:
        if hasattr(st, "link_button"):
            st.link_button("返回问卷平台", final_return_url)
        else:
            st.markdown(f"[返回问卷平台]({final_return_url})")
    if st.session_state.get("sheet_status") == "failed":
        st.warning("云端保存可能失败；请联系实验人员。")
    df = pd.DataFrame(st.session_state.responses)
    st.download_button("下载本次数据 CSV", df.to_csv(index=False), "human_experiment_events.csv", "text/csv")
