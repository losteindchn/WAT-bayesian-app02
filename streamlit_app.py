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


APP_VERSION = "human-two-stage-v2.4-training-timed-stage2-clean"


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
    "screen_timeout_sec",
    "timeout_flag",
    "invalid_reason",
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
        st.session_state.stage2_phase = "query"
        st.session_state.practice1_phase = "prior"
        st.session_state.practice2_history = []
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
    row.setdefault("screen_timeout_sec", "")
    row.setdefault("timeout_flag", False)
    row.setdefault("invalid_reason", "")
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


def reset_screen_timer() -> None:
    st.session_state.screen_started_at = time.time()


def elapsed_screen_sec() -> float:
    return time.time() - float(st.session_state.get("screen_started_at", time.time()))


def is_timed_out(limit_sec: int) -> bool:
    return bool(limit_sec and limit_sec > 0 and elapsed_screen_sec() > float(limit_sec))


def show_time_rule(limit_sec: int) -> None:
    if limit_sec and limit_sec > 0:
        minutes = max(1, round(limit_sec / 60))
        st.caption(f"本页作答时间上限约 {minutes} 分钟；超时后再提交将被记录为本轮无效。")


def rating_slider(label: str, key: str) -> int:
    return st.slider(f"{label}（0=完全没有，100=非常强；请拖动滑块选择）", 0, 100, 50, key=key)


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
    require_desktop = bool_secret("require_desktop", True)
    desktop_confirmed = st.checkbox("我正在使用电脑或笔记本电脑完成实验。")
    if require_desktop:
        st.caption("正式实验只接受电脑或笔记本作答；手机屏幕会改变阅读、输入和查询体验，可能导致数据无效。")
    else:
        st.caption("当前允许非电脑设备进入，但设备信息会用于后续数据质量审核。")
    consent = st.checkbox("我已了解实验说明，并自愿参加。")
    st.caption(f"招募来源：{platform_source}；实验条件：{condition}")

    stage1_n = int_secret("stage1_n", 12)
    stage2_n = int_secret("stage2_n", 8)
    min_queries = int_secret("min_queries", 3)
    max_queries = int_secret("max_queries", 8)
    stage1_prior_timeout_sec = int_secret("stage1_prior_timeout_sec", 180)
    stage1_update_timeout_sec = int_secret("stage1_update_timeout_sec", 180)
    stage2_query_timeout_sec = int_secret("stage2_query_timeout_sec", 480)
    stage2_answer_timeout_sec = int_secret("stage2_answer_timeout_sec", 240)
    if bool_secret("show_admin_controls", False):
        with st.expander("实验员设置"):
            stage1_n = st.number_input("Stage 1 题数", min_value=1, max_value=80, value=stage1_n)
            stage2_n = st.number_input("Stage 2 题数", min_value=1, max_value=80, value=stage2_n)
            min_queries = st.number_input("Stage 2 每题最少查询次数", min_value=0, max_value=20, value=min_queries)
            max_queries = st.number_input("Stage 2 每题最多查询次数", min_value=1, max_value=20, value=max_queries)
            stage1_prior_timeout_sec = st.number_input("Stage 1 初始判断时间上限（秒）", min_value=0, max_value=3600, value=stage1_prior_timeout_sec)
            stage1_update_timeout_sec = st.number_input("Stage 1 更新判断时间上限（秒）", min_value=0, max_value=3600, value=stage1_update_timeout_sec)
            stage2_query_timeout_sec = st.number_input("Stage 2 查询页时间上限（秒）", min_value=0, max_value=3600, value=stage2_query_timeout_sec)
            stage2_answer_timeout_sec = st.number_input("Stage 2 答案页时间上限（秒）", min_value=0, max_value=3600, value=stage2_answer_timeout_sec)
    st.caption(f"预计第一部分 {stage1_n} 题；第二部分 {stage2_n} 题，每题查询 {min_queries}-{max_queries} 次。")

    if get_secret("gsheet_url") is None:
        st.warning("当前为本地测试模式：未配置 Google Sheet secrets。")

    if st.button("开始"):
        group = infer_group(pid, group_choice)
        if not pid.strip() or not group or not consent or (require_desktop and not desktop_confirmed):
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
    st.title("实验规则说明")
    video_url = str(get_secret("instruction_video_url", "") or "").strip()
    if video_url:
        st.video(video_url)
    st.markdown(
        """
你要完成的是一个文字谜题联想任务。请不要直接搜索答案，也不要和别人讨论。

**第一部分**：你会看到谜面和一个“初始线索词”。请用 0-100 的滑块判断它与谜底或关键机制有多相关。随后系统给出一个提示词和分数，你再更新判断。

**第二部分**：你会看到新的谜题。你不能立刻填写答案，而是先输入若干个想查询的词。系统只返回这个词与真实答案的关联分数。分数越高，说明这个词越接近真实答案。你觉得已经知道答案后，再进入答案页填写解释和体验问卷。

所有滑块都是 0-100 分，请拖动圆点选择；不要只使用默认 50。
        """
    )
    st.info("正式题目有时间上限。超时提交会被记录为本轮无效，所以请在理解题目后及时作答。")
    if st.button("进入练习 1：判断更新"):
        st.session_state.page = "practice_stage1"
        st.session_state.practice1_phase = "prior"
        reset_screen_timer()
        st.rerun()


elif st.session_state.page == "practice_stage1":
    st.title("练习 1：如何使用滑块更新判断")
    st.caption("这是练习题，不记录为正式数据。")
    st.subheader("练习题：雨中的门口")
    st.write("一个人站在门口，外面正在下雨。他看了一眼手里的东西，突然决定不出门了。")
    st.markdown("初始线索词：**雨伞**")
    if st.session_state.practice1_phase == "prior":
        rating_slider("你认为这个线索词与谜底或关键机制相关的可能性", "practice_prior")
        st.caption("请拖动滑块。0 表示完全无关，100 表示非常相关。")
        if st.button("查看练习提示"):
            st.session_state.practice1_phase = "update"
            reset_screen_timer()
            st.rerun()
    else:
        st.info("分数表示提示词与真实答案的语义关联强度。")
        st.markdown("提示词：**钥匙**")
        st.markdown("关联分数：**82 / 100**")
        rating_slider("看到提示后，你现在的判断", "practice_updated")
        rating_slider("你对当前判断的信心", "practice_conf")
        if st.button("进入练习 2：主动查询"):
            st.session_state.page = "practice_stage2"
            st.session_state.practice2_history = []
            reset_screen_timer()
            st.rerun()


elif st.session_state.page == "practice_stage2":
    st.title("练习 2：如何主动查询")
    st.caption("这是练习题，不记录为正式数据。")
    st.subheader("练习题：打不开的门")
    st.write("一个人回到家门口，却没有立刻进门。他在口袋里找了很久，然后笑了。")
    practice_scores = {"钥匙": 90, "门": 72, "口袋": 61, "手机": 21, "蛋糕": 5}
    history = st.session_state.practice2_history
    if history:
        st.write("练习查询结果：")
        st.table(pd.DataFrame(history))
    q = st.text_input("输入一个你想查询的词，例如：钥匙、门、手机", key="practice_query")
    if st.button("查询练习词"):
        score = practice_scores.get(q.strip(), 30 if q.strip() else 0)
        if not q.strip():
            st.warning("请输入一个词。")
        else:
            history.append({"第几次": len(history) + 1, "查询词": q.strip(), "关联分数": f"{score} / 100"})
            st.session_state.practice2_history = history
            st.rerun()
    st.caption("正式实验中，每道题至少查询指定次数；达到次数后会出现“我知道答案了”按钮。")
    if len(history) >= 1 and st.button("我已理解，进入第一部分正式实验"):
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
    st.markdown(f"初始线索词：**{fb['anchor_word']}**")

    if st.session_state.stage1_phase == "prior":
        prior_limit = int(st.session_state.get("stage1_prior_timeout_sec", 0))
        show_time_rule(prior_limit)
        prior = rating_slider("你认为这个线索词与谜底或关键机制相关的可能性", f"prior_{item_id}")
        if st.button("查看提示", key=f"show_{item_id}"):
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
        st.info("分数表示提示词与真实答案的语义关联强度。")
        st.markdown(f"提示词：**{fb['cue_word']}**")
        st.markdown(f"关联分数：**{fb['target_cue_score']} / 100**")
        updated = rating_slider("看到提示后，你现在的判断", f"updated_{item_id}")
        confidence = rating_slider("你对当前判断的信心", f"conf_{item_id}")
        if st.button("提交本题", key=f"submit_stage1_{item_id}"):
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
    st.title("第二部分：主动探索")
    st.write("你可以输入词语来查询它与真实答案的关联强度。系统只会显示 0-100 的关联分数。")
    st.write("请先通过查询词逐步探索，不要一开始就填写答案。达到最少查询次数后，如果你觉得知道答案了，可以进入答案页。")
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
                {"第几次": h["query_index"], "查询词": h["query_raw"], "关联分数": f"{h['score']} / 100"}
                for h in history
            ]
            st.write("已查询结果：")
            st.table(pd.DataFrame(visible_history))

        st.caption(f"已查询 {len(history)} / {st.session_state.max_queries} 次；至少 {min_queries} 次后可以进入答案页。")
        if len(history) < st.session_state.max_queries:
            query = st.text_input("输入一个你想查询的词", key=f"query_{item_id}_{len(history)}")
            if st.button("查询", key=f"do_query_{item_id}_{len(history)}"):
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
                    "query_raw": query,
                }
                event.update(resolution.as_event_fields())
                event["query_word"] = resolution.matched_word
                add_researcher_fields(event, item)
                log_event(event)
                history.append(
                    {
                        "query_index": len(history) + 1,
                        "query_raw": query.strip(),
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
            if st.button("我知道答案了，进入答案页", key=f"know_{item_id}"):
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
                {"第几次": h["query_index"], "查询词": h["query_raw"], "关联分数": f"{h['score']} / 100"}
                for h in history
            ]
            st.write("你的查询结果：")
            st.table(pd.DataFrame(visible_history))
        guess = st.text_area("请写下你的最终答案/解释", key=f"guess_{item_id}")
        aha = rating_slider("你是否有突然明白的感觉", f"aha_{item_id}")
        aha_suddenness = rating_slider("这个想法出现得有多突然", f"aha_sudden_{item_id}")
        aha_surprise = rating_slider("这个答案让你有多惊讶", f"aha_surprise_{item_id}")
        confidence = rating_slider("你对最终答案的信心", f"final_conf_{item_id}")
        known_story = st.radio("你之前是否知道这个谜题或答案？", ["否", "不确定", "是"], horizontal=True, key=f"known_{item_id}")
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
