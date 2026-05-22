import streamlit as st
import json
import random
import difflib
from datetime import datetime
import pandas as pd
import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Experiment", layout="centered")


# ========= 工具 =========
def map_prob_to_score(prob):
    x = -math.log10(prob + 1e-12)
    return int(max(0, min(100, 100 * (1 - x / 6))))


# ========= 数据加载 =========
@st.cache_data
def load_riddles():
    with open("riddles.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_lookup_stage1(group):
    with open(f"lookup_{group}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_lookup_full(group):
    with open(f"lookup_full_{group}.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_vocab():
    with open("vocab_full.json", "r", encoding="utf-8") as f:
        return set(json.load(f))


# ========= fallback =========
def get_top_matches(word, vocab):
    return difflib.get_close_matches(word, vocab, n=3, cutoff=0.5)


# ========= GSheet =========
def init_gsheet():
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"],
        [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)

    return client.open_by_url(
        st.secrets["gsheet_url"]
    ).sheet1


def safe_append_row(sheet, row):
    try:
        sheet.append_row(row)
    except Exception as e:
        print("GSHEET ERROR:", e)


# ========= 初始化 =========
if "page" not in st.session_state:
    st.session_state.page = "intro"
    st.session_state.idx = 0
    st.session_state.responses = []
    st.session_state.phase = "prior"
    st.session_state.group = None

if "sheet" not in st.session_state:
    st.session_state.sheet = init_gsheet()


# ========= Intro =========
if st.session_state.page == "intro":

    st.title("🧠 欢迎参加文字谜题联想实验")

    pid = st.text_input("请输入参与者ID（注意前两位不要输错了）")

    if st.button("开始实验"):
        if pid.strip():

            st.session_state.pid = pid

            pid_upper = pid.upper()
            group_found = False

            for g in ["FH", "MH", "FN", "MN"]:
                if g in pid_upper:
                    st.session_state.group = g
                    group_found = True
                    break

            if not group_found:
                st.warning("ID中必须包含 FH / MH / FN / MN")
                st.stop()

            riddles = load_riddles()
            order = list(range(len(riddles)))
            random.shuffle(order)

            st.session_state.order = order
            st.session_state.page = "train"
            st.rerun()
        else:
            st.warning("请输入ID")


# ========= TRAIN =========
elif st.session_state.page == "train":

    st.title("训练阶段")

    st.markdown("""
你将看到一个谜题，以及一个候选词。

你的任务是判断：这个词是否可能是谜底。

随后你会看到一个提示词和分数：
- 分数越高 → 与答案越相关 → 应提高判断
- 分数越低 → 相关性弱 → 应降低判断
确保你理解了，然后点击进入
""")

    if st.button("进入正式实验"):
        st.session_state.page = "stage1"
        st.rerun()


# ========= STAGE 1 =========
elif st.session_state.page == "stage1":

    riddles = load_riddles()
    lookup = load_lookup_stage1(st.session_state.group)

    idx = st.session_state.idx

    if idx >= 10:
        st.session_state.page = "stage2"
        st.session_state.idx = 0
        st.rerun()

    item = riddles[st.session_state.order[idx]]
    item_id = item["item_id"]

    prob = lookup[item_id]["final_prob"]
    score = map_prob_to_score(prob)

    st.markdown(f"### 第 {idx+1} 题")
    st.markdown(item["riddle_text"])

    # 🔥 核心修复：显示 anchor
    st.markdown(f"👉 请评估这个词是否是谜底：**{item['anchor_word']}**")

    # ===== Phase 1 =====
    if st.session_state.phase == "prior":

        prior = st.slider(
            f"{item['anchor_word']} 是谜底的可能性",
            0, 100, 50
        )

        if st.button("查看提示"):

            st.session_state.temp_prior = prior
            st.session_state.phase = "update"
            st.rerun()

    # ===== Phase 2 =====
    elif st.session_state.phase == "update":

        st.info("分数越高表示提示词与答案越相关，应提高判断；越低则应降低。")

        st.markdown(f"提示词：{item['cue_word']}")
        st.markdown(f"分数：{score}")

        updated = st.slider("更新你的判断", 0, 100, 50)

        if st.button("提交"):

            record = [
                st.session_state.pid,
                st.session_state.group,
                "stage1",
                item_id,
                st.session_state.temp_prior / 100,
                updated / 100,
                "",
                "",
                datetime.now().isoformat()
            ]

            st.session_state.responses.append(record)
            safe_append_row(st.session_state.sheet, record)

            st.session_state.idx += 1
            st.session_state.phase = "prior"
            st.rerun()


# ========= STAGE 2 =========
elif st.session_state.page == "stage2":

    riddles = load_riddles()
    lookup_full = load_lookup_full(st.session_state.group)
    vocab = load_vocab()

    idx = st.session_state.idx

    if idx >= 10:
        st.session_state.page = "done"
        st.rerun()

    item = riddles[st.session_state.order[idx]]
    target = item["target_word"]

    st.markdown(f"### 第 {idx+1} 题（探索）")
    st.markdown(item["riddle_text"])

    word = st.text_input("输入一个词，探索它与谜底的相关性")

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []

    if st.button("查询"):

        if word in vocab:
            chosen = word
        else:
            matches = get_top_matches(word, vocab)

            if matches:
                st.session_state.suggestions = matches
                st.warning("你是不是想输入：")
                st.stop()
            else:
                st.error("未找到相似词")
                st.stop()

        prob = lookup_full.get(target, {}).get(chosen, 1e-6)
        score = map_prob_to_score(prob)

        st.success(f"{chosen} 的分数：{score}")

        record = [
            st.session_state.pid,
            st.session_state.group,
            "stage2",
            target,
            "",
            "",
            chosen,
            score,
            datetime.now().isoformat()
        ]

        st.session_state.responses.append(record)
        safe_append_row(st.session_state.sheet, record)


    if st.session_state.suggestions:

        for s in st.session_state.suggestions:
            if st.button(f"选择：{s}"):

                prob = lookup_full.get(target, {}).get(s, 1e-6)
                score = map_prob_to_score(prob)

                st.success(f"{s} 的分数：{score}")

                record = [
                    st.session_state.pid,
                    st.session_state.group,
                    "stage2",
                    target,
                    "",
                    "",
                    s,
                    score,
                    datetime.now().isoformat()
                ]

                st.session_state.responses.append(record)
                safe_append_row(st.session_state.sheet, record)

        st.session_state.suggestions = []


    if st.button("下一题"):
        st.session_state.idx += 1
        st.rerun()


# ========= DONE =========
elif st.session_state.page == "done":

    st.success("实验完成 🙏")

    df = pd.DataFrame(
        st.session_state.responses,
        columns=[
            "participant",
            "group",
            "type",
            "item_id_or_target",
            "prior",
            "updated",
            "query",
            "score",
            "timestamp"
        ]
    )

    st.download_button(
        "下载数据",
        df.to_csv(index=False),
        "results.csv"
    )