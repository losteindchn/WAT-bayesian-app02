import streamlit as st
import json
import random
from datetime import datetime
import pandas as pd
import math

# ========= 页面配置 =========
st.set_page_config(page_title="Riddle Experiment", layout="centered")

# ========= 概率映射 =========
def map_prob_to_score(prob):
    x = -math.log10(prob + 1e-12)
    score = 100 * (1 - x / 6)
    score = max(0, min(100, score))
    return round(score)


# ========= 加载 =========
@st.cache_data
def load_riddles():
    with open("riddles.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_lookup(group):
    with open(f"lookup_{group}.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ========= 初始化 =========
if "page" not in st.session_state:
    st.session_state.page = "intro"
    st.session_state.idx = 0
    st.session_state.order = []
    st.session_state.group = "FH"
    st.session_state.responses = []


# ========= Intro =========
if st.session_state.page == "intro":

    st.title("🧠 语义判断实验")

    pid = st.text_input("请输入参与者ID")

    if st.button("开始实验"):
        if pid.strip():
            st.session_state.pid = pid

            # 自动分组
            pid_upper = pid.upper()
            for g in ["FH", "MH", "FN", "MN"]:
                if g in pid_upper:
                    st.session_state.group = g

            riddles = load_riddles()
            order = list(range(len(riddles)))
            random.shuffle(order)

            st.session_state.order = order[:30]
            st.session_state.page = "trial"
            st.rerun()
        else:
            st.warning("请输入ID")


# ========= Trial =========
elif st.session_state.page == "trial":

    riddles = load_riddles()
    lookup = load_lookup(st.session_state.group)

    idx = st.session_state.idx

    if idx >= len(st.session_state.order):
        st.session_state.page = "done"
        st.rerun()

    item_idx = st.session_state.order[idx]
    item = riddles[item_idx]

    item_id = item["item_id"]

    # ===== 模型真实信号 =====
    prob = lookup[item_id]["final_prob"]

    # ===== 人类可理解展示 =====
    score = map_prob_to_score(prob)

    # ========= UI =========
    st.markdown(f"### 第 {idx+1} 题")

    st.markdown(item["riddle_text"])
    st.markdown(f"🔹 锚点词：**{item['anchor_word']}**")

    # ✅ wording 修正（关键）
    prior = st.slider(
        "你认为该词作为答案的可能性（直觉判断）",
        0, 100, 50,
        key=f"prior_{idx}"
    )

    st.markdown(f"🔸 提示词：**{item['cue_word']}**")

    # ✅ 使用 mapping 后的 score
    st.markdown(f"👉 系统提示：语义关联强度 **{score}/100**")

    updated = st.slider(
        "在看到提示后，你现在的判断",
        0, 100, 50,
        key=f"updated_{idx}"
    )

    confidence = st.slider(
        "你对当前判断的信心",
        0, 100, 50,
        key=f"conf_{idx}"
    )

    # ========= 提交 =========
    if st.button("提交", key=f"submit_{idx}"):

        st.session_state.responses.append({
            "participant": st.session_state.pid,
            "group": st.session_state.group,
            "item_id": item_id,

            # 行为数据
            "prior": prior / 100,
            "updated": updated / 100,
            "confidence": confidence / 100,

            # 核心模型输入
            "prob": prob,
            "log_prob": -math.log10(prob + 1e-12),

            # sanity check用（可选）
            "display_score": score,

            "timestamp": datetime.now().isoformat()
        })

        st.session_state.idx += 1
        st.rerun()


# ========= Done =========
elif st.session_state.page == "done":

    st.success("实验完成！感谢参与 🙏")

    df = pd.DataFrame(st.session_state.responses)

    st.download_button(
        "下载数据",
        df.to_csv(index=False),
        "results.csv",
        "text/csv"
    )