import streamlit as st
import json
import random
from datetime import datetime
import pandas as pd
import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
        # 不打断实验
        print("GSHEET ERROR:", e)


# ========= 初始化 =========
if "page" not in st.session_state:
    st.session_state.page = "intro"
    st.session_state.phase = "prior"
    st.session_state.idx = 0
    st.session_state.order = []
    st.session_state.group = "FH"
    st.session_state.responses = []

if "sheet" not in st.session_state:
    st.session_state.sheet = init_gsheet()


# ========= Intro =========
if st.session_state.page == "intro":

    st.title("🧠 海龟汤")

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

    prob = lookup[item_id]["final_prob"]
    score = map_prob_to_score(prob)

    st.markdown(f"### 第 {idx+1} 题")
    st.markdown(item["riddle_text"])
    

    # ========= Phase 1 =========
    if st.session_state.phase == "prior":

        st.markdown(f"🔹 注意这个词： **{item['anchor_word']}**")

        prior = st.slider(
            "你认为该词作为答案的可能性（直觉判断）",
            0, 100, 50,
            key=f"prior_{idx}"
        )

        if st.button("下一步", key=f"next_{idx}"):

            st.session_state.temp_prior = prior
            st.session_state.phase = "update"
            st.rerun()

    # ========= Phase 2 =========
    elif st.session_state.phase == "update":

        st.markdown(f"🔸 提示：注意这个词 **{item['cue_word']}**")
        st.markdown(f"👉 系统提示：**{item['cue_word']}** 和谜底的关联强度分数为 **{score}/100**")

        updated = st.slider(
    f"在看到提示后，你现在认为 **{item['anchor_word']}** 是谜底或与谜底直接相关的可能性",
    0, 100, 50,
    key=f"updated_{idx}"
)

        confidence = st.slider(
            "你对当前判断的信心",
            0, 100, 50,
            key=f"conf_{idx}"
        )

        if st.button("提交", key=f"submit_{idx}"):

            record = {
                "participant": st.session_state.pid,
                "group": st.session_state.group,
                "item_id": item_id,
                "prior": st.session_state.temp_prior / 100,
                "updated": updated / 100,
                "confidence": confidence / 100,
                "prob": prob,
                "log_prob": -math.log10(prob + 1e-12),
                "display_score": score,
                "timestamp": datetime.now().isoformat()
            }

            # 本地缓存（保留）
st.session_state.responses.append(record)

# ✅ 实时写入（安全版，不影响实验流程）
safe_append_row(
    st.session_state.sheet,
    [
        record["participant"],
        record["group"],
        record["item_id"],
        record["prior"],
        record["updated"],
        record["confidence"],
        record["prob"],
        record["display_score"],
        record["timestamp"]
    ]
)

            st.session_state.idx += 1
            st.session_state.phase = "prior"
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
