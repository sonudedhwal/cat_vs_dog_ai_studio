"""Cat vs Dog AI Studio — Streamlit + TensorFlow + Claude (LLM) + MySQL.

Run:  streamlit run app.py
"""
import io
import os

import pandas as pd
import streamlit as st
from PIL import Image

import database as db
import llm
import model as clf

st.set_page_config(page_title="Cat vs Dog AI Studio", page_icon="🐾", layout="wide")


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    return clf.load_model()


def to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def db_ready() -> bool:
    try:
        db.init_db()
        return True
    except Exception as e:
        st.session_state["db_error"] = str(e)
        return False


# ---------- sidebar ----------
st.sidebar.title("🐾 Cat vs Dog AI Studio")
page = st.sidebar.radio(
    "Menu", ["Classify", "History", "Analytics", "AI Pet Expert (Chat)", "Settings"])

key_in = st.sidebar.text_input("Anthropic API key (optional)", type="password",
                               value=os.getenv("ANTHROPIC_API_KEY", ""))
if key_in:
    os.environ["ANTHROPIC_API_KEY"] = key_in

DB_OK = db_ready()
st.sidebar.markdown(f"**MySQL:** {'🟢 connected' if DB_OK else '🔴 not connected'}")
st.sidebar.markdown(f"**LLM:** {'🟢 on' if llm.llm_available() else '⚪ off (no key)'}")
if not DB_OK:
    st.sidebar.error("Start MySQL in XAMPP, then check Settings.")


# ---------- pages ----------
if page == "Classify":
    st.title("Classify an image")
    _, mode = get_model()
    st.caption(f"Model: **{mode}**"
               + ("" if mode == "custom-trained"
                  else " — run `python train.py` to train your own."))

    files = st.file_uploader("Upload cat/dog images", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True)
    use_ai = st.toggle("Generate AI description (LLM)", value=llm.llm_available())

    for f in files or []:
        img = Image.open(f)
        result = clf.predict(img)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption=f.name, use_container_width=True)
        with col2:
            emoji = {"Cat": "🐱", "Dog": "🐶"}.get(result["label"], "❓")
            st.subheader(f"{emoji} {result['label']}  ({result['confidence']:.1%})")
            st.progress(result["cat_prob"], text=f"Cat {result['cat_prob']:.1%}")
            st.progress(result["dog_prob"], text=f"Dog {result['dog_prob']:.1%}")

            desc = None
            if use_ai:
                with st.spinner("Asking the AI..."):
                    desc = llm.describe_pet(to_jpeg_bytes(img), result["label"])
                st.markdown(desc)

            if DB_OK:
                state_key = f"saved_{f.name}_{f.size}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = db.save_prediction(
                        f.name, result, to_jpeg_bytes(img), desc)
                pid = st.session_state[state_key]
                st.caption(f"Saved to MySQL (id {pid})")
                fb = st.radio("Was this right? Correct label:", ["—", "Cat", "Dog"],
                              horizontal=True, key=f"fb_{pid}")
                if fb != "—" and st.button("Submit feedback", key=f"fbb_{pid}"):
                    db.update_feedback(pid, fb)
                    st.success("Thanks! Feedback saved.")
        st.divider()


elif page == "History":
    st.title("Prediction history")
    if not DB_OK:
        st.stop()
    df = db.get_history()
    if df.empty:
        st.info("No predictions yet.")
    else:
        flt = st.multiselect("Filter label", df["label"].unique(), default=list(df["label"].unique()))
        view = df[df["label"].isin(flt)]
        st.dataframe(view.drop(columns=["ai_description"]), use_container_width=True)
        st.download_button("Download CSV", view.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")

        st.subheader("Inspect / delete")
        pid = st.selectbox("Prediction id", view["id"].tolist())
        row = view[view["id"] == pid].iloc[0]
        c1, c2 = st.columns([1, 2])
        data = db.get_image(int(pid))
        if data:
            c1.image(data, use_container_width=True)
        c2.write(f"**{row['label']}** — {row['confidence']:.1%} — {row['created_at']}")
        if row["ai_description"]:
            c2.markdown(row["ai_description"])
        elif llm.llm_available() and data and c2.button("Generate AI description"):
            text = llm.describe_pet(data, row["label"])
            db.update_description(int(pid), text)
            st.rerun()
        if c2.button("🗑 Delete this record"):
            db.delete_prediction(int(pid))
            st.rerun()


elif page == "Analytics":
    st.title("Analytics")
    if not DB_OK:
        st.stop()
    s = db.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predictions", s["total"])
    c2.metric("Avg confidence", f"{s['avg_conf']:.1%}")
    c3.metric("Reviewed by you", s["reviewed"])
    acc = (s["correct"] / s["reviewed"]) if s["reviewed"] else None
    c4.metric("Accuracy (reviewed)", f"{acc:.1%}" if acc is not None else "n/a")

    if s["by_label"]:
        st.subheader("Predictions by label")
        st.bar_chart(pd.Series(s["by_label"]))
    if s["per_day"]:
        st.subheader("Predictions per day")
        st.line_chart(pd.DataFrame(s["per_day"], columns=["day", "count"]).set_index("day"))


elif page == "AI Pet Expert (Chat)":
    st.title("🤖 AI Pet Expert")
    if not llm.llm_available():
        st.warning("Enter an Anthropic API key in the sidebar to use the chat.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about cat/dog care, behavior, breeds..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        ctx = ""
        if DB_OK:
            h = db.get_history(5)
            ctx = ", ".join(f"{r.label} ({r.confidence:.0%})" for r in h.itertuples())
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = llm.chat(st.session_state.chat, ctx)
            st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
    if st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()


elif page == "Settings":
    st.title("Settings & connection")
    st.write("Database settings are read from environment variables "
             "(defaults match XAMPP):")
    st.code(
        "DB_HOST=localhost\nDB_PORT=3306\nDB_USER=root\nDB_PASSWORD=\nDB_NAME=cat_dog_db\n"
        "ANTHROPIC_API_KEY=your_key_here\nCLAUDE_MODEL=claude-sonnet-5-5\n"
        "CAT_DOG_MODEL=cat_dog_model.keras", language="bash")
    if st.button("Test MySQL connection"):
        try:
            db.init_db()
            conn = db.get_conn()
            conn.close()
            st.success("Connected and tables are ready ✅")
        except Exception as e:
            st.error(f"Connection failed: {e}")
    if "db_error" in st.session_state and not DB_OK:
        st.error(st.session_state["db_error"])
