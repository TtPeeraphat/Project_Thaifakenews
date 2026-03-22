
import streamlit as st
import logging
import time
from text_preprocessor import TextPreprocessor
from validators import InputValidator
from ai_engine import get_pipeline, predict_news
import database_ops as db
from scraper_ops import get_content_from_url



logger = logging.getLogger(__name__)


# ==========================================================
# PAGE HEADER
# ==========================================================

def page_header(icon: str, title: str, subtitle: str = ""):
    sub = f"<p style='color:#64748B;font-size:0.9rem;margin:4px 0 0;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div style="padding-bottom:20px;margin-bottom:24px;border-bottom:1px solid #E2E8F0;">
        <h1 style="margin:0">{icon} {title}</h1>
        {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# MAIN HOME PAGE
# ==========================================================

def show_home_improved():

    page_header("🔍", "ตรวจสอบข่าว", "วิเคราะห์เนื้อหาข่าวด้วย AI — ได้ผลลัพธ์ภายใน 3 วินาที")

    check_mode = st.radio(
        "เลือกโหมดการตรวจสอบ",
        ["📝 พิมพ์ / วางเนื้อหา", "🔗 URL ลิงก์ข่าว"],
        horizontal=True,
    )

    input_url = ""
    input_text = ""

    # =====================================================
    # URL MODE
    # =====================================================

    if check_mode == "🔗 URL ลิงก์ข่าว":

        if "input_url" not in st.session_state:
            st.session_state.input_url = ""

        col1, col2 = st.columns([5, 1])

        with col1:
            st.text_input(
                "URL ข่าว",
                key="input_url",
                placeholder="https://www.example.com/news/...",
            )

        with col2:
            if st.button("🗑️"):
                st.session_state.input_url = ""

        input_url = st.session_state.input_url

    # =====================================================
    # TEXT MODE
    # =====================================================

    else:

        if "input_text" not in st.session_state:
            st.session_state.input_text = ""

        with st.expander("💡 ตัวอย่างข่าว"):
            c1, c2 = st.columns(2)

            if c1.button("Fake Example"):
                st.session_state.input_text = (
                    "ข่าวล่าสุด: มนุษย์ต่างดาวลงจอดที่กรุงเทพ ใกล้สยามพารากอน"
                )

            if c2.button("Real Example"):
                st.session_state.input_text = (
                    "รัฐบาลประกาศวันหยุดพิเศษเพิ่มเพื่อกระตุ้นเศรษฐกิจ"
                )

        st.text_area(
            "เนื้อหาข่าว",
            key="input_text",
            height=200,
            placeholder="วางข่าวที่ต้องการตรวจสอบ",
        )

        input_text = st.session_state.input_text

    # =====================================================
    # ANALYZE BUTTON  ← ต้องอยู่ level เดียวกับ if/else ข้างบน
    # =====================================================

    if st.button("🚀 วิเคราะห์ข่าวนี้", use_container_width=True):

        progress = st.progress(0)
        scraper_content = ""

        # STEP 1
        if check_mode == "🔗 URL ลิงก์ข่าว":
            valid = InputValidator.validate_url(input_url)
            if not valid.is_valid:
                st.error(valid.error_message)
                st.stop()
            progress.progress(10)
            title, content = get_content_from_url(input_url)
            if not title:
                st.error("ไม่สามารถดึงข่าวจาก URL")
                st.stop()
            raw_text = f"{title}\n\n{content}"
            scraper_content = content
        else:
            raw_text = str(input_text).strip()

        if not raw_text:
            st.warning("กรุณาใส่ข่าว")
            st.stop()

        progress.progress(30)

        # STEP 2
        validation = InputValidator.validate_text(raw_text)
        if not validation.is_valid:
            st.error(validation.error_message)
            st.stop()
        progress.progress(45)

        # STEP 3
        cleaned_text, ok, msg = TextPreprocessor.preprocess(
            raw_text, max_length=5000, min_length=10,
        )
        if not ok:
            st.error(msg)
            st.stop()
        progress.progress(60)

        # STEP 4
        try:
            pipeline = get_pipeline()
        except Exception as e:
            st.error(f"โหลดโมเดลไม่ได้: {e}")
            st.stop()
        progress.progress(80)

        # STEP 5
        try:
            result = predict_news(cleaned_text, pipeline, content=scraper_content)
            progress.progress(100)
            if result.get("error"):
                st.error(result["error"])
                st.stop()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # SAVE RESULT  ← อยู่ใน if st.button ด้วยกัน
        pred_id = db.create_prediction(
            st.session_state.get("user_id"),
            cleaned_text[:50],
            cleaned_text,
            input_url if input_url else None,
            result["result"],
            result["confidence"],
        )
        st.session_state.current_result = result
        st.session_state.current_pred_id = pred_id
        st.session_state.feedback_given = False
        time.sleep(0.4)
        st.rerun()

    # SHOW RESULT  ← อยู่นอก if st.button
    if "current_result" in st.session_state:
        show_prediction_result(st.session_state.current_result)