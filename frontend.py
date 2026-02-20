import sys
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import altair as alt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from datetime import datetime,timedelta
import asyncio

# --- IMPORT MODULES ของเราเอง ---
import database_ops as db
import ai_engine as ai
from scraper_ops import get_content_from_url

# --- เพิ่มบล็อกนี้เพื่อแก้ปัญหา Playwright บน Windows ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# ----------------------------------------------------

st.markdown("""
<style>
    /* เลือกกลุ่มปุ่มทั้งหมด */
    div[data-testid="stSegmentedControl"] {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* เลือกปุ่มแต่ละปุ่มภายใน */
    div[data-testid="stSegmentedControl"] button {
        border-radius: 0px !important;  /* ทำให้เป็นสี่เหลี่ยม */
        border: none !important;        /* เอาขอบออก */
        background-color: transparent !important; /* พื้นหลังโปร่งใสตอนไม่ได้เลือก */
        margin-bottom: 2px;             /* ระยะห่างระหว่างเมนูแนวตั้ง */
        justify-content: flex-start;    /* จัดตัวอักษรชิดซ้าย (สำหรับแนวตั้ง) */
    }

    /* ตกแต่งตอนเอาเมาส์ไปวาง (Hover) หรือตอนกดเลือก (Active) */
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: rgba(255, 255, 255, 0.1) !important; /* สีพื้นหลังตอนเลือก */
        color: #ff4b4b !important; /* สีตัวอักษรตอนเลือก (ปรับตามธีมคุณ) */
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 📊 ฟังก์ชัน Admin: Model Analytics Dashboard
# ==========================================
def show_model_performance():

    # ==========================================
    # ส่วนที่ 1: Real-time Monitoring (ดึงจาก Predictions ทั้งหมด)
    # ==========================================
    df_all = db.get_model_performance_data() 
    
    st.subheader("1. AI Activity Monitor (Real-time)")
    
    if df_all.empty:
        st.warning("ยังไม่มีข้อมูลการทำนายเข้ามาในระบบ")
        return

    # คำนวณ Metric พื้นฐาน
    total_scans = len(df_all)
    
    # แปลง confidence เป็นตัวเลข
    if 'confidence' in df_all.columns:
        df_all['confidence'] = pd.to_numeric(df_all['confidence'], errors='coerce')
    
    avg_conf = df_all['confidence'].mean()
    fake_count = len(df_all[df_all['prediction'] == 'Fake'])
    real_count = len(df_all[df_all['prediction'] == 'Real'])

    # แสดง Metric 4 ช่อง
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total News Scanned", f"{total_scans:,}", "All time")
    m2.metric("Avg. Confidence", f"{avg_conf:.1f}%", "Model certainty")
    m3.metric("Detected FAKE", f"{fake_count}", f"{(fake_count/total_scans*100):.1f}% rate", delta_color="inverse")
    m4.metric("Detected REAL", f"{real_count}", f"{(real_count/total_scans*100):.1f}% rate")

    st.write("---")
    
    # ==========================================
    # ส่วนที่ 2: Accuracy Evaluation (เฉพาะที่มีเฉลยแล้ว)
    # ==========================================
    st.subheader("2. Accuracy Evaluation (Verified Cases Only)")
    
    # 1. ดึงข้อมูล
    df_evaluated = db.get_evaluated_data()

    has_valid_data = False

    if not df_evaluated.empty and 'status' in df_evaluated.columns and 'prediction' in df_evaluated.columns:
        # ลบช่องว่างหัวท้าย (ถ้ามี)
        df_evaluated['status'] = df_evaluated['status'].astype(str).str.strip()
        
        # กรองเอาเฉพาะแถวที่ status เป็น 'Real' หรือ 'Fake' เท่านั้น
        df_evaluated = df_evaluated[df_evaluated['status'].isin(['Real', 'Fake'])]
        
        # ถ้ากรองแล้วยังมีข้อมูลเหลืออยู่ ถือว่าพร้อมนำไปคำนวณ
        if not df_evaluated.empty:
            has_valid_data = True

    if not has_valid_data:
        st.info("ℹ️ ส่วนวัดผลความแม่นยำจะแสดงเมื่อ Admin ทำการ Review ข้อมูลในหน้า Feedback แล้วเท่านั้น")
        st.caption("Waiting for verification... (Currently 0 verified cases or missing required data)")
        st.progress(0)
    else:
        st.success(f"📈 คำนวณจากข้อมูลที่เฉลยแล้วจำนวน: {len(df_evaluated)} รายการ")
        
        y_true = df_evaluated['status']
        y_pred = df_evaluated['prediction'] 
        
        # กำหนด label เป้าหมาย 
        target_pos = 'Fake' 

        # คำนวณ Metrics
        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, pos_label=target_pos, zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=target_pos, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=target_pos, zero_division=0)

            # ฟังก์ชันวาดหลอดพลัง
            def safe_progress(val):
                return float(max(0.0, min(1.0, val)))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Overall Accuracy", f"{acc*100:.1f}%")
                st.progress(safe_progress(acc))
            with c2:
                st.metric("Precision (Fake)", f"{prec*100:.1f}%")
                st.progress(safe_progress(prec))
            with c3:
                st.metric("Recall (Fake)", f"{rec*100:.1f}%")
                st.progress(safe_progress(rec))
            with c4:
                st.metric("F1 Score", f"{f1*100:.1f}%")
                st.progress(safe_progress(f1))

            # Show Table of verified data
            with st.expander("ดูรายการที่ตรวจสอบแล้ว (Verified Logs)", expanded=True):
                cols_to_show = ['prediction', 'status', 'confidence']
                available_cols = [c for c in cols_to_show if c in df_evaluated.columns]
                st.dataframe(df_evaluated[available_cols], use_container_width=True)
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

import pandas as pd
import streamlit as st

# ==========================================
# 💬 ฟังก์ชัน Admin: Review Feedback
# ==========================================
def show_feedback_review():
    st.title("💬 Review Feedback (ระบบสอน AI)")
    st.write("ตรวจสอบความถูกต้องของ Model จาก Feedback ผู้ใช้งาน และนำไปสอน AI เพิ่มเติม")
    
    # -----------------------------------------------------------
    # 🌟 เพิ่มส่วนหัว: ปุ่มส่งออกข้อมูล (Export Dataset)
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧠 ส่งออกข้อมูลเพื่อสอนโมเดล (Continuous Learning)")
    
    # วางปุ่มในคอลัมน์เพื่อให้ดูสวยงาม
    col_export, col_desc = st.columns([1, 2])
    with col_desc:
        st.caption("ระบบจะดึงข้อมูลที่แอดมินตรวจสอบแล้ว (ข่าวจริง/ข่าวปลอม) รวมถึงข่าว Trending News มาสร้างเป็นไฟล์ CSV สำหรับให้คุณนำไป Train Model ใหม่")
    with col_export:
        if st.button("📥 ดาวน์โหลด Dataset ล่าสุด", type="primary", use_container_width=True):
            with st.spinner("กำลังรวบรวมข้อมูล..."):
                # 1. ดึงข่าวจาก Trending (สมมติว่ามีฟังก์ชันนี้อยู่แล้ว)
                df_trending = db.get_all_trending() 
                
                # 2. ดึงข่าวจาก Feedback เฉพาะที่ตรวจแล้ว ('Real', 'Fake')
                df_feedback = db.get_approved_feedbacks() 
                
                frames = []
                # จัดการข้อมูล Trending
                if not df_trending.empty and 'content' in df_trending.columns and 'label' in df_trending.columns:
                    # เลือกเฉพาะเนื้อหาและ Label
                    frames.append(df_trending[['content', 'label']]) 
                    
                # จัดการข้อมูล Feedback
                if not df_feedback.empty:
                    # สมมติว่าใน DB คอลัมน์ชื่อ text และ status ที่คุณเพิ่งอัปเดตไป
                    df_fb_clean = df_feedback[['text', 'status']].rename(columns={'text': 'content', 'status': 'label'})
                    frames.append(df_fb_clean)
                    
                # รวมร่าง
                if frames:
                    df_final = pd.concat(frames, ignore_index=True)
                    
                    # ลบค่าว่างและแถวซ้ำ (ทำความสะอาดข้อมูลเบื้องต้น)
                    df_final = df_final.dropna(subset=['content', 'label'])
                    df_final = df_final.drop_duplicates(subset=['content'])
                    
                    csv = df_final.to_csv(index=False).encode('utf-8')
                    st.success(f"✅ สร้างไฟล์สำเร็จ! พบข้อมูลทั้งหมด {len(df_final)} รายการ")
                    
                    st.download_button(
                        label="⬇️ คลิกเพื่อบันทึกไฟล์ (CSV)",
                        data=csv,
                        file_name="retrain_dataset.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ ยังไม่มีข้อมูลใหม่สำหรับเทรนโมเดล")
    st.markdown("---")

    # -----------------------------------------------------------
    # 📋 ส่วนเดิม: รายการรอตรวจสอบ (Pending Feedbacks)
    # -----------------------------------------------------------
    pending_items = db.get_pending_feedbacks()
    
    if not pending_items:
        st.success("🎉 เยี่ยมมาก! ไม่มีงานค้าง ตรวจครบหมดแล้ว")
        st.balloons()
        return

    st.info(f"📝 มีรายการรอตรวจสอบทั้งหมด: {len(pending_items)} รายการ")
    
    # 2. แสดงรายการ (Loop)
    for item in pending_items:
        # ส่วนหัว Card
        card_title = f"📰 {item['title'][:50]}..." if item['title'] else "No Title"
        
        with st.expander(f"{card_title} (AI: {item['ai_result']})", expanded=True):
            
            # แบ่งครึ่งซ้ายขวา
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown("#### เนื้อหาข่าว")
                st.write(item['text'])
                st.caption(f"📅 User Feedback Time: {item['timestamp']}")
            
            with c2:
                st.markdown("#### รายละเอียด")
                st.info(f"🤖 **AI ทายว่า:** {item['ai_result']}")
                st.write(f"📊 **ความมั่นใจ:** {item['ai_confidence']}%")
                st.warning(f"💬 **User บอกว่า:** {item['user_comment']}")

            st.write("---")
            st.markdown("### 👨‍⚖️ Admin Decision (เฉลยความจริง)")
            
            # ปุ่ม Action (Real / Fake / Ignore)
            b1, b2, b3 = st.columns(3)
            
            with b1:
                if st.button("✅ ข่าวจริง (Real)", key=f"real_{item['feedback_id']}", type="primary"):
                    if db.update_feedback_status(item['feedback_id'], 'Real'):
                        st.success("บันทึกว่า 'Real' เรียบร้อย!")
                        st.rerun() 
            
            with b2:
                if st.button("❌ ข่าวปลอม (Fake)", key=f"fake_{item['feedback_id']}", type="primary"):
                    if db.update_feedback_status(item['feedback_id'], 'Fake'):
                        st.error("บันทึกว่า 'Fake' เรียบร้อย!")
                        st.rerun()
            
            with b3:
                # ปุ่มข้าม หรือ ลบทิ้ง (ในกรณี Spam)
                if st.button("🗑️ ลบ/ข้าม (Ignore)", key=f"del_{item['feedback_id']}"):
                     if db.update_feedback_status(item['feedback_id'], 'Ignored'):
                        st.warning("ลบรายการนี้แล้ว")
                        st.rerun()
# ==========================================
# 📰 ฟังก์ชัน Admin: Manage Trending News
# ==========================================
def manage_trending_news():
    
    st.title("📰 Manage Trending News")
    
    # ==========================================
    # 🌟 เพิ่มส่วนดาวน์โหลดข้อมูล (Export Dataset)
    # ==========================================
    st.markdown("---")
    col_text, col_btn = st.columns([2, 1])
    
    with col_text:
        st.markdown("**🧠 ส่งออกข้อมูลข่าวเพื่อพัฒนาโมเดล**")
        st.caption("ดาวน์โหลดรายการข่าวทั้งหมดในหน้านี้ไปเป็นไฟล์ .csv สำหรับเทรน AI")
        
    with col_btn:
        st.write("") # ดันปุ่มลงมานิดหน่อยให้สวยงาม
        # 1. ดึงข้อมูลข่าวทั้งหมด
        df_export_news = db.get_all_trending() 
        
        if not df_export_news.empty:
            # 2. กรองเฉพาะคอลัมน์ที่จำเป็น (เนื้อหา และ สถานะความจริง)
            # เช็คว่ามีคอลัมน์ headline/content/label ตามที่คุณใช้งานจริง
            cols = []
            if 'headline' in df_export_news.columns: cols.append('headline')
            if 'content' in df_export_news.columns: cols.append('content')
            if 'label' in df_export_news.columns: cols.append('label')
            
            df_to_download = df_export_news[cols] if cols else df_export_news
            
            # 3. แปลงเป็น CSV
            csv_data = df_to_download.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="⬇️ ดาวน์โหลด Dataset (.csv)",
                data=csv_data,
                file_name="trending_news_dataset.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
        else:
            st.button("ไม่มีข้อมูลให้ดาวน์โหลด", disabled=True, use_container_width=True)
            
    st.markdown("---")


    # ==========================================
    # ส่วนจัดการข่าว (Tabs เดิมของคุณ)
    # ==========================================
    tab1, tab2 = st.tabs(["📋 รายการข่าวปัจจุบัน", "➕ เพิ่มข่าวใหม่"])
    
    with tab1:
        df_news = db.get_all_trending()
        
        if df_news.empty:
            st.info("ℹ️ ยังไม่มีข่าวที่เป็นกระแสในระบบ")
        else:
            # ==========================================
            # ส่วน Filter (กรองข้อมูลและค้นหา)
            # ==========================================
            col_search, col_filter = st.columns([2, 1])
            with col_search:
                search_query = st.text_input("🔍 ค้นหาหัวข้อหรือเนื้อหาข่าว", placeholder="พิมพ์คำค้นหา...")
            with col_filter:
                status_filter = st.selectbox("กรองตามสถานะ", ["All", "Real", "Fake", "Unverified"])
            
            # ทำการกรองข้อมูลใน DataFrame
            if search_query:
                df_news = df_news[
                    df_news['headline'].str.contains(search_query, case=False, na=False) | 
                    df_news['content'].str.contains(search_query, case=False, na=False)
                ]
            if status_filter != "All":
                df_news = df_news[df_news['label'] == status_filter]
                
            st.write(f"📊 พบข้อมูลทั้งหมด: **{len(df_news)}** ข่าว")
            st.markdown("---")
            
            # ==========================================
            # วนลูปแสดงรายการข่าว
            # ==========================================
            for index, row in df_news.iterrows():
                # ตั้งชื่อ Key สำหรับเช็คว่ากำลังอยู่ใน "โหมดแก้ไข" ของข่าวนี้อยู่หรือเปล่า
                edit_state_key = f"edit_mode_{row['id']}"
                if edit_state_key not in st.session_state:
                    st.session_state[edit_state_key] = False

                with st.expander(f"[{row['label']}] {row['headline']}", expanded=st.session_state[edit_state_key]):
                    
                    # 🔴 ถ้า "ไม่ได้" อยู่ในโหมดแก้ไข (แสดงผลปกติ)
                    if not st.session_state[edit_state_key]:
                        st.write("**เนื้อหา:**")
                        st.write(row['content'])
                        
                        upload_time = row.get('updated_at', '-')
                        if upload_time != '-':
                            try:
                                upload_time = str(upload_time).replace("T", " ")[:16]
                            except:
                                pass
                        st.caption(f"🕒 อัปเดตล่าสุด: {upload_time}")
                        
                        # ปุ่มกด แก้ไข และ ลบ
                        c1, c2, c3 = st.columns([1, 1, 4])
                        with c1:
                            if st.button("✏️ แก้ไข", key=f"btn_edit_{row['id']}"):
                                st.session_state[edit_state_key] = True
                                st.rerun()
                        with c2:
                            if st.button("🗑️ ลบ", key=f"del_{row['id']}"):
                                if db.delete_trending(row['id']):
                                    st.success("✅ ลบข่าวเรียบร้อยแล้ว!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ เกิดข้อผิดพลาดในการลบข่าว")

                    # 🟢 ถ้า "อยู่" ในโหมดแก้ไข (เปลี่ยนเป็นฟอร์ม)
                    else:
                        st.write("📝 **แก้ไขรายละเอียดข่าว**")
                        
                        # ช่องกรอกข้อมูลที่ดึงค่าเดิมมาแสดงรอไว้
                        edit_headline = st.text_input("หัวข้อข่าว", value=row['headline'], key=f"head_{row['id']}")
                        edit_content = st.text_area("เนื้อหาข่าว", value=row['content'], height=150, key=f"cont_{row['id']}")
                        
                        # หาระดับ Index ของตัวเลือกเดิมสำหรับ Selectbox
                        label_options = ["Fake", "Real", "Unverified"]
                        current_label_index = label_options.index(row['label']) if row['label'] in label_options else 0
                        edit_label = st.selectbox("สถานะของข่าว", label_options, index=current_label_index, key=f"label_{row['id']}")
                        
                        # ปุ่มบันทึกการแก้ไข หรือ ยกเลิก
                        b1, b2, b3 = st.columns([1, 1, 4])
                        with b1:
                            if st.button("💾 บันทึก", key=f"save_{row['id']}", type="primary"):
                                if not edit_headline.strip() or not edit_content.strip():
                                    st.warning("⚠️ กรุณากรอกข้อมูลให้ครบถ้วน")
                                else:
                                    if db.update_trending(row['id'], edit_headline, edit_content, edit_label):
                                        st.success("✅ อัปเดตข้อมูลสำเร็จ!")
                                        st.session_state[edit_state_key] = False # ออกจากโหมดแก้ไข
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("❌ เกิดข้อผิดพลาดในการอัปเดต")
                        with b2:
                            if st.button("❌ ยกเลิก", key=f"cancel_{row['id']}"):
                                st.session_state[edit_state_key] = False # ออกจากโหมดแก้ไข
                                st.rerun()

    # ==========================================
    # Tab 2: ฟอร์มเพิ่มข่าวใหม่
    # ==========================================
    with tab2:
        with st.form("add_trending_form", clear_on_submit=True):
            st.write("📝 **ฟอร์มเพิ่มข่าวใหม่ลงในระบบ**")
            
            new_headline = st.text_input("หัวข้อข่าว (Headline)", placeholder="พิมพ์พาดหัวข่าวที่นี่...")
            new_content = st.text_area("เนื้อหาข่าว (Content)", placeholder="พิมพ์รายละเอียดข่าวโดยย่อ...", height=150)
            new_label = st.selectbox("สถานะของข่าว (Label)", ["Fake", "Real", "Unverified"])
            
            submitted = st.form_submit_button("บันทึกข่าว 💾", type="primary", use_container_width=True)
            
            if submitted:
                if not new_headline.strip() or not new_content.strip():
                    st.warning("⚠️ กรุณากรอกพาดหัวข่าวและเนื้อหาให้ครบถ้วน")
                else:
                    success = db.create_trending(new_headline, new_content, new_label)
                    if success:
                        st.success("✅ เพิ่มข่าวลงระบบเรียบร้อยแล้ว!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ เกิดข้อผิดพลาดในการบันทึกข้อมูล")

def show_system_analytics():
    # ==========================================
    # 1. FETCH & PROCESS DATA (ดึงข้อมูลจริงจาก DB)
    # ==========================================
    with st.spinner("กำลังประมวลผลข้อมูล Analytics..."):
        data = db.get_system_analytics_data()
        
    total_users = data["total_users"]
    df_preds = data["df_preds"]
    df_logs = data["df_logs"]
    
    # ตัวแปรสำหรับเก็บค่าสถิติ
    total_checks = 0
    avg_daily_users = 0
    peak_hour = "00:00"
    
    # ------------------------------------------
    # คำนวณข้อมูล Predictions (ข่าวที่ถูกตรวจ)
    # ------------------------------------------
    if not df_preds.empty:
        total_checks = len(df_preds)
        df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])
        df_preds['date'] = df_preds['timestamp'].dt.date
        df_preds['hour'] = df_preds['timestamp'].dt.hour
        
        # หา Peak Hour (ชั่วโมงที่มีการใช้งานมากสุด)
        if not df_preds['hour'].empty:
            mode_hour = df_preds['hour'].mode()
            if not mode_hour.empty:
                peak_hour = f"{int(mode_hour[0]):02d}:00"
                
    # ------------------------------------------
    # คำนวณข้อมูล Active Users จาก Logs (7 วันย้อนหลัง)
    # ------------------------------------------
    if not df_logs.empty:
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        df_logs['date'] = df_logs['timestamp'].dt.date
        # นับจำนวน User ที่ไม่ซ้ำกันในแต่ละวัน แล้วหาค่าเฉลี่ย
        daily_users = df_logs.groupby('date')['user_id'].nunique()
        avg_daily_users = int(daily_users.mean()) if not daily_users.empty else 0

    # ==========================================
    # 2. KPI METRICS (4 การ์ดด้านบน)
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.container(border=True)
        st.metric(label="👥 Total Users", value=f"{total_users:,}")
        
    with col2:
        st.container(border=True)
        st.metric(label="📄 Total Checks", value=f"{total_checks:,}")
        
    with col3:
        st.container(border=True)
        st.metric(label="📈 Avg Daily Active Users", value=f"{avg_daily_users:,}", help="เฉลี่ยจากระบบย้อนหลัง 7 วัน")
        
    with col4:
        st.container(border=True)
        st.metric(label="🔥 Peak Hour", value=peak_hour, help="ช่วงเวลาที่มีการเช็คข่าวเยอะที่สุด")

    st.write("") # เว้นบรรทัด

    # ==========================================
    # 3. CHARTS ROW 1 (Trends & Classification)
    # ==========================================
    chart_col1, chart_col2 = st.columns([1.5, 1]) 

    with chart_col1:
        with st.container(border=True):
            st.markdown("**Daily Usage Trends (Last 7 Days)**")
            st.caption("Users and checks over time")
            
            # สร้างข้อมูล 7 วันย้อนหลังเป็นฐาน (เพื่อไม่ให้กราฟแหว่งถ้าบางวันไม่มีข้อมูล)
            last_7_days = [(datetime.now() - timedelta(days=i)).date() for i in range(6, -1, -1)]
            df_trends = pd.DataFrame({'Date': last_7_days})
            
            # รวมข้อมูล Checks
            checks_per_day = df_preds.groupby('date').size() if not df_preds.empty else pd.Series()
            df_trends['News Checks'] = df_trends['Date'].map(checks_per_day).fillna(0).astype(int)
            
            # รวมข้อมูล Active Users
            users_per_day = df_logs.groupby('date')['user_id'].nunique() if not df_logs.empty else pd.Series()
            df_trends['Users'] = df_trends['Date'].map(users_per_day).fillna(0).astype(int)
            
            # แปลงวันที่เป็น String เพื่อให้ Plotly แสดงผลสวยงาม
            df_trends['Date'] = pd.to_datetime(df_trends['Date']).dt.strftime('%Y-%m-%d')
            
            # สร้าง Area Chart
            fig_trend = px.area(
                df_trends, 
                x='Date', 
                y=['News Checks', 'Users'],
                color_discrete_sequence=['#42b996', '#2a8b94']
            )
            fig_trend.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                legend_title_text='',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

    with chart_col2:
        with st.container(border=True):
            st.markdown("**Classification Results**")
            st.caption("Distribution of real vs fake news")
            
            if not df_preds.empty and 'result' in df_preds.columns:
                # ทำความสะอาดข้อมูลตัวพิมพ์เล็ก/ใหญ่ และนับจำนวน
                df_preds['result'] = df_preds['result'].astype(str).str.capitalize()
                df_class = df_preds['result'].value_counts().reset_index()
                df_class.columns = ['Result', 'Count']
                
                # เปลี่ยนชื่อให้สวยงาม
                df_class['Result'] = df_class['Result'].replace({'Real': 'Real News', 'Fake': 'Fake News'})
                
                fig_pie = px.pie(
                    df_class, 
                    values='Count', 
                    names='Result',
                    hole=0.0, 
                    color='Result',
                    color_discrete_map={'Real News': '#13c276', 'Fake News': '#ef4444', 'Error': '#cbd5e1'} 
                )
                fig_pie.update_traces(textposition='outside', textinfo='percent+label')
                fig_pie.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
            else:
                st.info("ยังไม่มีข้อมูลการตรวจจับข่าวเพียงพอ")

    # ==========================================
    # 4. CHARTS ROW 2 (Activity by Time)
    # ==========================================
    with st.container(border=True):
        st.markdown("**Activity by Time of Day**")
        st.caption("Number of news checks throughout the day (All Time)")
        
        # สร้าง DataFrame สำหรับชั่วโมง 00:00 - 23:00
        df_time = pd.DataFrame({'Hour': range(24)})
        
        if not df_preds.empty:
            hourly_counts = df_preds.groupby('hour').size()
            df_time['Number of Checks'] = df_time['Hour'].map(hourly_counts).fillna(0).astype(int)
        else:
            df_time['Number of Checks'] = 0
            
        df_time['Time'] = df_time['Hour'].apply(lambda x: f"{x:02d}:00")
        
        fig_time = px.line(
            df_time, 
            x='Time', 
            y='Number of Checks',
            markers=True,
            color_discrete_sequence=['#8b5cf6']
        )
        fig_time.update_traces(line_shape='spline') 
        fig_time.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_time, use_container_width=True, key="time_chart")

    # ==========================================
    # 5. RECENT SYSTEM LOGS
    # ==========================================
    # 🌟 1. เพิ่ม height=400 (หรือความสูงที่ต้องการ) เพื่อให้เกิด Scroll bar
    with st.container(border=True, height=400): 
        st.markdown("**Recent System Logs**")
        st.caption("Latest system events and activities")
        
        # 🌟 2. เปลี่ยน limit จาก 5 เป็น 100 (หรือ 500 ตามต้องการ) เพื่อให้มีข้อมูลให้เลื่อนดู
        recent_logs = db.get_system_logs(limit=100) 
        
        if recent_logs:
            for row in recent_logs:
                ts, user, action, details, level = row
                
                # จัดรูปแบบวันที่
                try:
                    fmt_time = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    fmt_time = ts
                    
                log_msg = f"{action}: {details} (User: {user})"
                
                # กำหนดสีตาม Level ของ Log
                color_tag = "#6c757d" # Default (INFO)
                if level == "ERROR":
                    color_tag = "#ef4444"
                elif level == "WARNING":
                    color_tag = "#f59e0b"
                    
                st.markdown(
                    f"""
                    <div style="background-color: #1e1e24; padding: 10px; border-radius: 5px; margin-bottom: 8px; font-family: monospace; font-size: 14px; border-left: 4px solid {color_tag};">
                        <span style="color: #8b949e;">[{fmt_time}]</span> <span style="color: #e6edf3;">{log_msg}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent system logs found.")


#manage_users() ฟังก์ชันนี้จะถูกเพิ่มในอนาคตสำหรับการจัดการผู้ใช้โดย Admin 
def manage_users_page():
    st.caption("View and manage user accounts, permissions, and status")
    st.markdown("---")

    # ==========================================
    # 1. ดึงข้อมูล
    # ==========================================
    with st.spinner("Loading user data..."):
        df_users = db.get_user_management_data()

    if df_users.empty:
        st.warning("ไม่พบข้อมูลผู้ใช้ในระบบ")
        return

    # จัดรูปแบบวันที่ให้สวยงาม
    df_users['created_at'] = pd.to_datetime(df_users['created_at']).dt.strftime('%b %d, %Y')
    if 'last_active' in df_users.columns:
        df_users['last_active'] = pd.to_datetime(df_users['last_active']).dt.strftime('%b %d, %Y').fillna('Never')
    else:
        df_users['last_active'] = 'Never'

    # คำนวณ KPI
    total_users = len(df_users)
    active_users = len(df_users[df_users['status'] == 'active']) if 'status' in df_users.columns else total_users
    total_admins = len(df_users[df_users['role'] == 'admin'])
    total_checks = df_users['checks'].sum()

    # ==========================================
    # 2. แสดงผล KPI (4 ช่อง)
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.container(border=True).metric(label="👥 Total Users", value=f"{total_users:,}")
    with col2:
        st.container(border=True).metric(label="🟢 Active Users", value=f"{active_users:,}")
    with col3:
        st.container(border=True).metric(label="🛡️ Administrators", value=f"{total_admins:,}")
    with col4:
        st.container(border=True).metric(label="📄 Total Checks", value=f"{total_checks:,}")

    st.write("")

    # ==========================================
    # 3. Filter & Search สำหรับตาราง
    # ==========================================
    st.markdown("#### User List")
    st.caption("All registered users and their information")
    
    col_search, col_filter, _ = st.columns([2, 1, 1])
    with col_search:
        search_query = st.text_input("🔍 ค้นหาผู้ใช้ในตาราง (Email หรือ Username)", placeholder="พิมพ์เพื่อค้นหา...")
    with col_filter:
        role_filter = st.selectbox("กรองตามสิทธิ์ (Role)", ["All", "user", "admin"])

    # ประมวลผลการค้นหา (Filter DataFrame)
    df_display = df_users.copy()
    if search_query:
        df_display = df_display[df_display['email'].str.contains(search_query, case=False, na=False) | 
                                df_display['username'].str.contains(search_query, case=False, na=False)]
    if role_filter != "All":
        df_display = df_display[df_display['role'] == role_filter]

    # ==========================================
    # 4. แสดงตารางผู้ใช้งาน
    # ==========================================
    # 🚨 เพิ่ม 'id' เข้ามาแสดงผลด้วย เพื่อให้แอดมินดู ID ไปพิมพ์แก้ไขได้ง่ายขึ้น
    display_cols = {
        'id': 'ID',
        'email': 'Email',
        'role': 'Role',
        'status': 'Status',
        'checks': 'Checks',
        'created_at': 'Joined',
        'last_active': 'Last Active'
    }
    
    valid_cols = [c for c in display_cols.keys() if c in df_display.columns]
    df_table = df_display[valid_cols].rename(columns=display_cols)
    
    st.dataframe(
        df_table, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Checks": st.column_config.NumberColumn("Checks", format="%d 📈"),
            "Role": st.column_config.TextColumn("Role", width="small"),
        }
    )

    # ==========================================
    # 5. ส่วนของการจัดการบัญชี (Action Menu - แบบใหม่ ไม่ใช้ Dropdown)
    # ==========================================
    st.markdown("---")
    st.markdown("#### ⚙️ User Actions (จัดการบัญชีผู้ใช้)")
    
    with st.expander("คลิกเพื่อแก้ไขสิทธิ์และสถานะของผู้ใช้งาน", expanded=True):
        st.write("🔍 **ค้นหาบัญชีที่ต้องการแก้ไข**")
        
        # 🚨 ใช้ Text Input ให้พิมพ์หาแทนการเลื่อน Dropdown
        search_edit = st.text_input("ระบุ Email หรือ ID ของผู้ใช้ที่ต้องการจัดการ:", placeholder="เช่น 1 หรือ admin@gmail.com")
        
        if search_edit.strip():
            # ลองค้นหาจาก ID ก่อน (ถ้าผู้ใช้พิมพ์ตัวเลข)
            if search_edit.strip().isdigit():
                target_user = df_users[df_users['id'] == int(search_edit.strip())]
            else:
                # ถ้าไม่ใช่ตัวเลข ให้หาจาก Email แบบ Exact Match หรือ Contains
                target_user = df_users[df_users['email'].str.contains(search_edit.strip(), case=False, na=False)]
                
            # ตรวจสอบผลการค้นหา
            if target_user.empty:
                st.warning("❌ ไม่พบผู้ใช้งานที่ตรงกับข้อมูลที่ระบุ")
            elif len(target_user) > 1:
                st.info("⚠️ พบผู้ใช้งานหลายคนที่มีคำค้นหานี้ (กรุณาระบุ Email ให้ครบถ้วน หรือพิมพ์เป็น ID แทน)")
                st.dataframe(target_user[['id', 'email', 'role']], hide_index=True)
            else:
                # กรณีเจอผู้ใช้ 1 คนพอดี (พร้อมให้แก้ไข)
                user_data = target_user.iloc[0]
                selected_id = user_data['id']
                
                st.success(f"✅ พบผู้ใช้งาน: **{user_data['email']}** (ID: {selected_id})")
                
                # ไม่ให้ Admin แก้ไขตัวเอง 
                if selected_id == st.session_state.get('user_id'):
                    st.info("ℹ️ คุณไม่สามารถเปลี่ยนแปลงสิทธิ์หรือระงับบัญชีของตัวคุณเองได้")
                else:
                    st.write("---")
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        new_role = st.radio("ระดับสิทธิ์ (Role)", ["user", "admin"], index=0 if user_data['role'] == 'user' else 1, key="role_edit")
                    
                    with c2:
                        current_status = user_data['status'] if 'status' in user_data else 'active'
                        new_status = st.radio("สถานะบัญชี (Status)", ["active", "inactive"], index=0 if current_status == 'active' else 1, key="status_edit")
                    
                    with c3:
                        st.write("")
                        st.write("")
                        if st.button("💾 บันทึกการเปลี่ยนแปลง", type="primary"):
                            if db.update_user_role_status(selected_id, new_role, new_status):
                                # บันทึก Log
                                db.log_system_event(
                                    st.session_state.get('user_id'), 
                                    "UPDATE_USER", 
                                    f"Updated user ID {selected_id} to Role:{new_role}, Status:{new_status}", 
                                    "WARNING" 
                                )
                                st.success(f"อัปเดตบัญชี {user_data['email']} สำเร็จ!")
                                import time
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("เกิดข้อผิดพลาดในการอัปเดตข้อมูล")

# ==========================================
# 🕒 ฟังก์ชันแปลงเวลา
# ==========================================
def time_ago(timestamp_str):
    """แปลง Timestamp เป็นคำว่า 'X mins ago'"""
    try:
        if isinstance(timestamp_str, str):
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S") 
        else:
            dt = timestamp_str
        
        diff = datetime.now() - dt
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "Just now"
        elif seconds < 3600:
            return f"{int(seconds // 60)} mins ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours ago"
        else:
            return f"{int(seconds // 86400)} days ago"
    except:
        return str(timestamp_str)


# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Config)
# ==========================================
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🧠",
    layout="wide"
)

# เริ่มต้น Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = ""
    st.session_state['role'] = ""

# State สำหรับจัดการหน้าจอ (Page Router)
if 'reset_mode' not in st.session_state:
    st.session_state['reset_mode'] = False
if 'register_mode' not in st.session_state:
    st.session_state['register_mode'] = False
if 'otp_sent' not in st.session_state:
    st.session_state['otp_sent'] = False
if 'reset_email_temp' not in st.session_state:
    st.session_state['reset_email_temp'] = ""

# ==========================================
# 2. ส่วน Logic การเลือกแสดงผลหน้าจอ
# ==========================================

# ------------------------------------------
# CASE A: หน้า Reset Password (เต็มจอ)
# ------------------------------------------
if st.session_state['reset_mode']:
    st.title("🔑 กู้คืนรหัสผ่าน (Reset Password)")
    st.markdown("---")
    
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.info("📧 ระบบจะส่งรหัส OTP ไปยังอีเมลของคุณเพื่อยืนยันตัวตน")
        
        if not st.session_state['otp_sent']:
            email_input = st.text_input("กรอกอีเมลที่ใช้สมัครสมาชิก", key="reset_email_input")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("ส่งรหัส OTP", use_container_width=True, type="primary"):
                    if email_input:
                        with st.spinner("กำลังส่งอีเมล..."):
                            success, message = db.send_otp_email(email_input)
                        if success:
                            st.success(message)
                            st.session_state['otp_sent'] = True
                            st.session_state['reset_email_temp'] = email_input
                        else:
                            st.error(message)
                    else:
                        st.warning("กรุณากรอกอีเมล")
            with c2:
                if st.button("ยกเลิก / กลับไปหน้า Login", use_container_width=True):
                    st.session_state['reset_mode'] = False
                    st.rerun()

        else:
            st.success(f"✅ ส่ง OTP ไปที่ {st.session_state['reset_email_temp']} แล้ว")
            otp_input = st.text_input("กรอกรหัส OTP 6 หลัก", max_chars=6)
            new_pass = st.text_input("ตั้งรหัสผ่านใหม่", type="password", key="new_pass_reset")
            confirm_pass = st.text_input("ยืนยันรหัสผ่านใหม่", type="password", key="confirm_pass_reset")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("ยืนยันการเปลี่ยนรหัสผ่าน", use_container_width=True, type="primary"):
                    if new_pass != confirm_pass:
                        st.error("รหัสผ่านไม่ตรงกัน")
                    else:
                        success, message = db.verify_otp_and_reset(st.session_state['reset_email_temp'], otp_input, new_pass)
                        if success:
                            st.balloons()
                            st.success(message)
                            time.sleep(2)
                            st.session_state['reset_mode'] = False
                            st.session_state['otp_sent'] = False
                            st.rerun()
                        else:
                            st.error(message)
            with c2:
                if st.button("ย้อนกลับ", use_container_width=True):
                    st.session_state['otp_sent'] = False
                    st.rerun()

# ------------------------------------------
# CASE B: หน้า Register (เต็มจอ)
# ------------------------------------------
elif st.session_state['register_mode']:
    st.title("📝 สมัครสมาชิกใหม่ (Register)")
    st.markdown("---")
    
    col_reg = st.columns([1, 2, 1])
    with col_reg[1]: 
        new_user = st.text_input("Username ตั้งใหม่", key="reg_user")
        new_email = st.text_input("Email", key="reg_email")
        new_pw = st.text_input("Password ตั้งใหม่", type="password", key="reg_pw")
        confirm_pw = st.text_input("ยืนยัน Password", type="password", key="reg_pw_con")
        
        st.write("") 
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ ยืนยันการสมัคร", type="primary", use_container_width=True):
                if not new_user or not new_pw or not new_email:
                    st.warning("กรุณากรอกข้อมูลให้ครบ")
                elif new_pw != confirm_pw:
                    st.error("รหัสผ่านไม่ตรงกัน")
                else:
                    if db.create_user(new_user, new_pw, new_email):
                        st.success("สมัครสมาชิกสำเร็จ! กำลังกลับไปหน้า Login...")
                        time.sleep(2)
                        st.session_state['register_mode'] = False
                        st.rerun()
                    else:
                        st.error("Username หรือ Email นี้มีผู้ใช้งานแล้ว")
        
        with b2:
            if st.button("⬅️ กลับไปหน้า Login", use_container_width=True):
                st.session_state['register_mode'] = False
                st.rerun()

# ------------------------------------------
# CASE C: หน้า Login (Default)
# ------------------------------------------
elif not st.session_state['logged_in']:
    
    col_layout = st.columns([1, 1.5, 1])
    
    with col_layout[1]:
        col_header = st.columns([2, 1]) 
        with col_header[0]:
            st.markdown("## 🧠 AI Fake News")
            st.caption("เข้าสู่ระบบเพื่อใช้งาน")
        with col_header[1]:
            st.image("https://cdn-icons-png.flaticon.com/512/3021/3021707.png", width=100)
        
        st.markdown("---")
        
        user = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        
        if st.button("🚀 เข้าสู่ระบบ", type="primary", use_container_width=True):
            user_data = db.authenticate_user(user, pw)
            
            if user_data:
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user_data[0]
                st.session_state['username'] = user_data[1]
                st.session_state['role'] = user_data[2]
                
                db.log_system_event(
                    user_id=user_data[0],
                    action="USER_LOGIN",
                    details=f"User '{user_data[1]}' logged in successfully",
                    level="INFO"
                )
                
                st.success(f"ยินดีต้อนรับ {user}!")
                time.sleep(0.5)
                st.rerun() 
                
            else:
                db.log_system_event(
                    user_id=None,
                    action="LOGIN_FAILED",
                    details=f"Failed login attempt for username: {user}",
                    level="WARNING"
                )
                st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
        
        col_actions = st.columns([1, 0.1, 1]) 
        
        with col_actions[0]:
            if st.button("🔑 ลืมรหัสผ่าน?", use_container_width=True):
                st.session_state['reset_mode'] = True
                st.rerun()
                
        with col_actions[2]:
            if st.button("📝 สมัครสมาชิก", use_container_width=True):
                st.session_state['register_mode'] = True
                st.rerun()

# ------------------------------------------
# CASE D: ล็อกอินแล้ว (เข้าสู่ระบบหลัก)
# ------------------------------------------
else:
    if 'active_menu' not in st.session_state:
        st.session_state.active_menu = "🏠 หน้าหลัก"

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=80)
        st.markdown(f"### Hello, {st.session_state.get('username', 'User')}")
        st.caption(f"Role: {st.session_state.get('role', 'user').upper()}")
        st.divider()

        st.markdown("### 📌 เมนูหลัก")
        general_menu_options = [
            "🏠 หน้าหลัก", 
            "📜 ประวัติการตรวจสอบ", 
            "🔥 ข่าวที่เป็นกระแส", 
            "👤 ข้อมูลส่วนตัว"
        ]

        for option in general_menu_options:
            is_active = st.session_state.active_menu == option
            if st.button(option, key=f"nav_{option}", 
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_menu = option
                st.rerun()

        if st.session_state.get('role') == 'admin':
            st.divider()
            st.markdown("### 🛠️ Admin Panel")
            st.caption("Manage system settings")
            
            admin_menu_options = {
                "📊 Dashboard": "dashboard",
                "📈 Model Performance": "model_performance",
                "📰 Manage News": "manage_news",
                "💬 Review Feedback": "review_feedback",
                "🔬 System Analytics": "analytics",
                "👥 Manage Users": "manage_users"
            }
            
            for label, key_suffix in admin_menu_options.items():
                is_active = st.session_state.get('active_menu') == label
                
                if st.button(
                    label, 
                    key=f"admin_nav_{key_suffix}", 
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.active_menu = label
                    st.rerun()

    menu = st.session_state.active_menu

    # =========================================================
    #  PAGE ROUTING (การนำทางหน้าต่างๆ)
    # =========================================================
    
    # 🏠 หน้าหลัก
    if menu == "🏠 หน้าหลัก":
        st.title("🔍 ตรวจสอบความน่าเชื่อถือของข่าว")

        check_mode = st.radio(
            "เลือกวิธีการนำเข้าข้อมูล:", 
            ["📝 พิมพ์ข้อความ/วางเนื้อหา", "🔗 วางลิงก์ข่าว (URL)"], 
            horizontal=True
        )

        input_url = None
        input_text = ""

        if check_mode == "🔗 วางลิงก์ข่าว (URL)":
            input_url = st.text_input("วางลิงก์ข่าว (URL) ที่ต้องการตรวจสอบ:")
        else:
            with st.expander("📝 ลองใช้ข่าวตัวอย่าง (Demo News)", expanded=False):
                col_mock1, col_mock2 = st.columns(2)
                if col_mock1.button("👽 ข่าว Aliens (Fake)"):
                    st.session_state['input_text'] = "ข่าวล่าสุด: มนุษย์ต่างดาวลงจอดที่กรุงเทพฯ ใกล้กับสยามพารากอน! พยานระบุว่าพวกมันมีสีเขียวและเป็นมิตร"
                if col_mock2.button("🏛️ ข่าวรัฐบาล (Real)"):
                    st.session_state['input_text'] = "รัฐบาลประกาศวันหยุดพิเศษเพิ่มอีก 1 วัน เพื่อกระตุ้นเศรษฐกิจและการท่องเที่ยวในช่วงเทศกาล"
            
            input_val = st.session_state.get('input_text', "")
            input_text = st.text_area("เนื้อหาข่าวที่ต้องการวิเคราะห์:", value=input_val, height=200)

        if st.button("🚀 Analyze News (ตรวจสอบข่าว)", type="primary", use_container_width=True):
            clean_text = ""
            source_type = ""

            if check_mode == "🔗 วางลิงก์ข่าว (URL)":
                if not input_url:
                    st.warning("⚠️ กรุณาวางลิงก์ข่าว (URL) ก่อนทำการตรวจสอบ")
                    st.stop() 
                
                with st.spinner("⏳ กำลังดึงข้อมูลข่าวจากลิงก์..."):
                    title, content = get_content_from_url(input_url)
                    
                    if title and not str(content).startswith("Error"):
                        st.success(f"ดึงข้อมูลสำเร็จ: {title}")
                        clean_text = f"{title}\n\n{content}"
                        source_type = "URL"
                    else:
                        st.error(f"❌ ไม่สามารถดึงข้อมูลได้: {content}")
                        st.stop() 
            else:
                clean_text = str(input_text).strip()
                source_type = "Text"
                if not clean_text:
                    st.warning("⚠️ กรุณาใส่เนื้อหาข่าว")
                    st.stop()

            with st.spinner("🧠 AI กำลังวิเคราะห์เนื้อหา..."):
                try:
                    sanitized_text = re.sub(r'\s+', ' ', clean_text).strip()
                    result = ai.predict_news(sanitized_text)
                    
                    if result is not None:
                        time.sleep(0.5) 
                        
                        res_label = result.get('result', 'Error')
                        res_conf = result.get('confidence', 0.0)
                        
                        log_msg = f"Analyzed: {clean_text[:30]}... Result: {res_label}"
                        db.log_system_event(
                            st.session_state.get('user_id'), 
                            "AI_PREDICT", 
                            log_msg, 
                            "INFO" if res_label != 'Error' else "ERROR"
                        )
                        
                        pred_id = db.create_prediction(
                            st.session_state.get('user_id'), 
                            clean_text[:50]+"...", 
                            clean_text, 
                            input_url if input_url else None,
                            res_label, 
                            res_conf
                        )
                        
                        st.session_state['current_result'] = result
                        st.session_state['current_pred_id'] = pred_id
                        st.session_state['feedback_given'] = False
                        st.rerun()
                    else:
                        st.error("AI ไม่ตอบสนอง (None Result)")
                        
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")

        if 'current_result' in st.session_state:
            res = st.session_state['current_result']
            label = res['result']
            conf = res['confidence']
            
            if conf < 70.0:
                bg_color = "#332900"
                border_color = "#FFC107"
                text_color = "#FFC107"
                icon = "⚠️"
                display_label = "UNVERIFIED"
                desc = "ไม่สามารถยืนยันได้ (ข้อมูลไม่เพียงพอ หรือ AI ยังมีความลังเลสูง)"
            elif label == "Fake":
                bg_color = "#381E1E"
                border_color = "#FF4B4B"
                text_color = "#FF4B4B"
                icon = "🚨"
                display_label = "FAKE"
                desc = "เนื้อหานี้มีลักษณะเป็น ข่าวปลอม หรือ บิดเบือน"
            else:
                bg_color = "#1E3822"
                border_color = "#56F066"
                text_color = "#56F066"
                icon = "✅"
                display_label = "REAL"
                desc = "เนื้อหานี้ดูสมเหตุสมผลและ น่าเชื่อถือ"

            st.markdown(f"""
            <div style="
                padding: 25px;
                border-radius: 15px;
                background-color: {bg_color};
                border-left: 8px solid {border_color};
                margin-top: 20px;
                margin-bottom: 20px;">
                <h2 style="color: {text_color}; margin:0;">{icon} {display_label} ({conf:.1f}%)</h2>
                <p style="color: white; margin-top: 10px; font-size: 1.1em;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(conf / 100)

            st.markdown("---")
            st.subheader("💡 Help Us Improve")

            if not st.session_state.get('feedback_given', False):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 Correct (AI ทายถูก)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Correct")
                        st.session_state['feedback_given'] = True
                        st.toast("ขอบคุณครับ! บันทึกข้อมูลแล้ว") 
                        time.sleep(1)
                        st.rerun()
                with c2:
                    if st.button("👎 Incorrect (AI ทายผิด)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Incorrect")
                        st.session_state['feedback_given'] = True
                        st.toast("ขอบคุณครับ! เราจะนำไปปรับปรุง") 
                        time.sleep(1)
                        st.rerun()
            else:
                st.info("✅ คุณได้ส่ง Feedback สำหรับข่าวนี้แล้ว")

    # 📜 ประวัติการตรวจสอบ
    elif menu == "📜 ประวัติการตรวจสอบ":
        st.title("📜 ประวัติการตรวจสอบข่าว")
        uid = st.session_state.get('user_id')
        
        if uid:
            history_data = db.get_user_history(uid)
            
            if history_data:
                df = pd.DataFrame(history_data)
                df.columns = [c.lower() for c in df.columns]
                
                search_term = st.text_input("🔍 ค้นหาหัวข้อข่าว", placeholder="พิมพ์คำค้นหา...")
                if search_term:
                    df = df[df['title'].str.contains(search_term, case=False, na=False)]

                if not df.empty:
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M')

                    display_mapping = {
                        'title': 'หัวข้อข่าว',
                        'result': 'ผลการวิเคราะห์',
                        'confidence': 'ความเชื่อมั่น (%)', 
                        'timestamp': 'วันที่-เวลา'
                    }
                    
                    valid_cols = [c for c in display_mapping.keys() if c in df.columns]
                    df_display = df[valid_cols].copy()
                    df_display.rename(columns=display_mapping, inplace=True)
                    df_display.index = list(range(1, len(df_display) + 1))

                    st.write(f"พบข้อมูล {len(df_display)} รายการ")
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.warning(f"❌ ไม่พบหัวข้อข่าวที่มีคำว่า '{search_term}'")
            else:
                st.info("ℹ️ คุณยังไม่มีประวัติการตรวจสอบข่าว")

    # 🔥 ข่าวที่เป็นกระแส (มุมมอง User)
    elif menu == "🔥 ข่าวที่เป็นกระแส":
        st.title("🔥 ข่าวที่เป็นกระแส (Trending News)")
        st.write("อัปเดตข่าวสารที่กำลังถูกพูดถึงในขณะนี้ และตรวจสอบข้อเท็จจริงโดยทีมงาน")
        st.markdown("---")

        df_news = db.get_all_trending()
        
        if df_news.empty:
            st.info("ℹ️ ขณะนี้ยังไม่มีข่าวที่เป็นกระแส")
        else:
            for index, row in df_news.iterrows():
                label_color = "gray"
                icon = "📰"
                if row['label'] == 'Fake':
                    label_color = "#FF4B4B"
                    icon = "🚨"
                elif row['label'] == 'Real':
                    label_color = "#56F066"
                    icon = "✅"
                elif row['label'] == 'Unverified':
                    label_color = "#FFC107"
                    icon = "⚠️"

                with st.container():
                    st.markdown(f"#### {icon} {row['headline']}")
                    st.markdown(f"**สถานะ:** <span style='color:{label_color}; font-weight:bold;'>{row['label']}</span>", unsafe_allow_html=True)
                    st.write(row['content'])
                    
                    # 🚨 แก้ไขตรงนี้ครับ เปลี่ยนจาก upload_at เป็น updated_at
                    upload_time = row.get('updated_at', '-') 
                    
                    if upload_time != '-':
                        try:
                            upload_time = str(upload_time).replace("T", " ")[:16]
                        except:
                            pass
                    st.caption(f"🕒 อัปเดตเมื่อ: {upload_time}")
                    st.markdown("---")

    # 👤 ข้อมูลส่วนตัว 
    elif menu == "👤 ข้อมูลส่วนตัว":
        st.title("👤 ข้อมูลส่วนตัว")
        st.info("หน้านี้อยู่ระหว่างการพัฒนา 🚧")
        if st.button("🚪 ออกจากระบบ (Logout)", type="primary"):
            st.session_state.clear()
            st.rerun()

    # ==========================================
    # ADMIN PAGES ROUTING (รวมการเช็คสิทธิ์ไว้ที่เดียว)
    # ==========================================
    elif menu in admin_menu_options.keys():
        
        # 1. เช็คสิทธิ์ Admin
        if st.session_state.get('role') != 'admin':
            st.error("⛔ Access Denied: หน้านี้สำหรับ Admin เท่านั้น")
            st.stop() # หยุดการทำงานหากไม่ใช่ Admin
            
        # 2. แยกการทำงานตามหน้าต่างๆ ของ Admin
        if menu == "📊 Dashboard": 
            st.title("📊 Admin Dashboard")
            st.caption("สรุปภาพรวมประสิทธิภาพของระบบ (Real-time Data)")
            
            stats = db.get_dashboard_kpi()
            
            if not stats: 
                stats = {"checks_today": 0, "active_users": 0, "accuracy": 0.0, "feedback_total": 0}

            st.markdown("### Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.container(border=True)
                st.metric(
                    label="🔍 Total Checks Today", 
                    value=f"{stats.get('checks_today', 0):,}", 
                    delta="วันนี้",
                    delta_color="off" 
                )
            
            with col2:
                st.container(border=True)
                st.metric(
                    label="👥 Active Users (24h)", 
                    value=f"{stats.get('active_users', 0):,}", 
                    help="จำนวนผู้ใช้งานที่ไม่ซ้ำกันใน 24 ชม. ล่าสุด"
                )

            with col3:
                st.container(border=True)
                val_acc = stats.get('accuracy', 0.0)
                
                # เปลี่ยนสี Delta ถ้าต่ำกว่า 50%
                if val_acc < 50:
                    delta_text = "⚠️ ต่ำกว่าเกณฑ์"
                    d_color = "inverse" # สีแดง
                else:
                    delta_text = "✅ ปกติ"
                    d_color = "normal"  # สีเขียว
                    
                st.metric(
                    label="🎯 Model Accuracy", 
                    value=f"{val_acc}%", 
                    delta=delta_text,
                    delta_color=d_color
                )

            with col4:
                st.container(border=True)
                st.metric(
                    label="💬 Feedback Total", 
                    value=f"{stats.get('feedback_total', 0):,}",
                    help="จำนวนครั้งที่ผู้ใช้กด Correct/Incorrect"
                )

            st.markdown("---")
            
            st.subheader("⏱️ Recent Activity")
            logs = db.get_system_logs(limit=10) 
            
            if logs:
                for row in logs:
                    ts, user, action, details, level = row
                    with st.container(border=True):
                        c_icon, c_info, c_time = st.columns([0.5, 6, 2])
                        with c_icon:
                            if "admin" in str(user).lower():
                                st.write("🛡️")
                            else:
                                st.write("👤")
                        with c_info:
                            st.markdown(f"**{user}**")
                            short_detail = (details[:75] + '..') if len(details) > 75 else details
                            st.caption(f"{action} • {short_detail}")
                        with c_time:
                            st.caption(time_ago(ts))
            else:
                st.info("ยังไม่มีประวัติการใช้งานในระบบ")

        elif menu == "📈 Model Performance":
            st.title("📈 Model Performance")
            st.caption("ดูรายละเอียดความแม่นยำและการทำนายของ AI Model")
            show_model_performance() 

        elif menu == "💬 Review Feedback":
            show_feedback_review()
            
        elif menu == "📰 Manage News":
            manage_trending_news()
        
        elif menu == "🔬 System Analytics":
            st.title("🔬 System Analytics")
            show_system_analytics()
        
        elif menu == "👥 Manage Users":
            st.title("👥 Manage Users")
            manage_users_page()


