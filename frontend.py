import streamlit as st
import pandas as pd
import time
import plotly.express as px

# --- IMPORT MODULES ของเราเอง ---
import database_ops as db
import ai_engine as ai

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

# ==========================================
# 2. ส่วน Authentication (Login/Register)
# ==========================================
if not st.session_state['logged_in']:
    st.title("🧠 Fake News Detection System")
    st.markdown("### ระบบตรวจสอบข่าวปลอมด้วย AI (WangchanBERTa + GNN)")
    
    col_img, col_auth = st.columns([1, 2])
    
    with col_img:
        st.image("https://cdn-icons-png.flaticon.com/512/3021/3021707.png", width=200)

    with col_auth:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

        with tab1: # LOGIN
            user = st.text_input("Username", key="login_user")
            pw = st.text_input("Password", type="password", key="login_pw")
            
            if st.button("เข้าสู่ระบบ", type="primary", use_container_width=True):
                user_data = db.authenticate_user(user, pw)
                if user_data:
                    st.success(f"ยินดีต้อนรับ {user}!")
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = user_data[0]
                    st.session_state['username'] = user_data[1]
                    st.session_state['role'] = user_data[2]
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

        with tab2: # REGISTER
            new_user = st.text_input("Username ใหม่", key="reg_user")
            new_pw = st.text_input("Password ใหม่", type="password", key="reg_pw")
            new_email = st.text_input("Email", key="reg_email")
            
            if st.button("สมัครสมาชิก", use_container_width=True):
                if db.create_user(new_user, new_pw, new_email):
                    st.success("สมัครสมาชิกสำเร็จ! กรุณาไปที่หน้า Login")
                else:
                    st.error("ชื่อผู้ใช้นี้มีคนใช้แล้ว")

# ==========================================
# 3. ส่วนหลักหลัง Login
# ==========================================
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=80)
        st.markdown(f"### Hello, {st.session_state['username']}")
        st.caption(f"Role: {st.session_state['role'].upper()}")
        st.divider()
        
        menu = st.radio("เมนูหลัก", 
                        ["🔍 ตรวจสอบข่าว", "📜 ประวัติของฉัน", "📊 Admin Dashboard"])
        
        st.divider()
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    # ------------------------------------------------
    # หน้าที่ 1: ตรวจสอบข่าว (Check News)
    # ------------------------------------------------
    if menu == "🔍 ตรวจสอบข่าว":
        st.title("🔍 ตรวจสอบความน่าเชื่อถือของข่าว")

        # --- ส่วนตัวอย่างข่าว (Mock Data จาก GitHub แต่ใส่ตรงนี้เลย) ---
        with st.expander("📝 ลองใช้ข่าวตัวอย่าง (Demo News)", expanded=False):
            col_mock1, col_mock2 = st.columns(2)
            if col_mock1.button("👽 ข่าว Aliens (Fake)"):
                st.session_state['input_text'] = "Breaking News: Aliens have landed in Bangkok near Siam Paragon! Witnesses say they are green and friendly."
            if col_mock2.button("🏛️ ข่าวรัฐบาล (Real)"):
                st.session_state['input_text'] = "รัฐบาลประกาศวันหยุดพิเศษเพิ่มอีก 1 วัน เพื่อกระตุ้นเศรษฐกิจและการท่องเที่ยวในช่วงเทศกาล"

        # Text Area รับค่า (ดึงจาก session_state ถ้ามีการกดปุ่มตัวอย่าง)
        input_val = st.session_state.get('input_text', "")
        input_text = st.text_area("วางเนื้อหาข่าวที่นี่:", value=input_val, height=200)
        
        if st.button("🚀 Analyze News", type="primary", use_container_width=True):
            if not input_text:
                st.warning("กรุณาใส่เนื้อหาข่าวก่อนครับ")
            else:
                with st.spinner("AI กำลังวิเคราะห์... (WangchanBERTa + GNN)"):
                    # 1. เรียก AI
                    result = ai.predict_news(input_text)
                    time.sleep(0.8) # หน่วงให้ดูเหมือนคิดนิดนึง

                    # 2. บันทึกลง DB
                    pred_id = db.create_prediction(
                        st.session_state['user_id'], 
                        input_text[:50]+"...", input_text, None, 
                        result['result'], result['confidence']
                    )
                    
                    # 3. เก็บ State ไว้แสดงผล
                    st.session_state['current_result'] = result
                    st.session_state['current_pred_id'] = pred_id
                    st.session_state['feedback_given'] = False

        # --- ส่วนแสดงผล (Styling แบบ GitHub) ---
        if 'current_result' in st.session_state:
            res = st.session_state['current_result']
            label = res['result']
            conf = res['confidence']
            
            # กำหนดสีตามผลลัพธ์
            if label == "Fake":
                bg_color = "#381E1E" # แดงเข้ม (Dark Red theme)
                border_color = "#FF4B4B" # แดงสว่าง
                text_color = "#FF4B4B"
                icon = "🚨"
                desc = "เนื้อหานี้มีลักษณะเป็น ข่าวปลอม หรือ บิดเบือน"
            else:
                bg_color = "#1E3822" # เขียวเข้ม (Dark Green theme)
                border_color = "#56F066" # เขียวสว่าง
                text_color = "#56F066"
                icon = "✅"
                desc = "เนื้อหานี้ดูสมเหตุสมผลและ น่าเชื่อถือ"

            st.markdown(f"""
            <div style="
                padding: 25px;
                border-radius: 15px;
                background-color: {bg_color};
                border-left: 8px solid {border_color};
                margin-top: 20px;
                margin-bottom: 20px;">
                <h2 style="color: {text_color}; margin:0;">{icon} {label.upper()} ({conf:.1f}%)</h2>
                <p style="color: white; margin-top: 10px; font-size: 1.1em;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(conf / 100)

            # --- Feedback Section (เหมือน GitHub) ---
            st.markdown("---")
            st.subheader("💡 Help Us Improve")
            st.caption("AI ทำนายถูกต้องหรือไม่? ความเห็นของคุณจะช่วยเทรนให้มันฉลาดขึ้น")

            if not st.session_state.get('feedback_given', False):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 Correct (แม่นยำ)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Correct")
                        st.session_state['feedback_given'] = True
                        st.success("ขอบคุณครับ! บันทึกข้อมูลแล้ว")
                        st.rerun()
                with c2:
                    if st.button("👎 Incorrect (ผิดพลาด)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Incorrect")
                        st.session_state['feedback_given'] = True
                        st.error("ขอบคุณครับ! เราจะนำไปปรับปรุง")
                        st.rerun()
            else:
                st.info("✅ คุณได้ส่ง Feedback สำหรับข่าวนี้แล้ว")

    # ------------------------------------------------
    # หน้าที่ 2: ประวัติ (User History)
    # ------------------------------------------------
    elif menu == "📜 ประวัติของฉัน":
        st.header("📜 ประวัติการใช้งาน")
        history = db.read_user_history(st.session_state['user_id'])
        if history:
            df = pd.DataFrame(history, columns=['Date', 'Title', 'Result', 'Confidence'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("ไม่พบประวัติการใช้งาน")

    # ------------------------------------------------
    # หน้าที่ 3: Admin Dashboard (ฟีเจอร์จัดเต็ม)
    # ------------------------------------------------
    elif menu == "📊 Admin Dashboard":
        if st.session_state['role'] != 'admin':
            st.error("⛔ Access Denied: สำหรับผู้ดูแลระบบเท่านั้น")
        else:
            st.title("🛠 Admin Control Panel")
            
            adm_mode = st.radio("เลือกโหมด:", ["Overview Stats", "Accuracy Review", "Manage Users"], horizontal=True)

            # โหมด 1: ดูภาพรวม
            if adm_mode == "Overview Stats":
                # ดึงข้อมูลทั้งหมดมาคำนวณแบบ Pandas (เหมือน GitHub)
                all_preds = db.read_all_predictions() # ต้องไปเพิ่มฟังก์ชันนี้ใน db
                if all_preds:
                    df = pd.DataFrame(all_preds, columns=['ID', 'User', 'Title', 'Text', 'Result', 'Conf', 'Timestamp'])
                    
                    total = len(df)
                    fake_count = len(df[df['Result'] == 'Fake'])
                    real_count = len(df[df['Result'] == 'Real'])
                    avg_conf = df['Conf'].mean()

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Checks", total)
                    col2.metric("Fake Found", fake_count, delta_color="inverse")
                    col3.metric("Real Found", real_count)
                    col4.metric("Avg Confidence", f"{avg_conf:.1f}%")

                    st.subheader("📈 Usage Timeline")
                    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
                    usage_chart = df.groupby('Date').size()
                    st.line_chart(usage_chart)
                else:
                    st.info("ยังไม่มีข้อมูลในระบบ")

            # โหมด 2: ตรวจสอบความแม่นยำ (ฟีเจอร์เด็ดจาก GitHub)
            elif adm_mode == "Accuracy Review":
                st.markdown("### 🎯 ตรวจสอบความถูกต้อง (Manual Labeling)")
                st.caption("แอดมินสามารถช่วยระบุได้ว่า AI ทายถูกหรือผิด เพื่อนำไปเทรนต่อ")
                
                # ดึงรายการล่าสุด
                recent_preds = db.read_all_predictions_limit(10) # ดึง 10 อันล่าสุด
                
                for item in recent_preds:
                    # item = (id, username, title, text, result, conf, timestamp)
                    with st.expander(f"[{item[4]}] {item[2]} ({item[5]}%)"):
                        st.write(f"**News:** {item[3]}")
                        st.write(f"**AI Predicted:** {item[4]}")
                        st.caption(f"User: {item[1]} | Time: {item[6]}")
                        
                        # ปุ่มแก้ Label
                        c1, c2 = st.columns(2)
                        if c1.button("Mark as REAL", key=f"real_{item[0]}"):
                            # db.update_prediction_actual(item[0], 'Real') # (ต้องเพิ่มใน DB Ops)
                            st.toast(f"Updated ID {item[0]} -> Real")
                        if c2.button("Mark as FAKE", key=f"fake_{item[0]}"):
                            # db.update_prediction_actual(item[0], 'Fake')
                            st.toast(f"Updated ID {item[0]} -> Fake")

            # โหมด 3: จัดการ User
            elif adm_mode == "Manage Users":
                st.write("จัดการผู้ใช้งาน (Feature นี้เขียนเพิ่มได้ตามต้องการ)")