import streamlit as st
import pandas as pd
import time
import plotly.express as px

# --- IMPORT MODULES ของเราเอง ---
import database_ops as db
import ai_engine as ai
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
if 'register_mode' not in st.session_state: # ✅ เพิ่มโหมดสมัครสมาชิก
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
        
        # ขั้นที่ 1: กรอกอีเมล
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

        # ขั้นที่ 2: กรอก OTP
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
    with col_reg[1]: # จัดกึ่งกลาง
        new_user = st.text_input("Username ตั้งใหม่", key="reg_user")
        new_email = st.text_input("Email", key="reg_email")
        new_pw = st.text_input("Password ตั้งใหม่", type="password", key="reg_pw")
        confirm_pw = st.text_input("ยืนยัน Password", type="password", key="reg_pw_con")
        
        st.write("") # เว้นบรรทัด
        
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
    
    # จัด Layout ให้ฟอร์มอยู่ตรงกลาง (ซ้ายว่าง, กลางฟอร์ม, ขวาว่าง)
    col_layout = st.columns([1, 1.5, 1])
    
    with col_layout[1]:
        # --- 1. ส่วนโลโก้ (ชิดขวาบน เหนือ User) ---
        col_header = st.columns([2, 1]) # แบ่งพื้นที่เป็น 2 ส่วน (ซ้ายชื่อเว็บ, ขวารูป)
        with col_header[0]:
            st.markdown("## 🧠 AI Fake News")
            st.caption("เข้าสู่ระบบเพื่อใช้งาน")
        with col_header[1]:
            # ใส่ URL รูปโลโก้ของคุณที่นี่
            st.image("https://cdn-icons-png.flaticon.com/512/3021/3021707.png", width=100)
        
        st.markdown("---")
        
        # --- 2. ฟอร์ม Login ---
        user = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        
        if st.button("🚀 เข้าสู่ระบบ", type="primary", use_container_width=True):
            user_data = db.authenticate_user(user, pw)
            
            if user_data:
                # 1. ตั้งค่า Session State ก่อน
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user_data[0]
                st.session_state['username'] = user_data[1]
                st.session_state['role'] = user_data[2]
                
                # 2. 🟢 บันทึก Log เมื่อ Login สำเร็จ (เพิ่มตรงนี้!)
                db.log_system_event(
                    user_id=user_data[0],
                    action="USER_LOGIN",
                    details=f"User '{user_data[1]}' logged in successfully",
                    level="INFO"
                )
                
                st.success(f"ยินดีต้อนรับ {user}!")
                time.sleep(0.5)
                st.rerun()  # ย้ายมาไว้หลังสุดเพื่อให้ Log ทำงานเสร็จก่อน
                
            else:
                # 3. 🔴 บันทึก Log เมื่อ Login พลาด
                db.log_system_event(
                    user_id=None,
                    action="LOGIN_FAILED",
                    details=f"Failed login attempt for username: {user}",
                    level="WARNING"
                )
                st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
        
        # --- 3. ปุ่ม ลืมรหัส & สมัครสมาชิก (อยู่บรรทัดเดียวกัน ห่างกัน) ---
        col_actions = st.columns([1, 0.1, 1]) # [ปุ่มซ้าย, ช่องว่าง, ปุ่มขวา]
        
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
    # 1. จัดการเลือกเมนูด้วย Session State (ย้ายมาไว้ก่อน Sidebar เพื่อความชัวร์)
    menu_options = [
        "🏠 หน้าหลัก", 
        "📜 ประวัติการตรวจสอบ", 
        "🔥 ข่าวที่เป็นกระแส", 
        "📊 Admin Dashboard", 
        "👤 ข้อมูลส่วนตัว"
    ]

    if 'active_menu' not in st.session_state:
        st.session_state.active_menu = "🏠 หน้าหลัก"

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=80)
        st.markdown(f"### Hello, {st.session_state['username']}")
        st.caption(f"Role: {st.session_state['role'].upper()}")
        st.divider()
        
        st.write("### เมนูหลัก")
        for option in menu_options:
            # ตรวจสอบว่าเป็นเมนูที่เลือกอยู่หรือไม่
            is_active = st.session_state.active_menu == option
            
            # ใช้สไตล์แยกสำหรับปุ่มที่ active (ถ้าเลือกอยู่ให้เปลี่ยนสีพื้นหลัง)
            button_style = "primary" if is_active else "secondary"
            
            if st.button(option, key=f"btn_{option}", 
                         use_container_width=True,
                         type=button_style): # ใช้ type ช่วยคุมสีเบื้องต้น
                st.session_state.active_menu = option
                st.rerun()
# 3. ใช้ค่าจาก Session State มาคุม Logic หน้าเว็บ
    menu = st.session_state.active_menu


    # --- ส่วนเนื้อหาหลัก (เหมือนเดิม) ---
    if menu == "🏠 หน้าหลัก":
        st.title("🔍 ตรวจสอบความน่าเชื่อถือของข่าว")

        with st.expander("📝 ลองใช้ข่าวตัวอย่าง (Demo News)", expanded=False):
            col_mock1, col_mock2 = st.columns(2)
            if col_mock1.button("👽 ข่าว Aliens (Fake)"):
                st.session_state['input_text'] = "ข่าวล่าสุด: มนุษย์ต่างดาวลงจอดที่กรุงเทพฯ ใกล้กับสยามพารากอน! พยานระบุว่าพวกมันมีสีเขียวและเป็นมิตร"
            if col_mock2.button("🏛️ ข่าวรัฐบาล (Real)"):
                st.session_state['input_text'] = "รัฐบาลประกาศวันหยุดพิเศษเพิ่มอีก 1 วัน เพื่อกระตุ้นเศรษฐกิจและการท่องเที่ยวในช่วงเทศกาล"

        input_val = st.session_state.get('input_text', "")
        input_text = st.text_area("วางเนื้อหาข่าวที่นี่:", value=input_val, height=200)
        
        # --- ส่วนตรวจสอบข่าว ---
        if st.button("🚀 Analyze News", type="primary", use_container_width=True):
           
            clean_text = str(input_text).strip() if input_text else ""
            
            if not clean_text:
                st.warning("กรุณาใส่เนื้อหาข่าว")
            else:
                with st.spinner("AI กำลังวิเคราะห์..."):
                    try:
                        # 2. เรียก AI วิเคราะห์
                        result = ai.predict_news(clean_text)
                        
                        if result is not None:
                            time.sleep(0.8)
                            
                            res_label = result.get('result', 'Error')
                            res_conf = result.get('confidence', 0.0)
                            
                            # บันทึก Log
                            log_msg = f"Analyzed: {clean_text[:30]}... Result: {res_label}"
                            db.log_system_event(
                                st.session_state.get('user_id'), 
                                "AI_PREDICT", 
                                log_msg, 
                                "INFO" if res_label != 'Error' else "ERROR"
                            )
                            
                            # บันทึก Prediction ลง Database
                            pred_id = db.create_prediction(
                                st.session_state.get('user_id'), 
                                clean_text[:50]+"...", 
                                clean_text, 
                                None, 
                                res_label, 
                                res_conf
                            )
                            
                            # อัปเดต State
                            st.session_state['current_result'] = result
                            st.session_state['current_pred_id'] = pred_id
                            st.session_state['feedback_given'] = False
                            st.rerun()
                        else:
                            st.error("AI ไม่ตอบสนอง (None Result)")
                            
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {str(e)}")

        # --- ส่วนแสดงผล (อยู่นอกเงื่อนไขปุ่มกด) ---
        if 'current_result' in st.session_state:
            # ดำเนินการแสดง UI ผลลัพธ์ต่อไป...
            pass
        if 'current_result' in st.session_state:
            res = st.session_state['current_result']
            label = res['result']
            conf = res['confidence']
            
            if label == "Fake":
                bg_color = "#381E1E"
                border_color = "#FF4B4B"
                text_color = "#FF4B4B"
                icon = "🚨"
                desc = "เนื้อหานี้มีลักษณะเป็น ข่าวปลอม หรือ บิดเบือน"
            else:
                bg_color = "#1E3822"
                border_color = "#56F066"
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

            st.markdown("---")
            st.subheader("💡 Help Us Improve")

            if not st.session_state.get('feedback_given', False):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 Correct (แม่นยำ)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Correct")
                        st.session_state['feedback_given'] = True
                        st.success("ขอบคุณครับ! บันทึกข้อมูลแล้ว")
                        time.sleep(1)
                        st.rerun()
                with c2:
                    if st.button("👎 Incorrect (ผิดพลาด)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Incorrect")
                        st.session_state['feedback_given'] = True
                        st.error("ขอบคุณครับ! เราจะนำไปปรับปรุง")
                        time.sleep(1)
                        st.rerun()
            else:
                st.info("✅ คุณได้ส่ง Feedback สำหรับข่าวนี้แล้ว")

    elif menu == "📜 ประวัติการตรวจสอบ":
        st.title("📜 ประวัติการตรวจสอบข่าว")
        uid = st.session_state.get('user_id')
        
        if uid:
            history_data = db.get_user_history(uid)
            
            if history_data:
                # 1. สร้าง DataFrame และเตรียมข้อมูล
                df = pd.DataFrame(history_data)
                df.columns = [c.lower() for c in df.columns]
                
                # --- ส่วนของ FILTER (เพิ่มใหม่) ---
                search_term = st.text_input("🔍 ค้นหาหัวข้อข่าวที่คุณเคยตรวจสอบ", placeholder="พิมพ์คำที่ต้องการค้นหาที่นี่...")
                
                # กรองข้อมูลตามคำค้นหา (ถ้ามีการพิมพ์คำค้นหา)
                if search_term:
                    # ค้นหาคำในคอลัมน์ title โดยไม่สนตัวพิมพ์เล็ก-ใหญ่ (case=False)
                    df = df[df['title'].str.contains(search_term, case=False, na=False)]
                # ------------------------------

                if not df.empty:
                    # 2. จัดการข้อมูลก่อนแสดงผล (เหมือนเดิม)
                    df['content_display'] = df['title'].astype(str)
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M')

                    # 3. เลือกและเปลี่ยนชื่อคอลัมน์ให้เป็นภาษาไทย
                    display_mapping = {
                        'content_display': 'หัวข้อข่าว',
                        'result': 'ผลการวิเคราะห์',
                        'confidence': 'ความเชื่อมั่น (%)', 
                        'timestamp': 'วันที่-เวลา'
                    }
                    
                    present_cols = [c for c in display_mapping.keys() if c in df.columns]
                    df_display = df[present_cols].copy()
                    df_display.rename(columns=display_mapping, inplace=True)
                    
                    # รีเซ็ตเลข Index ให้เริ่มที่ 1
                    df_display.index = list(range(1, len(df_display) + 1))

                    st.write(f"พบข้อมูลทั้งหมด {len(df_display)} รายการ")
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.warning(f"❌ ไม่พบหัวข้อข่าวที่มีคำว่า '{search_term}'")
                
            else:
                st.info("ℹ️ คุณยังไม่มีประวัติการตรวจสอบข่าวในระบบ")


    elif menu == "📊 Admin Dashboard":
        if st.session_state.get('role') != 'admin':
            st.error("⛔ Access Denied")
        else:
            st.title("🛠 Admin Control Panel")
            
            # ตัวเลือกโหมดหน้า Admin
            adm_mode = st.radio(
                "เลือกโหมด:", 
                ["Overview Stats", "Accuracy Review", "System Logs"], 
                horizontal=True
            )

            # --- โหมด 1: Overview ---
            if adm_mode == "Overview Stats":
                all_preds = db.read_all_predictions()
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

            # --- โหมด 2: Accuracy Review ---
            elif adm_mode == "Accuracy Review":
                st.markdown("### 🎯 ตรวจสอบความถูกต้อง")
                recent_preds = db.read_all_predictions_limit(10)
                
                if recent_preds:
                    for item in recent_preds:
                        with st.expander(f"[{item[4]}] {item[2]} ({item[5]}%)"):
                            st.write(f"**News:** {item[3]}")
                            st.write(f"**AI Predicted:** {item[4]}")
                            st.caption(f"User: {item[1]} | Time: {item[6]}")
                            
                            c1, c2 = st.columns(2)
                            if c1.button("Mark as REAL", key=f"real_{item[0]}"):
                                st.toast(f"Updated ID {item[0]} -> Real")
                            if c2.button("Mark as FAKE", key=f"fake_{item[0]}"):
                                st.toast(f"Updated ID {item[0]} -> Fake")
                else:
                    st.info("ไม่มีข้อมูลการวิเคราะห์ล่าสุด")

            # --- โหมด 3: System Logs (ตรวจสอบว่าเยื้องเข้ามาตรงนี้!) ---
            elif adm_mode == "System Logs":
                st.subheader("🤖 บันทึกการทำงานของระบบ (System Logs)")
                
                if st.button("🔄 Refresh Logs"):
                    st.rerun()
                
                # 1. ดึงข้อมูล
                logs_data = db.get_system_logs(100) 
                
                # 2. ตรวจสอบว่ามีข้อมูลหรือไม่
                if logs_data:
                    # สร้าง df_log ภายในเงื่อนไขที่มีข้อมูลเท่านั้น
                    df_log = pd.DataFrame(logs_data, columns=['เวลา', 'idผู้ใช้', 'กิจกรรม', 'รายละเอียด', 'ระดับ'])
                    
                    # 3. ฟังก์ชันใส่สี (นิยามไว้ใช้เฉพาะตอนมีข้อมูล)
                    def style_rows(row):
                        if row['ระดับ'] == 'ERROR':
                            return ['background-color: #ff4b4b; color: white; font-weight: bold'] * len(row)
                        elif row['ระดับ'] == 'INFO':
                            return ['background-color: #f0fdf4; color: #166534'] * len(row)
                        return [''] * len(row)

                    # 4. แสดงผลตาราง (เรียกใช้ df_log ตรงนี้)
                    st.dataframe(
                        df_log.style.apply(style_rows, axis=1), 
                        use_container_width=True, 
                        height=500,
                        hide_index=True
                    )
                else:
                    # ถ้าไม่มีข้อมูล ให้แสดงข้อความแจ้งเตือนแทนการเรียกใช้ df_log
                    st.info("ℹ️ ยังไม่มีข้อมูลบันทึกในระบบ")