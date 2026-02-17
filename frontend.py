import streamlit as st
import pandas as pd
import time
import plotly.express as px
from supabase_auth import datetime

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
from datetime import datetime

def time_ago(timestamp_str):
    """แปลง Timestamp เป็นคำว่า 'X mins ago'"""
    try:
        if isinstance(timestamp_str, str):
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S") # ปรับ format ให้ตรงกับ DB
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
    # 1. ตั้งค่าเริ่มต้นเมนู (ถ้ายังไม่มี)
    if 'active_menu' not in st.session_state:
        st.session_state.active_menu = "🏠 หน้าหลัก"

    # --- SIDEBAR ---
    with st.sidebar:
        # ส่วนแสดงโปรไฟล์
        st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=80)
        st.markdown(f"### Hello, {st.session_state.get('username', 'User')}")
        st.caption(f"Role: {st.session_state.get('role', 'user').upper()}")
        st.divider()

        # -------------------------------------------------------
        # ส่วนที่ 1: เมนูหลัก (General Menu)
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # ส่วนที่ 2: Admin Panel - เห็นเฉพาะ Role = admin
        # -------------------------------------------------------
        if st.session_state.get('role') == 'admin':
            st.divider()
            st.markdown("### 🛠️ Admin Panel")
            st.caption("Manage system settings")
            
            # ✅ เพิ่มเมนูให้ครบตามรูปภาพ
            admin_menu_options = {
                "📊 Dashboard": "dashboard",
                "📈 Model Performance": "model_performance",
                "📰 Manage News": "manage_news",
                "💬 Review Feedback": "review_feedback",
                "🔬 System Analytics": "analytics",
                "👥 Manage Users": "manage_users"
            }
            
            for label, key_suffix in admin_menu_options.items():
                # ตรวจสอบสถานะว่าปุ่มไหนถูกเลือกอยู่ (Active)
                is_active = st.session_state.get('active_menu') == label
                
                if st.button(
                    label, 
                    key=f"admin_nav_{key_suffix}", 
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.active_menu = label
                    st.rerun()

    # ดึงค่าเมนูที่เลือกมาใช้งานต่อในส่วน Main Content
    menu = st.session_state.active_menu

    # =========================================================
    #  PAGE: หน้าหลัก (Home)
    # =========================================================
    if menu == "🏠 หน้าหลัก":
        st.title("🔍 ตรวจสอบความน่าเชื่อถือของข่าว")

        # Demo Buttons
        with st.expander("📝 ลองใช้ข่าวตัวอย่าง (Demo News)", expanded=False):
            col_mock1, col_mock2 = st.columns(2)
            if col_mock1.button("👽 ข่าว Aliens (Fake)"):
                st.session_state['input_text'] = "ข่าวล่าสุด: มนุษย์ต่างดาวลงจอดที่กรุงเทพฯ ใกล้กับสยามพารากอน! พยานระบุว่าพวกมันมีสีเขียวและเป็นมิตร"
            if col_mock2.button("🏛️ ข่าวรัฐบาล (Real)"):
                st.session_state['input_text'] = "รัฐบาลประกาศวันหยุดพิเศษเพิ่มอีก 1 วัน เพื่อกระตุ้นเศรษฐกิจและการท่องเที่ยวในช่วงเทศกาล"

        # Text Input
        input_val = st.session_state.get('input_text', "")
        input_text = st.text_area("วางเนื้อหาข่าวที่นี่:", value=input_val, height=200)
        
        # Action Button
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
                            time.sleep(0.5) # ลดเวลาลงนิดหน่อยให้ไวขึ้น
                            
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

        # --- ส่วนแสดงผลผลลัพธ์ (Result Display) ---
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

            # Feedback Section
            st.markdown("---")
            st.subheader("💡 Help Us Improve")

            if not st.session_state.get('feedback_given', False):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 Correct (แม่นยำ)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Correct")
                        st.session_state['feedback_given'] = True
                        st.toast("ขอบคุณครับ! บันทึกข้อมูลแล้ว") # ใช้ toast แทน success
                        time.sleep(1)
                        st.rerun()
                with c2:
                    if st.button("👎 Incorrect (ผิดพลาด)"):
                        db.save_feedback(st.session_state['current_pred_id'], "Incorrect")
                        st.session_state['feedback_given'] = True
                        st.toast("ขอบคุณครับ! เราจะนำไปปรับปรุง") # ใช้ toast แทน error
                        time.sleep(1)
                        st.rerun()
            else:
                st.info("✅ คุณได้ส่ง Feedback สำหรับข่าวนี้แล้ว")

    # =========================================================
    #  PAGE: ประวัติการตรวจสอบ (History)
    # =========================================================
    elif menu == "📜 ประวัติการตรวจสอบ":
        st.title("📜 ประวัติการตรวจสอบข่าว")
        uid = st.session_state.get('user_id')
        
        if uid:
            history_data = db.get_user_history(uid)
            
            if history_data:
                df = pd.DataFrame(history_data)
                df.columns = [c.lower() for c in df.columns]
                
                # Filter
                search_term = st.text_input("🔍 ค้นหาหัวข้อข่าว", placeholder="พิมพ์คำค้นหา...")
                if search_term:
                    df = df[df['title'].str.contains(search_term, case=False, na=False)]

                if not df.empty:
                    # Prep Data
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M')

                    display_mapping = {
                        'title': 'หัวข้อข่าว',
                        'result': 'ผลการวิเคราะห์',
                        'confidence': 'ความเชื่อมั่น (%)', 
                        'timestamp': 'วันที่-เวลา'
                    }
                    
                    # เลือกเฉพาะคอลัมน์ที่มีอยู่จริงเพื่อป้องกัน KeyError
                    valid_cols = [c for c in display_mapping.keys() if c in df.columns]
                    df_display = df[valid_cols].copy()
                    df_display.rename(columns=display_mapping, inplace=True)
                    
                    # ✅ แก้ไขตรงนี้: แปลง range เป็น list
                    df_display.index = list(range(1, len(df_display) + 1))

                    st.write(f"พบข้อมูล {len(df_display)} รายการ")
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.warning(f"❌ ไม่พบหัวข้อข่าวที่มีคำว่า '{search_term}'")
            else:
                st.info("ℹ️ คุณยังไม่มีประวัติการตรวจสอบข่าว")
                
    # =========================================================
    #  PAGE: Admin Dashboard
    # =========================================================
    elif menu == "📊 Dashboard": 
        if st.session_state.get('role') != 'admin':
            st.error("⛔ Access Denied: หน้านี้สำหรับ Admin เท่านั้น")
        else:
            st.title("📊 Admin Dashboard")
            st.caption("สรุปภาพรวมประสิทธิภาพของระบบ (Real-time Data)")
            
            # --- 1. ดึงข้อมูลจริงจาก Database ---
            # เรียกใช้ฟังก์ชันที่เราเพิ่งสร้าง
            stats = db.get_dashboard_kpi()
            
            # (ป้องกัน Error กรณี stats เป็น None แม้จะกันไว้ใน db แล้วก็ตาม)
            if not stats: 
                stats = {"checks_today": 0, "active_users": 0, "accuracy": 0.0, "feedback_total": 0}

            # --- 2. แสดงผล KPI Cards ---
            st.markdown("### Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.container(border=True)
                st.metric(
                    label="🔍 Total Checks Today", 
                    value=f"{stats['checks_today']:,}", 
                    delta="วันนี้",
                    delta_color="off" 
                )
            
            with col2:
                st.container(border=True)
                st.metric(
                    label="👥 Active Users (24h)", 
                    value=f"{stats['active_users']:,}", 
                    help="จำนวนผู้ใช้งานที่ไม่ซ้ำกันใน 24 ชม. ล่าสุด"
                )

            with col3:
                st.container(border=True)
                val_acc = stats['accuracy']
                # เปลี่ยนสี Delta ถ้าต่ำกว่า 50%
                st.metric(
                    label="🎯 Model Accuracy", 
                    value=f"{val_acc}%", 
                    delta="จาก Feedback ผู้ใช้"
                )

            with col4:
                st.container(border=True)
                st.metric(
                    label="💬 Feedback Total", 
                    value=f"{stats['feedback_total']:,}",
                    help="จำนวนครั้งที่ผู้ใช้กด Correct/Incorrect"
                )

            st.markdown("---")
            
            # --- 3. Recent Activity (ส่วนเดิมที่คุณมี) ---
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
                            # ตัดข้อความถ้ายาวเกินไป
                            short_detail = (details[:75] + '..') if len(details) > 75 else details
                            st.caption(f"{action} • {short_detail}")
                        with c_time:
                            st.caption(time_ago(ts))
            else:
                st.info("ยังไม่มีประวัติการใช้งานในระบบ")

    def show_model_performance():
            st.title("📈 Model Performance")
            st.caption("Monitor and analyze AI model metrics")
    
    # ✅ แก้ไขที่ 1: เรียกผ่าน db. และต้องย่อหน้าให้ตรงกันเพื่อให้โค้ดอยู่ภายในฟังก์ชัน
            df = db.get_model_performance_data() 
    
            if df.empty:
                st.warning("ไม่พบข้อมูลสำหรับการคำนวณ Performance")
                return # ตอนนี้ return จะทำงานได้ปกติเพราะอยู่ในฟังก์ชันแล้ว

    # --- ส่วนการคำนวณ (Logic) ---
    # (ตรวจสอบให้แน่ใจว่าโค้ดส่วนนี้ย่อหน้าเข้ามาทั้งหมด)
            y_true = df['label'].astype(str).str.lower()
            y_pred = df['prediction'].astype(str).str.lower()
    
            total = len(df)
            correct = (y_true == y_pred).sum()
    
            accuracy = correct / total if total > 0 else 0
    
            # คำนวณ Precision/Recall
            tp = ((y_true == 'fake') & (y_pred == 'fake')).sum()
            fp = ((y_true == 'real') & (y_pred == 'fake')).sum()
            fn = ((y_true == 'fake') & (y_pred == 'real')).sum()
    
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- 2. แสดงผลการ์ด Metric ---
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Accuracy", f"{accuracy*100:.1f}%")
                st.progress(accuracy)
            with col2:
                st.metric("Precision", f"{precision*100:.1f}%")
                st.progress(precision)
            with col3:
                st.metric("Recall", f"{recall*100:.1f}%")
                st.progress(recall)
            with col4:
                st.metric("F1 Score", f"{f1*100:.1f}%")
                st.progress(f1)

            # --- 3. กราฟแนวโน้ม ---
            st.write("---")
            st.subheader("Performance Trends")
            df['date'] = pd.to_datetime(df['created_at']).dt.date
            # ใช้กรุ๊ปข้อมูลเพื่อสร้างกราฟ
            trend_df = df.groupby('date').apply(lambda x: (x['label'] == x['prediction']).mean()).reset_index()
            trend_df.columns = ['Date', 'Accuracy']
            st.line_chart(trend_df.set_index('Date'))

            # --- 4. กราฟ Confidence Score ---
            st.write("---")
            st.subheader("Confidence Score Distribution")
            conf_scores = df['confidence_score'].dropna()
            bins = [0.6, 0.7, 0.8, 0.9, 1.0]
            labels = ['60-70%', '70-80%', '80-90%', '90-100%']
            dist_counts = pd.cut(conf_scores, bins=bins, labels=labels).value_counts().reindex(labels[::-1])
            st.bar_chart(dist_counts)