import streamlit as st
import pandas as pd
import time
import plotly.express as px 

# --- IMPORT MODULES ของเราเอง ---
# (ต้องมีไฟล์ database_ops.py และ ai_engine.py อยู่โฟลเดอร์เดียวกัน)
import database_ops as db
import ai_engine as ai

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Config)
# ==========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="wide"
)

# เริ่มต้น Session State (ตัวแปรจำค่าข้ามหน้า)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = ""
    st.session_state['role'] = ""

# ==========================================
# 2. ส่วน Authentication (Login/Register)
# ==========================================
if not st.session_state['logged_in']:
    st.title("🕵️ Fake News Detection System")
    st.info("ระบบตรวจสอบข่าวปลอมด้วย AI และ GNN")
    
    tab1, tab2 = st.tabs(["🔐 เข้าสู่ระบบ", "📝 สมัครสมาชิก"])

    with tab1: # LOGIN
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/2919/2919600.png", width=150)
        with col2:
            user = st.text_input("Username", key="login_user")
            pw = st.text_input("Password", type="password", key="login_pw")
            
            if st.button("เข้าสู่ระบบ", type="primary"):
                user_data = db.authenticate_user(user, pw)
                if user_data:
                    st.success(f"ยินดีต้อนรับ {user}!")
                    # เก็บค่าเข้าระบบ
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = user_data[0]
                    st.session_state['username'] = user_data[1]
                    st.session_state['role'] = user_data[2]
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

    with tab2: # REGISTER
        new_user = st.text_input("Username ใหม่", key="reg_user")
        new_pw = st.text_input("Password ใหม่", type="password", key="reg_pw")
        new_email = st.text_input("Email (Optional)", key="reg_email")
        
        if st.button("สมัครสมาชิก"):
            if new_user and new_pw:
                if db.create_user(new_user, new_pw, new_email):
                    st.success("สมัครสมาชิกสำเร็จ! กรุณาไปที่หน้า Login")
                else:
                    st.error("ชื่อผู้ใช้นี้มีคนใช้แล้ว")
            else:
                st.warning("กรุณากรอกข้อมูลให้ครบ")

# ==========================================
# 3. ส่วนหลักหลัง Login (Main Application)
# ==========================================
else:
    # Sidebar เมนู
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=100)
        st.title(f"สวัสดี, {st.session_state['username']}")
        st.caption(f"สถานะ: {st.session_state['role'].upper()}")
        st.divider()
        
        menu = st.radio("เมนูหลัก", 
                        ["🔍 ตรวจสอบข่าว", "📜 ประวัติของฉัน", "📊 Admin Dashboard"])
        
        st.divider()
        if st.button("ออกจากระบบ"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ------------------------------------------------
    # หน้าที่ 1: ตรวจสอบข่าว (Check News)
    # ------------------------------------------------
    if menu == "🔍 ตรวจสอบข่าว":
        st.header("🔍 ตรวจสอบความน่าเชื่อถือของข่าว")
        
        # --- ส่วนแสดง Trending News ---
        with st.expander("📢 ประกาศจากผู้ดูแลระบบ (Trending News)", expanded=True):
            
            # 🔥 เรียกใช้ฟังก์ชันเดิม
            news_items = db.get_all_trending()

            if news_items:
                # โชว์แค่ 3 ข่าวล่าสุดพอ (ใช้ [:3])
                for news in news_items[:3]:
                    icon = "🚨" if "Fake" in news[3] else "✅" if "Real" in news[3] else "⚠️"
                    st.markdown(f"**{icon} {news[1]}**") # headline
                    st.write(news[2]) # content
                    st.caption(f"เมื่อ: {news[4]}")
                    st.divider()
            else:
                st.write("ยังไม่มีประกาศใหม่")

        input_text = st.text_area("วางเนื้อหาข่าว หรือ พาดหัวข่าวที่นี่:", height=200, placeholder="ตัวอย่าง: รัฐบาลแจกเงินฟรี 5000 บาท กดลิงก์นี้เลย...")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            check_btn = st.button("🚀 วิเคราะห์ผล", type="primary", use_container_width=True)
        
        # --- Logic การตรวจสอบ ---
        if check_btn and input_text:
            with st.spinner("AI กำลังอ่านข่าวและประมวลผล..."):
                # 1. เรียก AI Engine (ทำงานในไฟล์นี้เลย ไม่ต้องยิง API)
                result = ai.predict_news(input_text)
                time.sleep(1) # หน่วงเวลาเท่ๆ นิดนึง

                # 2. บันทึกลง Database
                pred_id = db.create_prediction(
                    user_id=st.session_state['user_id'],
                    title=input_text[:50] + "...",
                    text=input_text,
                    url=None,
                    result=result['result'],
                    confidence=result['confidence']
                )
                
                # 3. เก็บ Log
                db.create_log(st.session_state['user_id'], "CHECK_NEWS")

                # 4. เก็บ Session ไว้แสดงผล Feedback
                st.session_state['last_pred_id'] = pred_id
                st.session_state['last_result'] = result
                st.session_state['feedback_submitted'] = False

        # --- ส่วนแสดงผลลัพธ์ (Show Result) ---
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            st.divider()
            
            # การ์ดแสดงผล
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                if res['result'] == 'Fake':
                    st.error(f"## 🚨 ผลการทำนาย: {res['result']}")
                    st.write("ข่าวนี้มีความเสี่ยงสูงที่จะเป็น **ข่าวปลอม**")
                else:
                    st.success(f"## ✅ ผลการทำนาย: {res['result']}")
                    st.write("ข่าวนี้มีแนวโน้มเป็น **ข่าวจริง**")
            
            with r_col2:
                st.metric("ความมั่นใจ (Confidence)", f"{res['confidence']:.2f}%")
                # แสดงเพื่อนบ้าน (Explainability)
                if 'neighbor_ids' in res:
                    st.caption(f"อ้างอิงจากข่าวเก่าที่คล้ายกัน {len(res['neighbor_ids'])} รายการ")

            # --- ส่วน Feedback (ปุ่มกดถูก/ผิด) ---
            st.write("---")
            if not st.session_state.get('feedback_submitted', False):
                st.write("📢 **ผลการทำนายนี้ถูกต้องหรือไม่?** (ช่วย AI เรียนรู้)")
                fb1, fb2 = st.columns([1,1])
                with fb1:
                    if st.button("👍 ถูกต้องแม่นยำ"):
                        db.save_feedback(st.session_state['last_pred_id'], "Correct")
                        st.session_state['feedback_submitted'] = True
                        st.toast("ขอบคุณสำหรับข้อมูลครับ!", icon="🙏")
                        st.rerun()
                with fb2:
                    if st.button("👎 ผิดพลาด"):
                        db.save_feedback(st.session_state['last_pred_id'], "Incorrect")
                        st.session_state['feedback_submitted'] = True
                        st.toast("เราจะนำไปปรับปรุงครับ!", icon="🔧")
                        st.rerun()
            else:
                st.info("✅ ขอบคุณที่คุณช่วยตรวจสอบความถูกต้องครับ")

    # ------------------------------------------------
    # หน้าที่ 2: ประวัติของฉัน (User History)
    # ------------------------------------------------
    elif menu == "📜 ประวัติของฉัน":
        st.header("📜 ประวัติการตรวจสอบข่าวของคุณ")
        history_data = db.read_user_history(st.session_state['user_id'])
        
        if history_data:
            df = pd.DataFrame(history_data, columns=['Date/Time', 'News Title', 'Result', 'Confidence'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("คุณยังไม่เคยตรวจสอบข่าวเลย ลองไปที่หน้าแรกสิ!")

    # ------------------------------------------------
    # หน้าที่ 3: Admin Dashboard (Updated)
    # ------------------------------------------------
    elif menu == "📊 Admin Dashboard":
        if st.session_state['role'] == 'admin':
            st.header("⚙️ Admin Management System")
            
            # แบ่งเป็น 2 Tabs: ดูสถิติ กับ จัดการข่าวประกาศ
            adm_tab1, adm_tab2 = st.tabs(["📊 สถิติระบบ (Analytics)", "📢 จัดการข่าวประกาศ (Trending)"])

            # --- TAB 1: สถิติ ---
            with adm_tab1:
                logs = db.read_logs_for_chart()
                feedbacks = db.read_all_feedbacks()
                
                # Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Usage", sum([x[1] for x in logs]) if logs else 0, "Checks")
                m2.metric("User Feedbacks", len(feedbacks), "Reports")
                m3.metric("System Status", "Active", "🟢")
                
                # Chart
                st.subheader("📈 User Activity")
                if logs:
                    df_logs = pd.DataFrame(logs, columns=['Hour', 'Count'])
                    fig = px.bar(df_logs, x='Hour', y='Count', title="กราฟแสดงช่วงเวลาการใช้งาน")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feedback Table
                st.subheader("💬 User Feedback Reports")
                if feedbacks:
                    df_fb = pd.DataFrame(feedbacks, columns=['ID', 'News Title', 'User Report', 'Comment', 'Status'])
                    st.dataframe(df_fb, use_container_width=True)

            # --- TAB 2: จัดการข่าวประกาศ (Trending) ---
            with adm_tab2:
                st.subheader("📢 สร้างประกาศ / ข่าวเตือนภัย")
                
                with st.form("add_trending_form"):
                    t_head = st.text_input("พาดหัวข่าว (Headline)")
                    t_content = st.text_area("เนื้อหาข่าว / รายละเอียด")
                    t_label = st.selectbox("ประเภท", ["ข่าวปลอม (Fake)", "ข่าวจริง (Real)", "ข้อควรระวัง (Warning)"])
                    
                    submitted = st.form_submit_button("ประกาศข่าว")
                    if submitted and t_head:
                        # เรียกฟังก์ชันจาก database_ops (ต้องมีฟังก์ชัน create_trending)
                        db.create_trending(t_head, t_content, t_label)
                        st.success("บันทึกประกาศเรียบร้อยแล้ว!")
                        time.sleep(1)
                        st.rerun()

                st.divider()
                st.subheader("รายการประกาศปัจจุบัน")
                # 🔥 เปลี่ยนจาก SQL ยาวๆ เป็นเรียกบรรทัดเดียวจบ
            trends = db.get_all_trending() 

            if trends:
                for item in trends:
                    # item[0]=id, [1]=headline, [2]=content, [3]=label, [4]=time
                    with st.expander(f"📢 {item[1]} ({item[3]})"):
                        st.write(item[2])
                        st.caption(f"อัปเดตเมื่อ: {item[4]}")
                        
                        col_edit, col_del = st.columns([1, 1])
                        with col_del:
                            if st.button("ลบประกาศ", key=f"del_{item[0]}", type="primary"):
                                db.delete_trending(item[0])
                                st.rerun()
            else:
                st.info("ยังไม่มีประกาศในระบบ")

        else:
            st.error("⛔ Access Denied")