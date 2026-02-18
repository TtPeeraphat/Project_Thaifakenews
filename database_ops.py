import hashlib
from datetime import datetime, timedelta, timezone
import smtplib
from email.mime.text import MIMEText
import random
import string
from supabase import create_client, Client
from typing import List, Dict, Any, Optional, Tuple, Union, cast
import pandas as pd
import psycopg2
import streamlit as st
from typing import List, Dict, Any  
import supabase
from postgrest.types import CountMethod
# ==========================================
# 🔑 ใส่ค่า CONFIG ของคุณที่นี่
# ==========================================
SUPABASE_URL = "https://orxtfxdernqmpkfmsijj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9yeHRmeGRlcm5xbXBrZm1zaWpqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzEyMDQ5OTgsImV4cCI6MjA4Njc4MDk5OH0.6dDVQio5hQpTQj6jnnS6yZBqR2GBReqFwazza6TqolQ"
SENDER_EMAIL = "nantwtf00@gmail.com"
SENDER_PASSWORD = "aiga bqgc jbrl rltl"

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 1. Users
# ==========================================

def create_user(username, password, email, role='user') -> bool:
    supabase = get_supabase()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    
    data_payload = {
        "username": username,
        "password_hash": pw_hash,
        "email": email,
        "role": role,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        response = supabase.table("users").insert(data_payload).execute()
        # เช็คว่าเป็น List และไม่ว่าง
        if response.data is not None and isinstance(response.data, list) and len(response.data) > 0:
            return True
        return False
    except Exception as e:
        print(f"Register Error: {e}")
        return False

def authenticate_user(username, password) -> Optional[Tuple[int, str, str]]:
    supabase = get_supabase()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        response = supabase.table("users").select("id, username, role")\
            .eq("username", username)\
            .eq("password_hash", pw_hash)\
            .execute()
            
        data = response.data
        
        if data is not None and isinstance(data, list) and len(data) > 0:
            user_data = data[0]
            if isinstance(user_data, dict):
                # ✅ FIX: แปลงเป็น str ก่อน แล้วค่อย int เพื่อหลอก Pylance ว่าไม่ใช่ Dict
                uid = int(str(user_data.get('id', 0)))
                uname = str(user_data.get('username', ''))
                urole = str(user_data.get('role', ''))
                
                return (uid, uname, urole)
        
        return None
    except Exception as e:
        print(f"Auth Error: {e}")
        return None

# ==========================================
# 2. Predictions
# ==========================================

def create_prediction(user_id, title, text, url, result, confidence) -> Optional[int]:
    supabase = get_supabase()
    payload = {
        "user_id": user_id,
        "title": title,
        "text": text,
        "url": url,
        "result": result,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = supabase.table("predictions").insert(payload).execute()
        data = response.data
        
        if data is not None and isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict):
                # ✅ FIX: แปลงเป็น str ก่อน cast
                return int(str(item.get('id', 0)))
        return None
    except Exception as e:
        print(f"Create Prediction Error: {e}")
        return None

from typing import Optional

def get_user_history(user_id: Any, limit: int = 50):
    supabase = get_supabase()
    try:
        # 1. ตรวจสอบว่ามี user_id มาจริงไหม
        if user_id is None:
            return []
            
        # 2. Query ข้อมูล (ใช้ .eq แบบไม่ระบุ type เข้มงวด)
        # ลองสลับจาก select("*") เป็นการระบุชื่อคอลัมน์ชัดๆ
        response = supabase.table("predictions")\
            .select("id, user_id, title, text, result, confidence, timestamp")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
            
        # 3. เช็คว่ามีข้อมูลออกมาจาก API จริงหรือไม่
        if hasattr(response, 'data') and response.data is not None:
            return response.data
        return []
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return []

# ==========================================
# 3. Feedbacks
# ==========================================

def save_feedback(prediction_id, user_report, comment="") -> bool:
    supabase = get_supabase()
    payload = {
        "prediction_id": prediction_id,
        "user_report": user_report,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    }
    try:
        supabase.table("feedbacks").insert(payload).execute()
        return True
    except Exception as e:
        print(f"Feedback Error: {e}")
        return False

def read_all_feedbacks() -> List[Tuple[int, str, str, str, str]]:
    supabase = get_supabase()
    try:
        # ดึง predictions(title) มาด้วย (Foreign Key)
        response = supabase.table("feedbacks")\
            .select("id, user_report, comment, status, timestamp, predictions(title)")\
            .order("timestamp", desc=True)\
            .execute()
            
        rows = []
        data = response.data
        
        if data is not None and isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    news_title = "Unknown"
                    predictions = item.get('predictions')
                    
                    # ✅ FIX: Pylance ไม่รู้ว่า predictions เป็น Dict หรือ None
                    # ต้องเช็ค isinstance อย่างละเอียดก่อนเรียก .get()
                    if predictions is not None:
                        if isinstance(predictions, dict):
                            news_title = str(predictions.get('title', "Unknown"))
                        elif isinstance(predictions, list) and len(predictions) > 0:
                            # บางที Supabase ส่งมาเป็น List ถ้าเป็น Relation 1-Many
                            first_pred = predictions[0]
                            if isinstance(first_pred, dict):
                                news_title = str(first_pred.get('title', "Unknown"))

                    rows.append((
                        int(str(item.get('id', 0))), 
                        news_title, 
                        str(item.get('user_report', '')), 
                        str(item.get('comment', '')), 
                        str(item.get('status', ''))
                    ))
        return rows
    except Exception as e:
        print(f"Read Feedback Error: {e}")
        return []

# ==========================================
# 4. Trending News
# ==========================================

def create_trending(headline, content, label):
    supabase = get_supabase()
    payload = {
        "headline": headline, 
        "content": content, 
        "label": label, 
        "updated_at": datetime.now().isoformat()
    }
    try:
        supabase.table("trending_news").insert(payload).execute()
    except Exception as e:
        print(f"Create Trending Error: {e}")

def get_all_trending() -> List[Tuple[int, str, str, str, str]]:
    supabase = get_supabase()
    try:
        response = supabase.table("trending_news").select("*").order("updated_at", desc=True).execute()
        rows = []
        data = response.data
        
        if data is not None and isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    rows.append((
                        int(str(item.get('id', 0))), 
                        str(item.get('headline', '')), 
                        str(item.get('content', '')), 
                        str(item.get('label', '')), 
                        str(item.get('updated_at', ''))
                    ))
        return rows
    except Exception as e:
        print(f"Get Trending Error: {e}")
        return []

def delete_trending(news_id):
    supabase = get_supabase()
    try:
        supabase.table("trending_news").delete().eq("id", news_id).execute()
    except Exception as e:
        print(f"Delete Trending Error: {e}")

# ==========================================
# 5. Password Reset System
# ==========================================

def send_otp_email(to_email) -> Tuple[bool, str]:
    supabase = get_supabase()
    
    try:
        user_check = supabase.table("users").select("id").eq("email", to_email).execute()
        if not (user_check.data is not None and isinstance(user_check.data, list) and len(user_check.data) > 0):
            return False, "❌ ไม่พบอีเมลนี้ในระบบ"
    except Exception as e:
        return False, f"Check Email Error: {e}"

    otp = ''.join(random.choices(string.digits, k=6))
    
    try:
        supabase.table("users").update({"reset_token": otp}).eq("email", to_email).execute()
    except Exception as e:
        return False, f"Database Error: {e}"

    subject = "🔑 รหัสยืนยันการเปลี่ยนรหัสผ่าน (Thai Fake News)"
    body = f"รหัส OTP ของคุณคือ: {otp}\n\nกรุณานำรหัสนี้ไปกรอกในหน้าเว็บเพื่อตั้งรหัสผ่านใหม่"
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        return True, "✅ ส่งรหัส OTP ไปที่อีเมลแล้ว"
    except Exception as e:
        print(f"Mail Error: {e}")
        return False, "❌ ส่งอีเมลไม่สำเร็จ (เช็ค App Password)"

def verify_otp_and_reset(email, otp, new_password) -> Tuple[bool, str]:
    supabase = get_supabase()
    
    try:
        response = supabase.table("users").select("id")\
            .eq("email", email)\
            .eq("reset_token", otp)\
            .execute()
            
        data = response.data
        
        if data is not None and isinstance(data, list) and len(data) > 0:
            new_pw_hash = hashlib.sha256(new_password.encode()).hexdigest()
            
            supabase.table("users").update({
                "password_hash": new_pw_hash,
                "reset_token": None 
            }).eq("email", email).execute()
            
            return True, "✅ เปลี่ยนรหัสผ่านสำเร็จ! กรุณาล็อกอินใหม่"
        else:
            return False, "❌ รหัส OTP ไม่ถูกต้อง หรือหมดอายุ"
    except Exception as e:
        return False, f"Reset Error: {e}"
# ==========================================
# ➕ ส่วนเสริมสำหรับ Admin Dashboard
# ==========================================

def read_all_predictions() -> List[Tuple[int, str, str, str, str, float, str]]:
    """ดึงข้อมูลการทำนายทั้งหมด เพื่อนำไปคำนวณสถิติ"""
    supabase = get_supabase()
    try:
        # join ตาราง users เพื่อเอา username มาแสดงด้วย
        response = supabase.table("predictions")\
            .select("*, users(username)")\
            .order("timestamp", desc=True)\
            .execute()
            
        rows = []
        data = response.data
        
        if data is not None and isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # จัดการ user object ที่ join มา (อาจเป็น dict หรือ list)
                    user_obj = item.get('users')
                    username = "Unknown"
                    
                    if isinstance(user_obj, dict):
                        username = str(user_obj.get('username', 'Unknown'))
                    elif isinstance(user_obj, list) and len(user_obj) > 0:
                        first_user = user_obj[0] # type: ignore
                        if isinstance(first_user, dict):
                            username = str(first_user.get('username', 'Unknown'))

                    # เรียงลำดับให้ตรงกับที่ frontend.py ต้องการ:
                    # columns=['ID', 'User', 'Title', 'Text', 'Result', 'Conf', 'Timestamp']
                    rows.append((
                        int(str(item.get('id', 0))), 
                        username,
                        str(item.get('title', '')), 
                        str(item.get('text', '')), 
                        str(item.get('result', '')), 
                        float(str(item.get('confidence', 0.0))),
                        str(item.get('timestamp', ''))
                    ))
        return rows
    except Exception as e:
        print(f"Read All Admin Error: {e}")
        return []

def read_all_predictions_limit(limit_num: int) -> List[Tuple[int, str, str, str, str, float, str]]:
    """ดึงข้อมูลล่าสุด N รายการ สำหรับหน้า Review"""
    supabase = get_supabase()
    try:
        response = supabase.table("predictions")\
            .select("*, users(username)")\
            .order("timestamp", desc=True)\
            .limit(limit_num)\
            .execute()
            
        rows = []
        data = response.data
        
        if data is not None and isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # จัดการ user object
                    user_obj = item.get('users')
                    username = "Unknown"
                    if isinstance(user_obj, dict):
                        username = str(user_obj.get('username', 'Unknown'))
                    elif isinstance(user_obj, list) and len(user_obj) > 0:
                        first_user = user_obj[0] # type: ignore
                        if isinstance(first_user, dict):
                            username = str(first_user.get('username', 'Unknown'))

                    rows.append((
                        int(str(item.get('id', 0))), 
                        username,
                        str(item.get('title', '')), 
                        str(item.get('text', '')), 
                        str(item.get('result', '')), 
                        float(str(item.get('confidence', 0.0))),
                        str(item.get('timestamp', ''))
                    ))
        return rows
    except Exception as e:
        print(f"Read Limit Admin Error: {e}")
        return []

    
# ==========================================
# 6. SYSTEM LOGGING (แก้ไขใหม่ แก้ Pylance Error)
# ==========================================

def get_system_logs(limit: int = 50):
    supabase = get_supabase()
    try:
        # ดึงข้อมูลแบบเรียบง่ายที่สุด (ไม่ Join) เพื่อให้มั่นใจว่าข้อมูลจะโผล่บนเว็บก่อน
        response = supabase.table("system_logs")\
            .select("timestamp, action, details, level, user_id")\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
            
        data = response.data
        rows = []

        if isinstance(data, list):
            for item in data:
                # ตรวจสอบว่าเป็น dict แน่ๆ เพื่อปิดปาก Pylance
                if isinstance(item, dict):
                    ts = str(item.get('timestamp', '-'))
                    act = str(item.get('action', '-'))
                    det = str(item.get('details', '-'))
                    lvl = str(item.get('level', 'INFO'))
                    uid = str(item.get('user_id', 'username'))
                    
                    rows.append((ts, uid, act, det, lvl))
        
        return rows
    except Exception as e:
        print(f"❌ Error fetching logs: {e}")
        return []
    
def log_system_event(user_id, action, details, level="INFO"):
    try:
        supabase = get_supabase()
        payload = {
            "user_id": user_id,
            
            "action": action,
            "details": details,
            "level": level,
            "timestamp": datetime.now().isoformat()
        }
        supabase.table("system_logs").insert(payload).execute()
    except Exception as e:
        print(f"Log Error: {e}")
# --- เพิ่มใน database_ops.py ---

# เพิ่มฟังก์ชันนี้เพื่อให้เชื่อมต่อ Supabase ได้
def get_db_connection():
    # ดึงค่าจาก st.secrets ที่เราตั้งไว้
    # ตรวจสอบให้แน่ใจว่าใน secrets.toml ใช้ชื่อหัวข้อว่า [supabase] หรือ [postgres]
    # แนะนำให้ใช้ Connection String ที่ได้จาก Supabase (Transaction Pooler หรือ Session Pooler)
    
    conn = psycopg2.connect(
        host=st.secrets["supabase"]["host"],
        database=st.secrets["supabase"]["dbname"],
        user=st.secrets["supabase"]["user"],
        password=st.secrets["supabase"]["password"],
        port=st.secrets["supabase"]["port"]
    )
    return conn

def get_dashboard_kpi():
    supabase = get_supabase()
    now_utc = datetime.now(timezone.utc)
    last_24h_str = (now_utc - timedelta(hours=24)).isoformat()

    stats = {"checks_today": 0, "active_users": 0, "accuracy": 0.0, "feedback_total": 0}

    try:
        # --- 1. Total Checks ---
        # แก้ปัญหา 'exact' โดยการบอก Python ว่านี่คือประเภทที่ยอมรับได้
        res_checks = supabase.table('predictions').select('*', count='exact').gte('timestamp', last_24h_str).execute() # type: ignore
        
        # ใช้การเช็ค hasattr เพื่อความปลอดภัย
        if hasattr(res_checks, 'count'):
            stats['checks_today'] = res_checks.count if res_checks.count is not None else 0

        # --- 2. Active Users ---
        res_users = supabase.table('system_logs').select('user_id').gte('timestamp', last_24h_str).execute()
        
        # แก้ปัญหา Error ".get() is unknown" และ "Unhashable"
        # โดยการบังคับประเภท (Type Casting) ให้ชัดเจนว่าเป็น List ของ Dict
        logs_data = res_users.data if res_users.data else []
        active_users_set = set()
        
        for row in logs_data:
            if isinstance(row, dict): # เช็คว่าเป็น dictionary จริงไหม
                uid = row.get('user_id')
                if uid is not None:
                    active_users_set.add(str(uid)) # แปลงเป็น str เพื่อให้ hashable แน่นอน
        
        stats['active_users'] = len(active_users_set)

       # --- 3. Accuracy & Feedback (ฉบับแก้ Pylance Error) ---
        res_fb = supabase.table('feedbacks').select('status').execute()
        fb_list: list = res_fb.data if res_fb.data else [] # ระบุว่าเป็น list
        stats['feedback_total'] = len(fb_list)

        if fb_list:
            # แก้บรรทัด DEBUG ให้ปลอดภัยขึ้น
            statuses = [str(item.get('status')) for item in fb_list if isinstance(item, dict)]
            print(f"DEBUG: Status values in DB are: {set(statuses)}")

        if stats['feedback_total'] > 0:
            correct_count = 0
            for item in fb_list:
                # การเช็ค isinstance(item, dict) จะทำให้ Pylance รู้ว่า item มี .get()
                if isinstance(item, dict):
                    # ระบุชนิดตัวแปร d: dict = item เพื่อความชัวร์ 100%
                    d: dict = item 
                    status_val = str(d.get('status', '')).lower().strip()
                    
                    if status_val in ['correct', 'true', 'yes', '1', 'pending']:
                        correct_count += 1
            
            stats['accuracy'] = round((correct_count / stats['feedback_total']) * 100, 1)
        return stats

    except Exception as e:
        print(f"❌ Error getting dashboard KPI: {e}")
        return stats
    
def get_model_performance_data():
    supabase = get_supabase()
    try:
        # 1. ดึงข้อมูลจากตาราง predictions อย่างเดียว
        # (Columns: id, user_id, title, text, url, result, confident, timestamp)
        res = supabase.table('predictions').select('*').execute()
        
        df = pd.DataFrame(res.data)

        # ถ้าไม่มีข้อมูล ให้ส่ง DataFrame เปล่ากลับไป
        if df.empty:
            return pd.DataFrame()

        # ---------------------------------------------------------
        # 2. แปลงชื่อคอลัมน์ให้ตรงกับที่ Frontend ต้องใช้
        # ---------------------------------------------------------

        # แปลง 'result' -> 'prediction' (เพื่อให้ตรงกับตัวแปร y_pred ใน frontend)
        if 'result' in df.columns:
            df.rename(columns={'result': 'prediction'}, inplace=True)

        # แปลง 'confident' -> 'confidence' (เผื่อ frontend ใช้คำนี้)
        # แปลง confident เป็น confidence และทำให้เป็นตัวเลข
        if 'confident' in df.columns:
            df.rename(columns={'confident': 'confidence'}, inplace=True)
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce') # บังคับเป็นตัวเลข

        # ---------------------------------------------------------
        # 3. จัดการเรื่อง Label (เฉลย)
        # ---------------------------------------------------------
        # เนื่องจากเราไม่ดึง feedbacks แล้ว เราจะไม่มี "ค่าจริง" (Ground Truth) มาเทียบ
        # แต่ Frontend ยังต้องการคอลัมน์ 'label' เพื่อคำนวณกราฟ
        # เราจึงต้องสร้างคอลัมน์หลอกขึ้นมา เพื่อกันไม่ให้โปรแกรม Error
        
        if 'label' not in df.columns:
            # ใส่เป็น 'pending' ทั้งหมด (เพราะเราไม่รู้ว่าจริงๆ แล้วข่าวนั้นจริงหรือปลอม)
            df['label'] = df['prediction']
            
            # ⚠️ หมายเหตุ: การทำแบบนี้ กราฟ Accuracy จะเป็น 0% เสมอ 
            # เพราะ 'prediction' (Real/Fake) จะไม่ตรงกับ 'label' (pending)

        # ---------------------------------------------------------
        # 4. จัดการชนิดข้อมูล (Data Types)
        # ---------------------------------------------------------
        
        # แปลง timestamp เป็น datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # แปลง id เป็น string
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)

        # คืนค่า DataFrame ที่เตรียมเสร็จแล้ว
        return df

    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        return pd.DataFrame()
    
def get_evaluated_data():
        """
        ดึงข้อมูล Prediction ที่มีการตรวจสอบ (Review) แล้ว
        """
        supabase = get_supabase()
        try:
            # ✅ แก้ไข: เปลี่ยน p.confident เป็น p.confidence
            response = supabase.table('predictions') \
                .select('id, content, prediction, confidence, result, timestamp') \
                .not_.is_('result', 'null') \
                .execute()
            
            data = response.data
            
            if data:
                # แปลงให้เป็น DataFrame เพื่อความง่าย
                df = pd.DataFrame(data)
                
                # Rename columns ให้เข้าใจง่ายขึ้น (Optional)
                df = df.rename(columns={
                    'result': 'status',
                    'timestamp': 'timestamp',
                    'text': 'text'
                })
                return df
            return pd.DataFrame()
            
        except Exception as e:
            # print(f"❌ Error getting evaluated data: {e}") # Debug ดู error เดิม
            return pd.DataFrame()

def get_pending_feedbacks():
    supabase = get_supabase()
    try:
        # 1. ดึง Feedback ที่ยังไม่ตรวจ
        res_f = supabase.table('feedbacks').select('*').eq('status', 'pending').execute()
        
        # เช็คให้ชัวร์ว่าเป็น List และไม่ว่าง
        feedbacks = res_f.data if res_f.data is not None else []
        
        if not feedbacks:
            return []

        result_list = []
        for fb in feedbacks:
            # 🚨 แก้ Error: เช็คว่า fb เป็น dict จริงๆ
            if not isinstance(fb, dict):
                continue

            pred_id = fb.get('prediction_id')
            
            if pred_id:
                # ดึงข้อมูลข่าว
                res_p = supabase.table('predictions').select('*').eq('id', str(pred_id)).execute()
                
                # 🚨 แก้ Error: เช็คว่ามี data และเป็น list ที่มีของ
                if res_p.data and isinstance(res_p.data, list) and len(res_p.data) > 0:
                    pred_data = res_p.data[0] # หยิบตัวแรก
                    
                    # เช็คอีกทีว่าเป็น dict ถึงจะใช้ .get()
                    if isinstance(pred_data, dict):
                        combined_data = {
                            'feedback_id': fb.get('id'),
                            'prediction_id': pred_id,
                            'title': pred_data.get('title', 'No Title'),
                            'text': pred_data.get('text', ''),
                            'ai_result': pred_data.get('result', 'Unknown'),
                            'ai_confidence': pred_data.get('confident', 0),
                            'user_comment': fb.get('feedback_text', '-'),
                            'timestamp': fb.get('created_at')
                        }
                        result_list.append(combined_data)
        
        return result_list

    except Exception as e:
        print(f"❌ Error fetching pending feedbacks: {e}")
        return []

def update_feedback_status(feedback_id, new_status):
    """
    อัปเดตสถานะ Feedback (เช่น pending -> Real หรือ Fake)
    """
    supabase = get_supabase()
    try:
        data = supabase.table('feedbacks').update({'status': new_status}).eq('id', feedback_id).execute()
        return True
    except Exception as e:
        print(f"❌ Error updating feedback: {e}")
        return False