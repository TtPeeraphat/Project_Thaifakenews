import hashlib
import random
import string
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Union, cast

import pandas as pd
import psycopg2
import streamlit as st
from supabase import create_client, Client
from config import config

# ==========================================
# ⚙️ 0. CONFIGURATION & DATABASE INIT
# ==========================================
# ✅ FIX #3: Credentials now loaded from .env file via config.py
# This prevents hardcoded credentials in source code

SUPABASE_URL = config.database.supabase_url
SUPABASE_KEY = config.database.supabase_key
SENDER_EMAIL = config.email.sender_email
SENDER_PASSWORD = config.email.sender_password

def get_supabase() -> Client:
    """เชื่อมต่อกับ Supabase API"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_db_connection():
    """เชื่อมต่อกับ PostgreSQL โดยตรง (สำหรับบาง Query ที่ซับซ้อน)"""
    return psycopg2.connect(
        host=st.secrets["supabase"]["host"],
        database=st.secrets["supabase"]["dbname"],
        user=st.secrets["supabase"]["user"],
        password=st.secrets["supabase"]["password"],
        port=st.secrets["supabase"]["port"]
    )

# ==========================================
# 👤 1. USER MANAGEMENT
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
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
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
            .eq("username", username).eq("password_hash", pw_hash).execute()
            
        data = response.data
        if data and isinstance(data, list) and len(data) > 0:
            user_data = data[0]
            if isinstance(user_data, dict):
                uid = int(str(user_data.get('id', 0)))
                uname = str(user_data.get('username', ''))
                urole = str(user_data.get('role', ''))
                return (uid, uname, urole)
        return None
    except Exception as e:
        print(f"Auth Error: {e}")
        return None

# ==========================================
# 🤖 2. PREDICTIONS (AI SCAN)
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
        if data and isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict):
                return int(str(item.get('id', 0)))
        return None
    except Exception as e:
        print(f"Create Prediction Error: {e}")
        return None

def get_user_history(user_id: Any, limit: int = 50):
    supabase = get_supabase()
    try:
        if user_id is None:
            return []
        response = supabase.table("predictions")\
            .select("id, user_id, title, text, result, confidence, timestamp")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(limit).execute()
            
        if hasattr(response, 'data') and response.data is not None:
            return response.data
        return []
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return []

# ==========================================
# 📝 3. FEEDBACK & EVALUATION (แก้ไข Logic แล้ว!)
# ==========================================
def save_feedback(prediction_id, user_report, comment="") -> bool:
    supabase = get_supabase()
    payload = {
        "prediction_id": prediction_id,
        "user_report": user_report,  # เช่น "Correct" หรือ "Incorrect"
        "comment": comment,
        "status": "pending", # เริ่มต้นต้องรอ Admin ตรวจ
        "timestamp": datetime.now().isoformat()
    }
    try:
        supabase.table("feedbacks").insert(payload).execute()
        return True
    except Exception as e:
        print(f"Feedback Error: {e}")
        return False

def get_pending_feedbacks():
    supabase = get_supabase()
    try:
        res_f = supabase.table('feedbacks').select('*').eq('status', 'pending').execute()
        feedbacks = res_f.data if res_f.data else []
        if not feedbacks: return []

        result_list = []
        for fb in feedbacks:
            if not isinstance(fb, dict): continue
            pred_id = fb.get('prediction_id')
            
            if pred_id:
                res_p = supabase.table('predictions').select('*').eq('id', str(pred_id)).execute()
                if res_p.data and isinstance(res_p.data, list) and len(res_p.data) > 0:
                    pred_data = res_p.data[0]
                    if isinstance(pred_data, dict):
                        result_list.append({
                            'feedback_id': fb.get('id'),
                            'prediction_id': pred_id,
                            'title': pred_data.get('title', 'No Title'),
                            'text': pred_data.get('text', ''),
                            'ai_result': pred_data.get('result', 'Unknown'),
                            'ai_confidence': pred_data.get('confidence', 0), # แก้ไขคำว่า confident
                            'user_comment': fb.get('comment', '-'),
                            'user_report': fb.get('user_report', '-'),
                            'timestamp': fb.get('timestamp')
                        })
        return result_list
    except Exception as e:
        print(f"❌ Error fetching pending feedbacks: {e}")
        return []

def update_feedback_status(feedback_id, new_status) -> bool:
    """Admin ใช้เปลี่ยนสถานะจาก pending -> verified_real หรือ verified_fake"""
    supabase = get_supabase()
    try:
        supabase.table('feedbacks').update({'status': new_status}).eq('id', feedback_id).execute()
        return True
    except Exception as e:
        print(f"❌ Error updating feedback: {e}")
        return False

def read_all_feedbacks() -> List[Tuple[int, str, str, str, str]]:
    supabase = get_supabase()
    try:
        response = supabase.table("feedbacks")\
            .select("id, user_report, comment, status, timestamp, predictions(title)")\
            .order("timestamp", desc=True).execute()
            
        rows = []
        data = response.data
        if data and isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    news_title = "Unknown"
                    preds = item.get('predictions')
                    if preds:
                        if isinstance(preds, dict):
                            news_title = str(preds.get('title', "Unknown"))
                        elif isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                            news_title = str(preds[0].get('title', "Unknown"))

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
# 📊 4. ADMIN DASHBOARD & KPI 
# ==========================================

def get_dashboard_kpi():
    supabase = get_supabase()
    # ใช้ UTC เพื่อให้ตรงกับมาตรฐานของ Supabase
    now_utc = datetime.now(timezone.utc)
    last_24h_str = (now_utc - timedelta(hours=24)).isoformat()
    
    stats = {"checks_today": 0, "active_users": 0, "accuracy": 0.0, "feedback_total": 0}

    try:
        # 1. Total Checks Today
        res_checks = supabase.table('predictions').select('id', count='exact').gte('timestamp', last_24h_str).execute()
        stats['checks_today'] = res_checks.count if res_checks.count else 0

        # 2. Active Users (จาก system_logs)
        res_users = supabase.table('system_logs').select('user_id').gte('timestamp', last_24h_str).execute()
        logs_data = res_users.data or []
        stats['active_users'] = len({str(row.get('user_id')) for row in logs_data if row.get('user_id')})

        # 3. Accuracy Calculation (นับจาก Feedback ที่ Verify แล้ว)
        # ดึง user_report และ ai_result (ผ่านการ Join ตาราง predictions)
        res_fb = supabase.table('feedbacks').select('user_report, status, predictions(result)').neq('status', 'pending').execute()
        fb_list = res_fb.data or []
        stats['feedback_total'] = len(fb_list)

        if stats['feedback_total'] > 0:
            # AI ทายถูก = (User บอกว่า Correct) หรือ (Status คือ verified_correct)
            correct_count = sum(1 for item in fb_list if 
                                str(item.get('user_report', '')).lower() == 'correct' or 
                                str(item.get('status', '')).lower() == 'verified_correct')
            stats['accuracy'] = round((correct_count / stats['feedback_total']) * 100, 1)
            
        return stats
    except Exception as e:
        print(f"KPI Error: {e}")
        return stats

def get_evaluated_data():
    """
    ดึงข้อมูล AI Prediction พร้อมกับ 'สถานะที่ Admin ตรวจสอบแล้ว' (status)
    """
    supabase = get_supabase()
    try:
        # 1. ดึงข้อมูล Feedback ที่ถูก Review แล้ว (status ไม่ใช่ pending)
        res_fb = supabase.table('feedbacks').select('prediction_id, status').neq('status', 'pending').execute()
        
        # คืนค่า DataFrame ว่าง ถ้ายังไม่มีข้อมูล
        if not res_fb.data:
            return pd.DataFrame()
            
        df_fb = pd.DataFrame(res_fb.data)
        
        # 2. ดึงข้อมูล Predictions
        # 🚨 แก้ไขตรงนี้ครับ: เปลี่ยนจาก confident เป็น confidence
        res_pred = supabase.table('predictions').select('id, result, confidence, timestamp').execute()
        
        if not res_pred.data:
            return pd.DataFrame()
            
        df_pred = pd.DataFrame(res_pred.data)
        
        # 3. คลีนข้อมูลและเปลี่ยนชื่อคอลัมน์ให้ตรงใจ Frontend
        # (เหลือแค่เปลี่ยน result เป็น prediction เพราะ confidence ชื่อตรงแล้ว)
        df_pred.rename(columns={'result': 'prediction'}, inplace=True)
        
        # แปลง ID เป็น string ป้องกันปัญหาตอน Join
        df_pred['id'] = df_pred['id'].astype(str)
        df_fb['prediction_id'] = df_fb['prediction_id'].astype(str)
        
        # 4. Join 2 ตารางเข้าด้วยกัน
        df_merged = pd.merge(df_pred, df_fb, left_on='id', right_on='prediction_id', how='inner')
        return df_merged
        
    except Exception as e:
        print(f"❌ Error getting evaluated data: {e}")
        return pd.DataFrame()

def get_model_performance_data():
    supabase = get_supabase()
    try:
        # ✅ จุดที่แก้ไข: เติม !fk_prediction หลังคำว่า predictions เพื่อระบุ Foreign Key ให้ชัดเจน
        res = supabase.table('feedbacks').select('''
            user_report, 
            status, 
            predictions!fk_prediction(result, confidence, timestamp)
        ''').neq('status', 'pending').execute()
        
        if not res.data:
            return pd.DataFrame()

        processed_data = []
        for item in res.data:
            # Supabase มักจะคืนค่า key เป็นชื่อตารางเหมือนเดิม
            pred = item.get('predictions')
            if not pred: continue
            
            ai_pred = str(pred.get('result')).capitalize() # Real / Fake
            user_rep = str(item.get('user_report')).lower()
            
            # หาค่าจริง (Ground Truth)
            if user_rep == 'correct':
                true_label = ai_pred
            else:
                true_label = "Fake" if ai_pred == "Real" else "Real"
                
            processed_data.append({
                "timestamp": pred.get('timestamp'),
                "prediction": ai_pred,
                "confidence": pred.get('confidence'),
                "label": true_label,
                "is_correct": user_rep == 'correct'
            })

        df = pd.DataFrame(processed_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    except Exception as e:
        print(f"Performance Data Error: {e}")
        return pd.DataFrame()

def read_all_predictions_limit(limit_num: int) -> List[Tuple[int, str, str, str, str, float, str]]:
    supabase = get_supabase()
    try:
        response = supabase.table("predictions").select("*, users(username)").order("timestamp", desc=True).limit(limit_num).execute()
        rows = []
        if response.data and isinstance(response.data, list):
            for item in response.data:
                if isinstance(item, dict):
                    user_obj = item.get('users')
                    username = "Unknown"
                    if isinstance(user_obj, dict): username = str(user_obj.get('username', 'Unknown'))
                    elif isinstance(user_obj, list) and len(user_obj) > 0 and isinstance(user_obj[0], dict): username = str(user_obj[0].get('username', 'Unknown'))

                    rows.append((
                        int(str(item.get('id', 0))), username, str(item.get('title', '')), 
                        str(item.get('text', '')), str(item.get('result', '')), 
                        float(str(item.get('confidence', 0.0))), str(item.get('timestamp', ''))
                    ))
        return rows
    except Exception as e:
        return []

# ==========================================
# 📈 5. TRENDING NEWS
# ==========================================
# ==========================================
# 📈 TRENDING NEWS MANAGEMENT
# ==========================================

def get_all_trending():
    """ดึงรายการข่าวที่เป็นกระแสทั้งหมด (คืนค่าเป็น DataFrame)"""
    supabase = get_supabase()
    try:
        # 🚨 แก้ชื่อคอลัมน์เป็น updated_at
        response = supabase.table("trending_news").select("*").order("updated_at", desc=True).execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame() 
        
    except Exception as e:
        print(f"❌ Get Trending Error: {e}")
        return pd.DataFrame()

def create_trending(headline, content, label):
    """เพิ่มข่าวที่เป็นกระแสใหม่"""
    supabase = get_supabase()
    payload = {
        "headline": headline, 
        "content": content, 
        "label": label, 
        "updated_at": datetime.now().isoformat() # 🚨 แก้ชื่อคอลัมน์เป็น updated_at
    }
    try:
        supabase.table("trending_news").insert(payload).execute()
        return True
    except Exception as e:
        print(f"❌ Create Trending Error: {e}")
        return False
    
def update_trending(news_id, headline, content, label):
    """แก้ไขข่าวที่เป็นกระแส"""
    supabase = get_supabase()
    payload = {
        "headline": headline,
        "content": content,
        "label": label,
        "updated_at": datetime.now().isoformat()
    }
    try:
        supabase.table("trending_news").update(payload).eq("id", news_id).execute()
        return True
    except Exception as e:
        print(f"❌ Update Trending Error: {e}")
        return False

def delete_trending(news_id):
    """ลบข่าวที่เป็นกระแส"""
    supabase = get_supabase()
    try:
        supabase.table("trending_news").delete().eq("id", news_id).execute()
        return True
    except Exception as e:
        print(f"❌ Delete Trending Error: {e}")
        return False

# ==========================================
# 🔐 6. PASSWORD RESET OTP
# ==========================================
def send_otp_email(to_email) -> Tuple[bool, str]:
    supabase = get_supabase()
    try:
        user_check = supabase.table("users").select("id").eq("email", to_email).execute()
        if not (user_check.data and isinstance(user_check.data, list) and len(user_check.data) > 0): return False, "❌ ไม่พบอีเมลนี้ในระบบ"
    except Exception as e: return False, f"Check Email Error: {e}"

    otp = ''.join(random.choices(string.digits, k=6))
    try: supabase.table("users").update({"reset_token": otp}).eq("email", to_email).execute()
    except Exception as e: return False, f"Database Error: {e}"

    msg = MIMEText(f"รหัส OTP ของคุณคือ: {otp}\n\nกรุณานำรหัสนี้ไปกรอกในหน้าเว็บเพื่อตั้งรหัสผ่านใหม่")
    msg['Subject'] = "🔑 รหัสยืนยันการเปลี่ยนรหัสผ่าน (Thai Fake News)"
    msg['From'] = SUPABASE_EMAIL
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SUPABASE_EMAIL, SUPABASE_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        return True, "✅ ส่งรหัส OTP ไปที่อีเมลแล้ว"
    except Exception as e:
        return False, "❌ ส่งอีเมลไม่สำเร็จ (เช็ค App Password)"

def verify_otp_and_reset(email, otp, new_password) -> Tuple[bool, str]:
    supabase = get_supabase()
    try:
        response = supabase.table("users").select("id").eq("email", email).eq("reset_token", otp).execute()
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            new_pw_hash = hashlib.sha256(new_password.encode()).hexdigest()
            supabase.table("users").update({"password_hash": new_pw_hash, "reset_token": None}).eq("email", email).execute()
            return True, "✅ เปลี่ยนรหัสผ่านสำเร็จ! กรุณาล็อกอินใหม่"
        return False, "❌ รหัส OTP ไม่ถูกต้อง หรือหมดอายุ"
    except Exception as e:
        return False, f"Reset Error: {e}"

# ==========================================
# 📝 7. SYSTEM LOGGING
# ==========================================
def get_system_logs(limit: int = 50):
    supabase = get_supabase()
    try:
        response = supabase.table("system_logs").select("timestamp, action, details, level, user_id").order("timestamp", desc=True).limit(limit).execute()
        rows = []
        if isinstance(response.data, list):
            for item in response.data:
                if isinstance(item, dict):
                    rows.append((str(item.get('timestamp', '-')), str(item.get('user_id', 'username')), str(item.get('action', '-')), str(item.get('details', '-')), str(item.get('level', 'INFO'))))
        return rows
    except Exception as e:
        return []
    


def log_system_event(user_id, action, details, level="INFO"):
    """
    บันทึกเหตุการณ์ลงตาราง system_logs
    """
    supabase = get_supabase()
    
    # 1. จัดการ User ID (ถ้าระบบทำเอง ให้ใช้ชื่อ "System")
    if not user_id:
        user_id = "System"
        
    # 2. จัดการเวลาปัจจุบัน (แบบ UTC เพื่อป้องกันปัญหา Timezone)
    now_utc = datetime.now(timezone.utc).isoformat()
    
    # 3. เตรียมข้อมูลให้คีย์ตรงกับชื่อคอลัมน์ใน Database ของคุณเป๊ะๆ
    log_data = {
        "user_id": str(user_id),
        "action": str(action).upper(),
        "timestamp": now_utc,      # ส่งเวลาไปเก็บในคอลัมน์ timestamp
        "details": str(details),    
        "level": str(level).upper()
    }
    
    try:
        # 4. Insert ข้อมูลลงตาราง
        supabase.table("system_logs").insert(log_data).execute()
        return True
        
    except Exception as e:
        print(f"❌ ไม่สามารถบันทึก Log ได้ [{action}]: {e}")
        return False


def get_system_analytics_data():
    """ดึงข้อมูลดิบจาก Supabase เพื่อนำมาวิเคราะห์ในหน้า System Analytics"""
    supabase = get_supabase()
    try:
        # 1. นับจำนวน Users ทั้งหมด
        res_users = supabase.table('users').select('id', count='exact').execute()
        total_users = res_users.count if hasattr(res_users, 'count') and res_users.count else 0
        
        # 2. ดึงข้อมูล Predictions ทั้งหมดเพื่อดูกราฟเวลาและสัดส่วน Fake/Real
        res_preds = supabase.table('predictions').select('id, result, timestamp').execute()
        df_preds = pd.DataFrame(res_preds.data) if res_preds.data else pd.DataFrame()
        
        # 3. ดึง System Logs (ย้อนหลัง 7 วัน) เพื่อดูจำนวน Active Users
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        res_logs = supabase.table('system_logs').select('timestamp, user_id').gte('timestamp', seven_days_ago).execute()
        df_logs = pd.DataFrame(res_logs.data) if res_logs.data else pd.DataFrame()
        
        return {
            "total_users": total_users,
            "df_preds": df_preds,
            "df_logs": df_logs
        }
    except Exception as e:
        print(f"❌ Error getting analytics data: {e}")
        return {
            "total_users": 0,
            "df_preds": pd.DataFrame(),
            "df_logs": pd.DataFrame()
        }
    
#manage user 
def get_user_management_data():
    """ดึงข้อมูลผู้ใช้ทั้งหมด พร้อมสถิติการใช้งานแต่ละคน"""
    supabase = get_supabase()
    try:
        # 1. ดึงข้อมูล User ทั้งหมด
        res_users = supabase.table('users').select('id, username, email, role, created_at').execute()
        df_users = pd.DataFrame(res_users.data) if res_users.data else pd.DataFrame()
        
        if df_users.empty:
            return pd.DataFrame()
            
        # สมมติว่าใน DB ยังไม่มีคอลัมน์ status ให้ตั้งค่าเริ่มต้นเป็น active ไว้ก่อน
        # (ถ้าใน DB คุณเพิ่มคอลัมน์ status แล้ว ให้ลบบรรทัดนี้ทิ้งและไปเพิ่มใน select() ด้านบนแทน)
        if 'status' not in df_users.columns:
            df_users['status'] = 'active'

        # 2. ดึงข้อมูล Predictions เพื่อนับจำนวน Checks ของแต่ละคน
        res_preds = supabase.table('predictions').select('user_id').execute()
        df_preds = pd.DataFrame(res_preds.data) if res_preds.data else pd.DataFrame()
        
        # 3. ดึงข้อมูล Logs เพื่อหาวันที่ Last Active
        res_logs = supabase.table('system_logs').select('user_id, timestamp').execute()
        df_logs = pd.DataFrame(res_logs.data) if res_logs.data else pd.DataFrame()

        # --- รวมข้อมูล (Merge) ---
        # นับจำนวน Checks
        if not df_preds.empty and 'user_id' in df_preds.columns:
            checks_count = df_preds.groupby('user_id').size().reset_index(name='checks')
            df_users = pd.merge(df_users, checks_count, left_on='id', right_on='user_id', how='left')
            df_users['checks'] = df_users['checks'].fillna(0).astype(int)
        else:
            df_users['checks'] = 0

        # หาวันที่ Active ล่าสุด
        if not df_logs.empty and 'user_id' in df_logs.columns:
            # หา Timestamp ล่าสุด (max) ของแต่ละ user
            last_active = df_logs.groupby('user_id')['timestamp'].max().reset_index(name='last_active')
            df_users = pd.merge(df_users, last_active, left_on='id', right_on='user_id', how='left')
        else:
            df_users['last_active'] = None

        return df_users
        
    except Exception as e:
        print(f"❌ Error getting user management data: {e}")
        return pd.DataFrame()

def update_user_role_status(target_user_id, new_role, new_status):
    """อัปเดตสิทธิ์ (Role) และสถานะ (Status) ของผู้ใช้"""
    supabase = get_supabase()
    try:
        payload = {"role": new_role}
        payload["status"] = new_status 
        
        supabase.table('users').update(payload).eq('id', target_user_id).execute()
        return True
    except Exception as e:
        print(f"❌ Error updating user: {e}")
        return False



def get_approved_feedbacks():
    """ดึง Feedback ที่แอดมินตรวจสอบแล้ว (Real หรือ Fake เท่านั้น) เพื่อนำไป Train โมเดล"""
    supabase = get_supabase()
    try:
        # ดึงเฉพาะรายการที่ status เป็น Real หรือ Fake (ข้าม Ignored หรือ Pending)
        # 🚨 อย่าลืมเช็คชื่อคอลัมน์ status และ text ให้ตรงกับในตาราง Supabase ของคุณนะครับ
        response = supabase.table("user_feedbacks") \
                           .select("text, status") \
                           .in_("status", ["Real", "Fake"]) \
                           .execute()
                           
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Get Approved Feedbacks Error: {e}")
        return pd.DataFrame()
    
def get_all_system_logs():
    supabase = get_supabase()
    try:
        # 🚨 เช็คว่าตรงนี้มี .limit(...) หรือเปล่า ถ้ามีให้แก้ให้ดึงเยอะขึ้น
        res = supabase.table('system_logs').select('*').order('timestamp', desc=True).limit(200).execute()
        return res.data
    except Exception as e:
        print(f"Error: {e}")
        return []
def get_user_email(user_id):
    """ฟังก์ชันดึงอีเมลจาก Supabase"""
    supabase = get_supabase()
    try:
        # ค้นหาอีเมลจากตาราง users โดยเทียบกับ id (หรือ user_id ของคุณ)
        response = supabase.table('users').select('email').eq('id', user_id).execute()
        
        if response.data and len(response.data) > 0:
            # ถ้ามีอีเมลให้ส่งค่ากลับไป ถ้าไม่มี(เป็น null) ให้ส่งเป็นค่าว่าง ""
            return response.data[0].get('email') or ""
        return ""
    except Exception as e:
        print(f"Error fetching email: {e}")
        return ""

def update_user_email(user_id, new_email):
    supabase = get_supabase()
    """ฟังก์ชันอัปเดตอีเมลลง Supabase"""
    try:
        # สั่งอัปเดตช่อง email ให้เป็นค่าใหม่ โดยเทียบกับ id
        response = supabase.table('users').update({'email': new_email}).eq('id', user_id).execute()
        return True # ส่งกลับ True เพื่อบอกว่าบันทึกสำเร็จ
    except Exception as e:
        print(f"Error updating email: {e}")
        return False
