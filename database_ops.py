import hashlib
import random
import string
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Union, cast

import pandas as pd

from config import config


# Optional import for Streamlit (only needed when running as Streamlit app)
try:
    import streamlit as st
except ImportError:
    st = None  # Will be None if not running in Streamlit context


from supabase import create_client, Client

# ==========================================
# ⚙️ 0. CONFIGURATION & DATABASE INIT
# ==========================================
# ลบ SUPABASE_URL และ SUPABASE_KEY ที่อยู่ข้างนอกฟังก์ชันทิ้งไปเลยครับ 

def get_supabase() -> Client:
    """เชื่อมต่อกับ Supabase API โดยดึงค่าแบบ Real-time"""
    
    # 1. ดึงค่าสดๆ หน้างาน เพื่อแก้ปัญหา Uvicorn อ่านไฟล์ .env ไม่ทัน
    raw_url = config.database.supabase_url
    raw_key = config.database.supabase_key
    
    # 2. คลีนข้อมูลให้สะอาดหมดจด (ลบฟันหนู ช่องว่าง และตัวขึ้นบรรทัดใหม่)
    clean_url = str(raw_url).strip().strip('"').strip("'") if raw_url else ""
    clean_key = str(raw_key).strip().strip('"').strip("'") if raw_key else ""
    
    # 3. ปริ้นท์เพื่อเช็กให้ชัวร์ว่าเซิร์ฟเวอร์เห็น URL เป็นอะไรตอนล็อกอิน
    print(f"👉 [DEBUG] ระบบกำลังเชื่อมต่อ Supabase ด้วย URL: {repr(clean_url)}")
    
    if not clean_url or not clean_key:
        raise ValueError(
            "Missing Supabase credentials! "
            "Ensure SUPABASE_URL and SUPABASE_KEY are properly loaded by config."
        )
    
    try:
        return create_client(clean_url, clean_key)
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
        raise



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
    """Authenticate user with username and password.
    
    Returns: (user_id, username, role) tuple or None if authentication fails
    
    Common errors:
    - OSError 11001: getaddrinfo failed → Network/DNS issue with Supabase
    - Invalid credentials → Wrong username or password
    """
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        supabase = get_supabase()
        
        # Attempt authentication
        response = supabase.table("users").select("id, username, role")\
            .eq("username", username).eq("password_hash", pw_hash).execute()
            
        data = response.data
        if data and isinstance(data, list) and len(data) > 0:
            user_data = data[0]
            if isinstance(user_data, dict):
                uid = int(str(user_data.get('id', 0)))
                uname = str(user_data.get('username', ''))
                urole = str(user_data.get('role', ''))
                print(f"✅ User '{username}' authenticated successfully")
                return (uid, uname, urole)
        
        # No user found with those credentials
        print(f"❌ Authentication failed for user '{username}': Invalid credentials")
        return None
        
    except OSError as e:
        if "11001" in str(e) or "getaddrinfo" in str(e):
            print(f"❌ [NETWORK ERROR] Cannot reach Supabase: {e}")
            print(f"   - Check internet connection")
            print(f"   - Verify SUPABASE_URL in .env is correct")
            print(f"   - Supabase project may be down")
        else:
            print(f"❌ [OS ERROR] {e}")
        return None
    except Exception as e:
        # Catch all other exceptions
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"❌ [AUTH ERROR] {error_type}: {error_msg}")
        
        # Log more details for debugging
        if "connection" in error_msg.lower():
            print(f"   → Connection issue. Check network and Supabase URL")
        elif "credentials" in error_msg.lower():
            print(f"   → Invalid credentials in .env file")
        
        return None

# ==========================================
# 🤖 2. PREDICTIONS (AI SCAN)
# ==========================================
def create_prediction(user_id, title, text, url, result, confidence, category=""):
    supabase = get_supabase()
    payload = {
        "user_id":    user_id,
        "title":      title,
        "text":       text,
        "url":        url,
        "result":     result,
        "confidence": confidence,
        "category":   category,   # ✅ เพิ่ม
        "timestamp":  datetime.now().isoformat()
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
        # ✅ ดึงทุก status ไม่ใช่แค่ pending
        res_f = supabase.table('feedbacks') \
                        .select('id, prediction_id, user_report, comment, status, timestamp') \
                        .execute()
        feedbacks = res_f.data if res_f.data else []
        if not feedbacks: return []

        fb_map = {}
        for fb in feedbacks:
            if not isinstance(fb, dict): continue
            if fb.get('id') is None: continue
            pid = str(fb.get('prediction_id'))
            fb_map[pid] = fb

        res_p = supabase.table('predictions').select('*').execute()
        predictions = res_p.data if res_p.data else []

        result_list = []
        for pred in predictions:
            if not isinstance(pred, dict): continue
            pid = str(pred.get('id'))
            fb  = fb_map.get(pid)
            if fb is None or fb.get('id') is None: continue

            # หาบรรทัด select predictions
            result_list.append({
                'feedback_id':   fb.get('id'),
                'prediction_id': pid,
                'title':         pred.get('title', 'No Title'),
                'text':          pred.get('text', ''),
                'ai_result':     pred.get('result', 'Unknown'),
                'ai_confidence': pred.get('confidence', 0),
                'category':      pred.get('category', 'ไม่ระบุ'),  
                'url':           pred.get('url', ''), 
                'user_report':   fb.get('user_report', None),
                'user_comment':  fb.get('comment', ''),
                'timestamp':     pred.get('timestamp'),
                'status':        fb.get('status', 'pending'),
                'has_feedback':  True,
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
    .select("id, user_report, comment, status, timestamp, predictions!fk_prediction(title)")\
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
        res_fb = supabase.table('feedbacks').select('user_report, status, predictions!fk_prediction(result)').neq('status', 'pending').execute()

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
        # ✅ ดึงจาก predictions โดยตรง — ไม่ filter ตาม user หรือ feedback status
        res = supabase.table('predictions') \
                      .select('result, confidence, timestamp') \
                      .order('timestamp', desc=True) \
                      .limit(500) \
                      .execute()

        if not res.data:
            return pd.DataFrame()

        processed_data = []
        for item in res.data:
            ai_pred = str(item.get('result', '')).capitalize()
            if ai_pred not in ('Real', 'Fake'):
                continue
            processed_data.append({
                'timestamp':  item.get('timestamp'),
                'prediction': ai_pred,
                'confidence': item.get('confidence'),
                'label':      ai_pred,   # ยังไม่มี ground truth
                'is_correct': None       # รอ Admin review
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

def create_trending(headline, content, label, category="ทั่วไป", image_url="", source_url=""):
    supabase = get_supabase()
    payload = {
        "headline":   headline,
        "content":    content,
        "label":      label,
        "category":   category,
        "image_url":  image_url if image_url else None,
        "source_url": source_url if source_url else None,   # ← เพิ่ม
        "created_at": datetime.now().isoformat(),            # ← เพิ่ม
        "updated_at": datetime.now().isoformat()
    }
    try:
        supabase.table("trending_news").insert(payload).execute()
        return True
    except Exception as e:
        print(f"❌ Create Trending Error: {e}")
        return False

    
# ✅ ใหม่
def update_trending(news_id, headline, content, label, category="ทั่วไป", image_url=None, source_url=None):
    supabase = get_supabase()
    payload = {
        "headline":   headline,
        "content":    content,
        "label":      label,
        "category":   category,
        "updated_at": datetime.now().isoformat()
    }
    if image_url:
        payload["image_url"] = image_url
    if source_url is not None:                # ← เพิ่ม (None = ไม่แก้, "" = ลบ)
        payload["source_url"] = source_url if source_url else None
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
        if not (user_check.data and isinstance(user_check.data, list) and len(user_check.data) > 0):
            return False, "❌ ไม่พบอีเมลนี้ในระบบ"
    except Exception as e:
        return False, f"Check Email Error: {e}"

    otp = ''.join(random.choices(string.digits, k=6))
    try:
        supabase.table("users").update({"reset_token": otp}).eq("email", to_email).execute()
    except Exception as e:
        return False, f"Database Error: {e}"

    # ✅ ดึงค่าจาก config แทนการใช้ตัวแปร global
    sender_email    = str(config.email.sender_email).strip()
    sender_password = str(config.email.sender_password).strip()
    print(f"📧 sender_email = {repr(sender_email)}")
    print(f"🔑 password length = {len(sender_password)} chars")

    msg = MIMEText(f"รหัส OTP ของคุณคือ: {otp}\n\nกรุณานำรหัสนี้ไปกรอกในหน้าเว็บเพื่อตั้งรหัสผ่านใหม่")
    msg['Subject'] = "🔑 รหัสยืนยันการเปลี่ยนรหัสผ่าน (Thai Fake News)"
    msg['From']    = sender_email  # ✅ ใช้ค่าจาก config
    msg['To']      = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)  # ✅ ใช้ค่าจาก config
            server.sendmail(sender_email, to_email, msg.as_string())
        return True, "✅ ส่งรหัส OTP ไปที่อีเมลแล้ว"
    except Exception as e:
         return False, f"❌ ส่งอีเมลไม่สำเร็จ: {str(e)}"

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
    try:
        supabase = get_supabase()
        
        if not user_id:
            user_id = "System"
            
        now_utc = datetime.now(timezone.utc).isoformat()
        log_data = {
            "user_id": str(user_id),
            "action": str(action).upper(),
            "timestamp": now_utc,
            "details": str(details),
            "level": str(level).upper()
        }
        supabase.table("system_logs").insert(log_data).execute()
        return True
        
    except Exception as e:
        print(f"⚠️ Log skipped (network unavailable): {action}")
        return False  # fail silently


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
    supabase = get_supabase()
    try:
        res = supabase.table("feedbacks") \
                      .select("id, status, timestamp, prediction_id, predictions!fk_prediction(text, title, result, category)") \
                      .in_("status", ["Real", "Fake"]) \
                      .execute()

        if not res.data:
            return pd.DataFrame()

        rows = []
        for item in res.data:
            if not isinstance(item, dict): continue
            pred = item.get('predictions')
            if not pred: continue

            rows.append({
                'text':      pred.get('text', '')    if isinstance(pred, dict) else '',
                'title':     pred.get('title', '')   if isinstance(pred, dict) else '',
                'category':  pred.get('category', '') if isinstance(pred, dict) else '',
                'status':    item.get('status'),
                'timestamp': item.get('timestamp'),
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

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
def get_user_by_id(user_id):
    try:
        supabase = get_supabase()
        res = supabase.table("users") \
                      .select("id, username, role") \
                      .eq("id", int(user_id)) \
                      .single() \
                      .execute()
        # ✅ เช็คว่าเป็น dict ก่อนเข้าถึง key
        if res.data and isinstance(res.data, dict):
            return {
                "id":       res.data.get("id"),
                "username": res.data.get("username"),
                "role":     res.data.get("role"),
            }
        return None
    except Exception:
        return None
def upload_image_to_supabase(file_bytes: bytes, filename: str) -> str:
    """
    อัปโหลดรูปไปยัง Supabase Storage
    คืนค่า public URL ของรูป
    """
    supabase = get_supabase()
    try:
        import uuid
        # สร้างชื่อไฟล์ unique ป้องกันซ้ำ
        ext = filename.split('.')[-1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        path = f"trending/{unique_name}"

        # อัปโหลดไฟล์
        supabase.storage.from_("news-images").upload(
            path=path,
            file=file_bytes,
            file_options={"content-type": f"image/{ext}"}
        )

        # ดึง public URL
        res = supabase.storage.from_("news-images").get_public_url(path)
        return res

    except Exception as e:
        print(f"❌ Upload Error: {e}")
        return ""

def get_feedback_stats():
    """ดึงสถิติ feedback ที่ user แนะนำว่าเป็นข่าวจริง/เท็จ"""
    supabase = get_supabase()
    try:
        res = supabase.table('feedbacks') \
                      .select('user_report, status, timestamp') \
                      .execute()
        if not res.data:
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True) \
                            .dt.tz_convert("Asia/Bangkok")
        return df
    except Exception as e:
        print(f"❌ Feedback Stats Error: {e}")
        return pd.DataFrame()
    
def update_prediction_category(prediction_id, new_category) -> bool:
    """Admin แก้ไขหมวดหมู่ของ prediction"""
    supabase = get_supabase()
    try:
        supabase.table('predictions') \
                .update({'category': new_category}) \
                .eq('id', prediction_id) \
                .execute()
        return True
    except Exception as e:
        print(f"❌ Update Category Error: {e}")
        return False