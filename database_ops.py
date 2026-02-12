import sqlite3
import hashlib
from datetime import datetime

DB_NAME = "core_data.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# ==========================================
# 1. CRUD สำหรับ Users (สมัครสมาชิก/Login)
# ==========================================

def create_user(username, password, email, role='user'):
    conn = get_connection()
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password_hash, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
                  (username, pw_hash, email, role, datetime.now()))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?", (username, pw_hash))
    user = c.fetchone()
    conn.close()
    return user # คืนค่า (id, username, role) ถ้าสำเร็จ

# ==========================================
# 2. CRUD สำหรับ Predictions (ประวัติการเช็คข่าว)
# ==========================================

def create_prediction(user_id, title, text, url, result, confidence):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("""INSERT INTO predictions (user_id, news_title, news_text, source_url, result, confidence, timestamp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, title, text, url, result, confidence, datetime.now()))
        conn.commit()
        
        # --- สิ่งที่เพิ่ม: คืนค่า ID ของรายการล่าสุด ---
        new_id = c.lastrowid 
        return new_id 
    except Exception as e:
        print(f"Error create_prediction: {e}")
        return None
    finally:
        conn.close()

def read_user_history(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT timestamp, news_title, result, confidence FROM predictions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# 3. CRUD สำหรับ Feedbacks (Admin รีวิวคำร้อง)
# ==========================================

def create_feedback(prediction_id, report_type, comment):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO feedbacks (prediction_id, user_report, comment, timestamp) VALUES (?, ?, ?, ?)",
              (prediction_id, report_type, comment, datetime.now()))
    conn.commit()
    conn.close()

def read_all_feedbacks():
    conn = get_connection()
    c = conn.cursor()
    # Join ตารางเพื่อให้ Admin เห็นว่า Feedback นี้มาจากข่าวไหน
    c.execute("""SELECT f.id, p.news_title, f.user_report, f.comment, f.status 
                 FROM feedbacks f JOIN predictions p ON f.prediction_id = p.id""")
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# 4. CRUD สำหรับ Trending News (จัดการข่าวเด่น)
# ================================

def create_trending(headline, content, label):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO trending_news (headline, content, label, updated_at) VALUES (?, ?, ?, ?)",
              (headline, content, label, datetime.now()))
    conn.commit()
    conn.close()

def update_trending(news_id, headline, content, label):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE trending_news SET headline=?, content=?, label=?, updated_at=? WHERE id=?",
              (headline, content, label, datetime.now(), news_id))
    conn.commit()
    conn.close()

def delete_trending(news_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM trending_news WHERE id=?", (news_id,))
    conn.commit()
    conn.close()
def get_all_trending():
    """ดึงข่าวประกาศทั้งหมด เรียงจากใหม่ไปเก่า"""
    conn = get_connection()
    c = conn.cursor()
    # ดึง id, พาดหัว, เนื้อหา, ป้ายกำกับ, เวลา
    c.execute("SELECT id, headline, content, label, updated_at FROM trending_news ORDER BY updated_at DESC")
    data = c.fetchall()
    conn.close()
    return data
# ==========================================
# 5. CRUD สำหรับ Analytics (Logs)
# ==========================================

def create_log(user_id, action):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO system_logs (user_id, action, timestamp) VALUES (?, ?, ?)",
              (user_id, action, datetime.now()))
    conn.commit()
    conn.close()

def read_logs_for_chart():
    conn = get_connection()
    c = conn.cursor()
    # ดึงเวลามาจัดกลุ่มเพื่อนทำกราฟ Activity by Time
    c.execute("SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM system_logs GROUP BY hour")
    data = c.fetchall()
    conn.close()
    return data

# ฟังก์ชัน save_feedback 
def save_feedback(prediction_id, user_report, comment=""):
    conn = get_connection()
    c = conn.cursor()
    try:
        # user_report: ส่งค่า 'Correct' (ถูก) หรือ 'Incorrect' (ผิด)
        c.execute("INSERT INTO feedbacks (prediction_id, user_report, comment, timestamp) VALUES (?, ?, ?, ?)",
                  (prediction_id, user_report, comment, datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error save_feedback: {e}")
        return False
    finally:
        conn.close()