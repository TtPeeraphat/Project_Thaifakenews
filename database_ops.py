import sqlite3
import hashlib
from datetime import datetime

# กำหนดชื่อ Database ชื่อเดียวเพื่อไม่ให้สับสน
DB_NAME = "core_data.db"

# ==========================================
# 0. System & Connection
# ==========================================

def create_connection():
    """สร้างการเชื่อมต่อกับฐานข้อมูล SQLite"""
    conn = None
    try:
        # check_same_thread=False จำเป็นมากสำหรับ Streamlit
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to DB: {e}")
        return None

def init_db():
    """สร้างตารางทั้งหมดถ้ายังไม่มี (กัน Error: no such table)"""
    conn = create_connection()
    if conn:
        c = conn.cursor()
        
        # 1. ตาราง Users
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            created_at DATETIME
        )''')

        # 2. ตาราง Predictions (เปลี่ยนชื่อคอลัมน์ให้ตรงกับ Admin Dashboard)
        c.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            text TEXT,
            url TEXT,
            result TEXT,
            confidence REAL,
            timestamp DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')

        # 3. ตาราง Feedbacks
        c.execute('''CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            user_report TEXT,
            comment TEXT,
            status TEXT DEFAULT 'pending',
            timestamp DATETIME,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )''')

        # 4. ตาราง Trending News
        c.execute('''CREATE TABLE IF NOT EXISTS trending_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT,
            content TEXT,
            label TEXT,
            updated_at DATETIME
        )''')

        # 5. ตาราง Logs
        c.execute('''CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            timestamp DATETIME
        )''')

        conn.commit()
        conn.close()
        print("✅ Database & Tables initialized successfully.")

# ==========================================
# 1. Users (Login/Register)
# ==========================================

def create_user(username, password, email, role='user'):
    conn = create_connection()
    if not conn: return False
    
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
                  (username, pw_hash, email, role, datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print(f"Register Error: {e}")
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = create_connection()
    if not conn: return None
    
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?", (username, pw_hash))
    user = c.fetchone() # คืนค่า (id, username, role)
    conn.close()
    return user 

# ==========================================
# 2. Predictions (Check News)
# ==========================================

def create_prediction(user_id, title, text, url, result, confidence):
    conn = create_connection()
    if not conn: return None
    
    try:
        c = conn.cursor()
        # ใช้ชื่อคอลัมน์ title, text ให้ตรงกับตอนดึงข้อมูล
        c.execute("""INSERT INTO predictions (user_id, title, text, url, result, confidence, timestamp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, title, text, url, result, confidence, datetime.now()))
        conn.commit()
        return c.lastrowid # คืนค่า ID ล่าสุด
    except Exception as e:
        print(f"Create Prediction Error: {e}")
        return None
    finally:
        conn.close()

def read_user_history(user_id):
    conn = create_connection()
    if not conn: return []
    
    c = conn.cursor()
    c.execute("SELECT timestamp, title, result, confidence FROM predictions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# 3. Feedbacks
# ==========================================

def save_feedback(prediction_id, user_report, comment=""):
    conn = create_connection()
    if not conn: return False
    
    try:
        c = conn.cursor()
        c.execute("INSERT INTO feedbacks (prediction_id, user_report, comment, timestamp) VALUES (?, ?, ?, ?)",
                  (prediction_id, user_report, comment, datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print(f"Feedback Error: {e}")
        return False
    finally:
        conn.close()

def read_all_feedbacks():
    conn = create_connection()
    if not conn: return []
    
    c = conn.cursor()
    c.execute("""SELECT f.id, p.title, f.user_report, f.comment, f.status 
                 FROM feedbacks f JOIN predictions p ON f.prediction_id = p.id
                 ORDER BY f.timestamp DESC""")
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# 4. Trending News
# ==========================================

def create_trending(headline, content, label):
    conn = create_connection()
    if not conn: return
    c = conn.cursor()
    c.execute("INSERT INTO trending_news (headline, content, label, updated_at) VALUES (?, ?, ?, ?)",
              (headline, content, label, datetime.now()))
    conn.commit()
    conn.close()

def get_all_trending():
    conn = create_connection()
    if not conn: return []
    c = conn.cursor()
    c.execute("SELECT id, headline, content, label, updated_at FROM trending_news ORDER BY updated_at DESC")
    data = c.fetchall()
    conn.close()
    return data

def delete_trending(news_id):
    conn = create_connection()
    if not conn: return
    c = conn.cursor()
    c.execute("DELETE FROM trending_news WHERE id=?", (news_id,))
    conn.commit()
    conn.close()

# ==========================================
# 5. Logs & Admin Charts
# ==========================================

def create_log(user_id, action):
    conn = create_connection()
    if not conn: return
    c = conn.cursor()
    c.execute("INSERT INTO system_logs (user_id, action, timestamp) VALUES (?, ?, ?)",
              (user_id, action, datetime.now()))
    conn.commit()
    conn.close()

def read_logs_for_chart():
    conn = create_connection()
    if not conn: return []
    c = conn.cursor()
    c.execute("SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM system_logs GROUP BY hour")
    data = c.fetchall()
    conn.close()
    return data

def read_all_predictions():
    """สำหรับ Admin Dashboard"""
    conn = create_connection()
    if not conn: return []
    
    c = conn.cursor()
    # ดึง username จากตาราง users มาแสดงด้วย
    c.execute('''SELECT p.id, u.username, p.title, p.text, p.result, p.confidence, p.timestamp 
                 FROM predictions p
                 JOIN users u ON p.user_id = u.id
                 ORDER BY p.timestamp DESC''')
    data = c.fetchall()
    conn.close()
    return data

def read_all_predictions_limit(limit=10):
    """สำหรับ Admin Dashboard (ดูตัวอย่างล่าสุด)"""
    conn = create_connection()
    if not conn: return []
    
    c = conn.cursor()
    c.execute(f'''SELECT p.id, u.username, p.title, p.text, p.result, p.confidence, p.timestamp 
                  FROM predictions p
                  JOIN users u ON p.user_id = u.id
                  ORDER BY p.timestamp DESC LIMIT ?''', (limit,))
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# RUN SCRIPT (สร้างตารางเมื่อรันครั้งแรก)
# ==========================================
if __name__ == "__main__":
    init_db()