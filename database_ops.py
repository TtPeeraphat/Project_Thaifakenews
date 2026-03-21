import random
import string
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import bcrypt
import pandas as pd

from config import config
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ⚙️ 0. SUPABASE CONNECTION
# ─────────────────────────────────────────────────────────────

try:
    import streamlit as st
    _use_streamlit = True
except ImportError:
    st = None
    _use_streamlit = False

_supabase_instance: Optional[Client] = None


def _create_supabase_client() -> Client:
    raw_url = config.database.supabase_url
    raw_key = config.database.supabase_key
    clean_url = str(raw_url).strip().strip('"').strip("'") if raw_url else ""
    clean_key = str(raw_key).strip().strip('"').strip("'") if raw_key else ""
    if not clean_url or not clean_key:
        raise ValueError("Missing Supabase credentials!")
    try:
        client = create_client(clean_url, clean_key)
        logger.info("Supabase connected: %s...", clean_url[:40])
        return client
    except Exception as e:
        logger.error("Supabase connection failed: %s", e)
        raise


# สร้าง cached version เฉพาะตอนรันใน Streamlit
if _use_streamlit and st is not None:
    @st.cache_resource(show_spinner=False)
    def _get_supabase_cached() -> Client:
        return _create_supabase_client()


def get_supabase() -> Client:
    global _supabase_instance
    if _use_streamlit and st is not None:
        return _get_supabase_cached()
    if _supabase_instance is None:
        _supabase_instance = _create_supabase_client()
    return _supabase_instance


# ─────────────────────────────────────────────────────────────
# 🔐 PASSWORD HELPERS (bcrypt)
# ─────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ─────────────────────────────────────────────────────────────
# 👤 1. USER MANAGEMENT
# ─────────────────────────────────────────────────────────────

def create_user(username: str, password: str, email: str, role: str = "user") -> bool:
    supabase = get_supabase()
    pw_hash = hash_password(password)   # ✅ bcrypt แทน sha256
    payload = {
        "username":      username,
        "password_hash": pw_hash,
        "email":         email,
        "role":          role,
        "created_at":    datetime.now().isoformat(),
    }
    try:
        response = supabase.table("users").insert(payload).execute()
        return bool(response.data)
    except Exception as e:
        logger.error("Register Error: %s", e)
        return False


def authenticate_user(username: str, password: str) -> Optional[Tuple[int, str, str]]:
    try:
        supabase = get_supabase()
        # ดึง password_hash มาเปรียบฝั่ง Python (ไม่ส่ง hash ไป query ตรง ๆ)
        response = supabase.table("users") \
            .select("id, username, role, password_hash") \
            .eq("username", username) \
            .execute()

        data = response.data
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        user: dict = data[0] if isinstance(data[0], dict) else {}
        if not user:
            return None  # cast ให้ Pylance รู้จัก key
        stored_hash = user.get("password_hash")
        if not stored_hash or not isinstance(stored_hash, str):
            return None

        if not verify_password(password, stored_hash):
            logger.warning("Wrong password for: %s", username)
            return None

        uid   = int(str(user.get("id", 0)))
        uname = str(user.get("username", ""))
        urole = str(user.get("role", ""))
        logger.info("User '%s' authenticated", uname)
        return (uid, uname, urole)

    except OSError as e:
        logger.error("Network error during auth: %s", e)
        return None
    except Exception as e:
        logger.error("Auth error (%s): %s", type(e).__name__, e)
        return None


def get_user_by_id(user_id: Any):
    try:
        supabase = get_supabase()
        res = supabase.table("users") \
                      .select("id, username, role") \
                      .eq("id", int(user_id)) \
                      .single() \
                      .execute()
        if res.data and isinstance(res.data, dict):
            return {
                "id":       res.data.get("id"),
                "username": res.data.get("username"),
                "role":     res.data.get("role"),
            }
        return None
    except Exception:
        return None


def get_user_email(user_id: Any) -> str:
    supabase = get_supabase()
    try:
        response = supabase.table("users").select("email").eq("id", user_id).execute()
        if response.data and len(response.data) > 0:
            return str(response.data[0].get("email") or "")
        return ""
    except Exception as e:
        logger.error("Error fetching email: %s", e)
        return ""


def update_user_email(user_id: Any, new_email: str) -> bool:
    supabase = get_supabase()
    try:
        supabase.table("users").update({"email": new_email}).eq("id", user_id).execute()
        return True
    except Exception as e:
        logger.error("Error updating email: %s", e)
        return False


def update_user_role_status(target_user_id: Any, new_role: str, new_status: str) -> bool:
    supabase = get_supabase()
    try:
        supabase.table("users").update({"role": new_role, "status": new_status}) \
                               .eq("id", target_user_id).execute()
        return True
    except Exception as e:
        logger.error("Error updating user: %s", e)
        return False


def get_user_management_data() -> pd.DataFrame:
    supabase = get_supabase()
    try:
        res_users = supabase.table("users").select("id, username, email, role, created_at").execute()
        df_users  = pd.DataFrame(res_users.data) if res_users.data else pd.DataFrame()
        if df_users.empty:
            return pd.DataFrame()
        if "status" not in df_users.columns:
            df_users["status"] = "active"

        res_preds = supabase.table("predictions").select("user_id").execute()
        df_preds  = pd.DataFrame(res_preds.data) if res_preds.data else pd.DataFrame()

        res_logs = supabase.table("system_logs").select("user_id, timestamp").execute()
        df_logs  = pd.DataFrame(res_logs.data) if res_logs.data else pd.DataFrame()

        if not df_preds.empty and "user_id" in df_preds.columns:
            checks = df_preds.groupby("user_id").size().reset_index(name="checks")
            df_users = pd.merge(df_users, checks, left_on="id", right_on="user_id", how="left")
            df_users["checks"] = df_users["checks"].fillna(0).astype(int)
        else:
            df_users["checks"] = 0

        if not df_logs.empty and "user_id" in df_logs.columns:
            last_active = df_logs.groupby("user_id")["timestamp"].max().reset_index(name="last_active")
            df_users = pd.merge(df_users, last_active, left_on="id", right_on="user_id", how="left")
        else:
            df_users["last_active"] = None

        return df_users
    except Exception as e:
        logger.error("Error getting user management data: %s", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 🤖 2. PREDICTIONS
# ─────────────────────────────────────────────────────────────

def create_prediction(user_id, title, text, url, result, confidence, category=""):
    supabase = get_supabase()
    payload = {
        "user_id":    user_id,
        "title":      title,
        "text":       text,
        "url":        url,
        "result":     result,
        "confidence": confidence,
        "category":   category,
        "timestamp":  datetime.now().isoformat(),
    }
    try:
        response = supabase.table("predictions").insert(payload).execute()
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            item = response.data[0]
            if isinstance(item, dict):
                return int(str(item.get("id", 0)))
        return None
    except Exception as e:
        logger.error("Create Prediction Error: %s", e)
        return None


def get_user_history(user_id: Any, limit: int = 50):
    supabase = get_supabase()
    try:
        if user_id is None:
            return []
        response = supabase.table("predictions") \
            .select("id, user_id, title, text, result, confidence, category, timestamp") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(limit).execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error("Database Error: %s", e)
        return []


def update_prediction_category(prediction_id: Any, new_category: str) -> bool:
    supabase = get_supabase()
    try:
        supabase.table("predictions") \
                .update({"category": new_category}) \
                .eq("id", prediction_id).execute()
        return True
    except Exception as e:
        logger.error("Update Category Error: %s", e)
        return False


def get_model_performance_data() -> pd.DataFrame:
    supabase = get_supabase()
    try:
        res = supabase.table("predictions") \
                      .select("result, confidence, timestamp") \
                      .order("timestamp", desc=True) \
                      .limit(500).execute()
        if not res.data:
            return pd.DataFrame()

        rows = []
        for item in res.data:
            ai_pred = str(item.get("result", "")).capitalize()
            if ai_pred not in ("Real", "Fake"):
                continue
            rows.append({
                "timestamp":  item.get("timestamp"),
                "prediction": ai_pred,
                "confidence": item.get("confidence"),
                "is_correct": None,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        logger.error("Performance Data Error: %s", e)
        return pd.DataFrame()


def get_evaluated_data() -> pd.DataFrame:
    supabase = get_supabase()
    try:
        res_fb = supabase.table("feedbacks") \
                         .select("prediction_id, status") \
                         .neq("status", "pending").execute()
        if not res_fb.data:
            return pd.DataFrame()
        df_fb = pd.DataFrame(res_fb.data)

        res_pred = supabase.table("predictions").select("id, result, confidence, timestamp").execute()
        if not res_pred.data:
            return pd.DataFrame()
        df_pred = pd.DataFrame(res_pred.data)
        df_pred.rename(columns={"result": "prediction"}, inplace=True)

        df_pred["id"]            = df_pred["id"].astype(str)
        df_fb["prediction_id"]   = df_fb["prediction_id"].astype(str)

        return pd.merge(df_pred, df_fb, left_on="id", right_on="prediction_id", how="inner")
    except Exception as e:
        logger.error("Error getting evaluated data: %s", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 📝 3. FEEDBACK
# ─────────────────────────────────────────────────────────────

def save_feedback(prediction_id, user_report, comment="") -> bool:
    supabase = get_supabase()
    payload = {
        "prediction_id": prediction_id,
        "user_report":   user_report,
        "comment":       comment,
        "status":        "pending",
        "timestamp":     datetime.now().isoformat(),
    }
    try:
        supabase.table("feedbacks").insert(payload).execute()
        return True
    except Exception as e:
        logger.error("Feedback Error: %s", e)
        return False


def get_pending_feedbacks() -> List[dict]:
    supabase = get_supabase()
    try:
        res_f = supabase.table("feedbacks") \
                        .select("id, prediction_id, user_report, comment, status, timestamp") \
                        .execute()
        feedbacks = res_f.data or []
        if not feedbacks:
            return []

        fb_map = {}
        for fb in feedbacks:
            if not isinstance(fb, dict) or fb.get("id") is None:
                continue
            pid = str(fb.get("prediction_id"))
            fb_map[pid] = fb

        res_p = supabase.table("predictions").select("*").execute()
        predictions = res_p.data or []

        result_list = []
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            pid = str(pred.get("id"))
            fb  = fb_map.get(pid)
            if fb is None or fb.get("id") is None:
                continue
            result_list.append({
                "feedback_id":   fb.get("id"),
                "prediction_id": pid,
                "title":         pred.get("title", "No Title"),
                "text":          pred.get("text", ""),
                "ai_result":     pred.get("result", "Unknown"),
                "ai_confidence": pred.get("confidence", 0),
                "category":      pred.get("category", "ไม่ระบุ"),
                "url":           pred.get("url", ""),
                "user_report":   fb.get("user_report"),
                "user_comment":  fb.get("comment", ""),
                "timestamp":     pred.get("timestamp"),
                "status":        fb.get("status", "pending"),
            })
        return result_list
    except Exception as e:
        logger.error("Error fetching feedbacks: %s", e)
        return []


def update_feedback_status(feedback_id: Any, new_status: str) -> bool:
    supabase = get_supabase()
    try:
        supabase.table("feedbacks").update({"status": new_status}).eq("id", feedback_id).execute()
        return True
    except Exception as e:
        logger.error("Error updating feedback: %s", e)
        return False


def get_approved_feedbacks() -> pd.DataFrame:
    supabase = get_supabase()
    try:
        res = supabase.table("feedbacks") \
                      .select("id, status, timestamp, prediction_id, predictions!fk_prediction(text, title, result, category)") \
                      .in_("status", ["Real", "Fake"]).execute()
        if not res.data:
            return pd.DataFrame()

        rows = []
        for item in res.data:
            if not isinstance(item, dict):
                continue
            pred = item.get("predictions")
            if not pred:
                continue
            rows.append({
                "text":      pred.get("text", "")     if isinstance(pred, dict) else "",
                "title":     pred.get("title", "")    if isinstance(pred, dict) else "",
                "category":  pred.get("category", "") if isinstance(pred, dict) else "",
                "status":    item.get("status"),
                "timestamp": item.get("timestamp"),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        logger.error("Get Approved Feedbacks Error: %s", e)
        return pd.DataFrame()


def get_feedback_stats() -> pd.DataFrame:
    supabase = get_supabase()
    try:
        res = supabase.table("feedbacks").select("user_report, status, timestamp").execute()
        if not res.data:
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Bangkok")
        return df
    except Exception as e:
        logger.error("Feedback Stats Error: %s", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 📈 4. TRENDING NEWS
# ─────────────────────────────────────────────────────────────

def get_all_trending() -> pd.DataFrame:
    supabase = get_supabase()
    try:
        response = supabase.table("trending_news").select("*").order("updated_at", desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        logger.error("Get Trending Error: %s", e)
        return pd.DataFrame()


def create_trending(headline, content, label, category="ทั่วไป", image_url="", source_url="") -> bool:
    supabase = get_supabase()
    now = datetime.now().isoformat()
    payload = {
        "headline":   headline,
        "content":    content,
        "label":      label,
        "category":   category,
        "image_url":  image_url  or None,
        "source_url": source_url or None,
        "created_at": now,
        "updated_at": now,
    }
    try:
        supabase.table("trending_news").insert(payload).execute()
        return True
    except Exception as e:
        logger.error("Create Trending Error: %s", e)
        return False


def update_trending(news_id, headline, content, label, category="ทั่วไป", image_url=None, source_url=None) -> bool:
    supabase = get_supabase()
    payload: dict = {
        "headline":   headline,
        "content":    content,
        "label":      label,
        "category":   category,
        "updated_at": datetime.now().isoformat(),
    }
    if image_url:
        payload["image_url"] = image_url
    if source_url is not None:
        payload["source_url"] = source_url or None
    try:
        supabase.table("trending_news").update(payload).eq("id", news_id).execute()
        return True
    except Exception as e:
        logger.error("Update Trending Error: %s", e)
        return False


def delete_trending(news_id) -> bool:
    supabase = get_supabase()
    try:
        supabase.table("trending_news").delete().eq("id", news_id).execute()
        return True
    except Exception as e:
        logger.error("Delete Trending Error: %s", e)
        return False


def upload_image_to_supabase(file_bytes: bytes, filename: str) -> str:
    supabase = get_supabase()
    try:
        import uuid
        ext = filename.split(".")[-1].lower()
        path = f"trending/{uuid.uuid4().hex}.{ext}"
        file_data: bytes = bytes(file_bytes)
        supabase.storage.from_("news-images").upload(
            path=path,
            file=file_data,
            file_options={"content-type": f"image/{ext}"},
        )
        return supabase.storage.from_("news-images").get_public_url(path)
    except Exception as e:
        logger.error("Upload Error: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────
# 🔐 5. PASSWORD RESET OTP
# ─────────────────────────────────────────────────────────────

def send_otp_email(to_email: str) -> Tuple[bool, str]:
    supabase = get_supabase()
    try:
        user_check = supabase.table("users").select("id").eq("email", to_email).execute()
        if not (user_check.data and len(user_check.data) > 0):
            return False, "❌ ไม่พบอีเมลนี้ในระบบ"
    except Exception as e:
        return False, f"Check Email Error: {e}"

    otp = "".join(random.choices(string.digits, k=6))
    try:
        supabase.table("users").update({"reset_token": otp}).eq("email", to_email).execute()
    except Exception as e:
        return False, f"Database Error: {e}"

    sender_email    = str(config.email.sender_email).strip()
    sender_password = str(config.email.sender_password).strip()

    msg = MIMEText(
    f"รหัส OTP ของคุณคือ: {otp}\n\n"
    "กรุณานำรหัสนี้ไปกรอกในหน้าเว็บเพื่อตั้งรหัสผ่านใหม่",
    "plain",
    "utf-8"
    )
    msg["Subject"] = "🔑 รหัสยืนยันการเปลี่ยนรหัสผ่าน (Thai Fake News)"
    msg["From"]    = sender_email
    msg["To"]      = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True, "✅ ส่งรหัส OTP ไปที่อีเมลแล้ว"
    except Exception as e:
        return False, f"❌ ส่งอีเมลไม่สำเร็จ: {e}"


def verify_otp_and_reset(email: str, otp: str, new_password: str) -> Tuple[bool, str]:
    supabase = get_supabase()
    try:
        response = supabase.table("users") \
            .select("id") \
            .eq("email", email) \
            .eq("reset_token", otp) \
            .execute()
        if not response.data or len(response.data) == 0:
            return False, "❌ รหัส OTP ไม่ถูกต้อง หรือหมดอายุ"

        # ✅ bcrypt แทน sha256
        new_pw_hash = hash_password(new_password)
        supabase.table("users").update({
            "password_hash": new_pw_hash,
            "reset_token":   None,
        }).eq("email", email).execute()

        return True, "✅ เปลี่ยนรหัสผ่านสำเร็จ! กรุณาล็อกอินใหม่"
    except Exception as e:
        return False, f"Reset Error: {e}"


# ─────────────────────────────────────────────────────────────
# 📝 6. SYSTEM LOGGING
# ─────────────────────────────────────────────────────────────

def log_system_event(user_id: Any, action: str, details: str, level: str = "INFO") -> bool:
    try:
        supabase = get_supabase()
        supabase.table("system_logs").insert({
            "user_id":   str(user_id) if user_id else "System",
            "action":    str(action).upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details":   str(details),
            "level":     str(level).upper(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("Log skipped (%s): %s", action, e)
        return False


def get_system_logs(limit: int = 50) -> List[Tuple]:
    supabase = get_supabase()
    try:
        response = supabase.table("system_logs") \
            .select("timestamp, action, details, level, user_id") \
            .order("timestamp", desc=True) \
            .limit(limit).execute()
        rows = []
        if isinstance(response.data, list):
            for item in response.data:
                if isinstance(item, dict):
                    rows.append((
                        str(item.get("timestamp", "-")),
                        str(item.get("user_id", "-")),
                        str(item.get("action", "-")),
                        str(item.get("details", "-")),
                        str(item.get("level", "INFO")),
                    ))
        return rows
    except Exception:
        return []


def get_system_analytics_data() -> dict:
    supabase = get_supabase()
    try:
        res_users = supabase.table("users").select("id", count="exact").execute()
        total_users = res_users.count if hasattr(res_users, "count") and res_users.count else 0

        res_preds = supabase.table("predictions").select("id, result, timestamp").execute()
        df_preds  = pd.DataFrame(res_preds.data) if res_preds.data else pd.DataFrame()

        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        res_logs = supabase.table("system_logs") \
            .select("timestamp, user_id") \
            .gte("timestamp", seven_days_ago).execute()
        df_logs = pd.DataFrame(res_logs.data) if res_logs.data else pd.DataFrame()

        return {"total_users": total_users, "df_preds": df_preds, "df_logs": df_logs}
    except Exception as e:
        logger.error("Error getting analytics data: %s", e)
        return {"total_users": 0, "df_preds": pd.DataFrame(), "df_logs": pd.DataFrame()}


def get_dashboard_kpi() -> dict:
    supabase = get_supabase()
    now_utc   = datetime.now(timezone.utc)
    last_24h  = (now_utc - timedelta(hours=24)).isoformat()
    stats     = {"checks_today": 0, "active_users": 0, "accuracy": 0.0, "feedback_total": 0}
    try:
        res_checks = supabase.table("predictions").select("id", count="exact").gte("timestamp", last_24h).execute()
        stats["checks_today"] = res_checks.count or 0

        res_users = supabase.table("system_logs").select("user_id").gte("timestamp", last_24h).execute()
        logs_data = res_users.data or []
        stats["active_users"] = len({str(r.get("user_id")) for r in logs_data if r.get("user_id")})

        res_fb = supabase.table("feedbacks") \
            .select("user_report, status, predictions!fk_prediction(result)") \
            .neq("status", "pending").execute()
        fb_list = res_fb.data or []
        stats["feedback_total"] = len(fb_list)
        if stats["feedback_total"] > 0:
            correct = sum(1 for i in fb_list if
                          str(i.get("user_report", "")).lower() == "correct" or
                          str(i.get("status", "")).lower() == "verified_correct")
            stats["accuracy"] = round(correct / stats["feedback_total"] * 100, 1)
        return stats
    except Exception as e:
        logger.error("KPI Error: %s", e)
        return stats