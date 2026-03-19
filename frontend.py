import email
import sys
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import altair as alt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from datetime import datetime, timedelta
import asyncio
from streamlit_cookies_controller import CookieController
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo 
import database_ops as db
import ai_engine as ai
from scraper_ops import get_content_from_url


if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

cookie_controller = CookieController()
TZ_BKK = ZoneInfo("Asia/Bangkok")     # UTC+7

def now_bkk() -> datetime:
    """Current time in Bangkok (GMT+7), timezone-aware."""
    return datetime.now(tz=TZ_BKK)

# 1. ฟังก์ชันล้างข้อความ (เอาไว้ด้านบนสุด หรือก่อนถึง if menu...)
def clear_text():
    st.session_state['input_text'] = ""

# 2. เช็คว่ามีค่าในระบบความจำหรือยัง ป้องกัน Error ตอนเปิดแอปครั้งแรก
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

def clear_url():
    st.session_state['input_url'] = ""
if 'input_url' not in st.session_state:
    st.session_state['input_url'] = ""
# ─────────────────────────────────────────────
# Cookie / session helpers
# ─────────────────────────────────────────────
def check_persistent_login():
    all_cookies = cookie_controller.getAll()
    if "cookie_wait" not in st.session_state:
        st.session_state["cookie_wait"] = 0
    if all_cookies is None and st.session_state["cookie_wait"] < 3:
        st.session_state["cookie_wait"] += 1
        time.sleep(0.3)
        st.rerun()
        return
    if isinstance(all_cookies, dict):
        if st.session_state.get('do_logout'):
            for k in ('saved_user_id',):  # ✅ ลบแค่ user_id พอ ไม่มี saved_role แล้ว
                try: cookie_controller.remove(k)
                except KeyError: pass
            st.session_state['logged_in'] = False
            st.session_state['do_logout'] = False
            for key in ['user_id', 'role', 'username', 'need_to_save_cookie']:
                if key in st.session_state:
                    del st.session_state[key]
            return
        if st.session_state.get('need_to_save_cookie'):
            cookie_controller.set('saved_user_id', str(st.session_state['user_id']), max_age=86400)
            # ✅ ลบออก: ไม่เก็บ role ใน cookie อีกต่อไป
            st.session_state['need_to_save_cookie'] = False
        saved_user_id = all_cookies.get('saved_user_id')
        if saved_user_id and not st.session_state.get('logged_in'):
            # ✅ ดึง role และข้อมูลจาก DB แทนการอ่านจาก cookie
            user_data = db.get_user_by_id(saved_user_id)
            if user_data:
                st.session_state['user_id']   = user_data['id']
                st.session_state['role']      = user_data['role']      # จาก DB ✅
                st.session_state['username']  = user_data['username']  # จาก DB ✅
                st.session_state['logged_in'] = True
                st.session_state["cookie_wait"] = 0
                st.rerun()
            else:
                # ✅ ถ้าไม่เจอ user ใน DB ให้ลบ cookie ทิ้งเลย (อาจถูกลบออกจากระบบ)
                try: cookie_controller.remove('saved_user_id')
                except KeyError: pass

check_persistent_login()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrueCheck AI — Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&display=swap');
/* ══════════════════════════════
   FORCE LIGHT MODE
══════════════════════════════ */
html {
  color-scheme: light only !important;
}
            
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stMain"],
section.main,
.main,
.block-container,
[class*="css"] {
  background-color: #EEF2F8 !important;
  color: #1E293B !important;
  color-scheme: light !important;
}
/* Neutralise any dark-mode media query Streamlit injects */
@media (prefers-color-scheme: dark) {
  html { color-scheme: light only !important; }
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMainBlockContainer"],
  .main, .block-container, [class*="css"] {
    background-color: #EEF2F8 !important;
    color: #1E293B !important;
  }
}
:root {
  --blue-800:  #0D47A1;
  --blue-700:  #1148A8;
  --blue-600:  #1565C0;
  --blue-500:  #1E88E5;
  --blue-100:  #DBEAFE;
  --blue-50:   #EFF6FF;
  --teal-600:  #00838F;
  --teal-500:  #0097A7;
  --teal-50:   #E0F7FA;
  --green-700: #166534;
  --green-600: #16A34A;
  --green-50:  #F0FDF4;
  --green-100: #DCFCE7;
  --red-700:   #991B1B;
  --red-600:   #DC2626;
  --red-50:    #FFF5F5;
  --red-100:   #FEE2E2;
  --amber-700: #92400E;
  --amber-500: #F59E0B;
  --amber-50:  #FFFBEB;
  --amber-100: #FEF3C7;
  --grey-900:  #0F172A;
  --grey-800:  #1E293B;
  --grey-700:  #334155;
  --grey-600:  #475569;
  --grey-500:  #64748B;
  --grey-400:  #94A3B8;
  --grey-300:  #CBD5E1;
  --grey-200:  #E2E8F0;
  --grey-100:  #F1F5F9;
  --grey-50:   #F8FAFC;
  --surface:   #FFFFFF;
  --bg:        #EEF2F8;
  --border:    #E2E8F0;
  --shadow-xs: 0 1px 2px rgba(0,0,0,0.06);
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 16px rgba(17,72,168,0.10);
  --shadow-lg: 0 8px 32px rgba(17,72,168,0.14);
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-xl: 22px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans Thai', sans-serif !important;  /* เปลี่ยนจาก Inter */
  color: var(--grey-800) !important;
  background: var(--bg) !important;
  -webkit-font-smoothing: antialiased !important;
}

h1,h2,h3,h4,h5,h6 {
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  color: var(--grey-900) !important;
  letter-spacing: -0.25px !important;
}
h1 { font-size:1.85rem !important; font-weight:800 !important; line-height:1.25 !important; }
h2 { font-size:1.35rem !important; font-weight:700 !important; }
h3 { font-size:1.1rem  !important; font-weight:700 !important; }
h4 { font-size:0.97rem !important; font-weight:600 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility:hidden !important; }
[data-testid="stDecoration"]{ display:none !important; }
[data-testid="stToolbar"]   { display:none !important; }

/* ── Canvas ── */
.main .block-container {
  background: var(--bg) !important;
  padding: 2rem 2.5rem 3.5rem !important;
  max-width: 1180px !important;
}

/* ══════════════════════════════
   SIDEBAR
══════════════════════════════ */
            /* ── Hide sidebar collapse / expand toggle ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  display: none !important;
}
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility:hidden !important; }
[data-testid="stDecoration"]        { display:none !important; }
[data-testid="stToolbar"]           { display:none !important; }

/* ── Hide sidebar collapse / expand toggle ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"]    { display:none !important; }
                        
[data-testid="stSidebar"] {
  background: #0A1929 !important;
  border-right: 1px solid rgba(255,255,255,0.05) !important;
  min-width: 256px !important;
  max-width: 256px !important;
}
[data-testid="stSidebar"] > div:first-child { padding:0 !important; }
[data-testid="stSidebar"] * { color:#fff !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.07) !important; }

/* Sidebar nav buttons — all states */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button:active {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: rgba(255,255,255,0.62) !important;
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 9px 14px !important;
  border-radius: var(--r-sm) !important;
  margin-bottom: 1px !important;
  width: 100% !important;
  outline: none !important;
  transition: all 0.14s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,0.07) !important;
  background-color: rgba(255,255,255,0.07) !important;
  color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: rgba(30,136,229,0.20) !important;
  background-color: rgba(30,136,229,0.20) !important;
  color: #93C5FD !important;
  font-weight: 600 !important;
  border-left: 2px solid #3B82F6 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: rgba(30,136,229,0.28) !important;
  background-color: rgba(30,136,229,0.28) !important;
  box-shadow: none !important;
  transform: none !important;
}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div { color: inherit !important; }

[data-testid="stSidebar"] div[data-testid="stContainer"],
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ══════════════════════════════
   MAIN COMPONENTS
══════════════════════════════ */

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 20px 22px !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--grey-500) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.7px !important;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1.9rem !important;
  color: var(--grey-900) !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] svg { display:none !important; }

/* Primary buttons — main */
.main .stButton > button[kind="primary"],
[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg,var(--blue-500) 0%,var(--blue-800) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-sm) !important;
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  padding: 11px 24px !important;
  box-shadow: 0 2px 8px rgba(13,71,161,0.28) !important;
  transition: all 0.17s ease !important;
}
.main .stButton > button[kind="primary"]:hover,
[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 18px rgba(13,71,161,0.36) !important;
}

/* Secondary buttons — main */
.main .stButton > button[kind="secondary"],
.main .stButton > button:not([kind="primary"]),
[data-testid="stMainBlockContainer"] .stButton > button[kind="secondary"],
[data-testid="stMainBlockContainer"] .stButton > button:not([kind="primary"]) {
  background: var(--surface) !important;
  color: var(--blue-600) !important;
  border: 1.5px solid var(--blue-100) !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  transition: all 0.15s ease !important;
}
.main .stButton > button[kind="secondary"]:hover,
[data-testid="stMainBlockContainer"] .stButton > button[kind="secondary"]:hover {
  background: var(--blue-50) !important;
  border-color: var(--blue-500) !important;
}

/* Text inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--surface) !important;
  border: 1.5px solid var(--grey-200) !important;
  border-radius: var(--r-sm) !important;
  color: var(--grey-900) !important;
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  font-size: 0.92rem !important;
  padding: 10px 14px !important;
  transition: border 0.15s, box-shadow 0.15s !important;
  box-shadow: var(--shadow-xs) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--blue-500) !important;
  box-shadow: 0 0 0 3px rgba(30,136,229,0.13) !important;
  outline: none !important;
}
.stTextInput label, .stTextArea label {
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  color: var(--grey-700) !important;
}

/* Selectbox */
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1.5px solid var(--grey-200) !important;
  border-radius: var(--r-sm) !important;
  color: var(--grey-900) !important;
  box-shadow: var(--shadow-xs) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 2px solid var(--border) !important;
  gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--grey-500) !important;
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  font-weight: 600 !important; font-size: 0.9rem !important;
  padding: 12px 20px !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
  color: var(--blue-600) !important;
  border-bottom: 2px solid var(--blue-600) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* Expander */
            [data-testid="stExpander"] .stButton > button {
  background: #FFFFFF !important;
  color: #1E293B !important;
  border: 1.5px solid #CBD5E1 !important;
  -webkit-text-fill-color: #1E293B !important;
}
[data-testid="stExpander"] .stButton > button:hover {
  background: #F1F5F9 !important;
  border-color: #1565C0 !important;
}
[data-testid="stExpander"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg,#EF4444,#991B1B) !important;
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  border: none !important;
}
.streamlit-expanderHeader,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > details > summary {
  background: #1E293B !important;
  border: 1px solid #334155 !important;
  border-radius: var(--r-sm) !important;
  color: #F1F5F9 !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
}
/* Kill Streamlit's red/colored focus ring on expander */
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] > details > summary:focus {
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(30,136,229,0.35) !important;
  border-color: #3B82F6 !important;
}
.streamlit-expanderContent,
[data-testid="stExpander"] > details > div {
  background: #1E293B !important;
  border: 1px solid #334155 !important;
  border-top: none !important;
  border-radius: 0 0 var(--r-sm) var(--r-sm) !important;
  color: #E2E8F0 !important;
}
/* All text inside expander content readable on dark bg */
[data-testid="stExpander"] p,
[data-testid="stExpander"] label,
[data-testid="stExpander"] span {
  color: #CBD5E1 !important;
}

/* Alerts */
div[data-testid="stInfo"]    { background:var(--blue-50)  !important; border-left:3px solid var(--blue-500)  !important; border-radius:var(--r-sm) !important; color:var(--blue-700)  !important; }
div[data-testid="stSuccess"] { background:var(--green-50) !important; border-left:3px solid var(--green-600) !important; border-radius:var(--r-sm) !important; color:var(--green-700) !important; }
div[data-testid="stWarning"] { background:var(--amber-50) !important; border-left:3px solid var(--amber-500) !important; border-radius:var(--r-sm) !important; color:var(--amber-700) !important; }
div[data-testid="stError"]   { background:var(--red-50)   !important; border-left:3px solid var(--red-600)   !important; border-radius:var(--r-sm) !important; color:var(--red-700)   !important; }

/* Progress */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg,var(--blue-500),var(--teal-500)) !important;
  border-radius: 99px !important;
}
.stProgress > div > div > div {
  background: var(--grey-200) !important;
  border-radius: 99px !important; height: 8px !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
  border-radius: var(--r-md) !important; overflow: hidden !important;
  border: 1px solid var(--border) !important; box-shadow: var(--shadow-sm) !important;
}

hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }
.stCaption, small, caption { color: var(--grey-500) !important; font-size:0.82rem !important; }
.stRadio > div { gap: 12px !important; }
.stRadio label { font-weight: 500 !important; color: var(--grey-700) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color:var(--grey-800) !important; font-weight:500 !important; }

.stDownloadButton > button {
  background: linear-gradient(135deg,var(--teal-500),var(--teal-600)) !important;
  color: white !important; border: none !important;
  border-radius: var(--r-sm) !important; font-weight: 700 !important;
  box-shadow: 0 2px 8px rgba(0,150,167,0.22) !important;
}
.stDownloadButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(0,150,167,0.32) !important;
}

[data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(135deg,var(--blue-500) 0%,var(--blue-800) 100%) !important;
  color: #fff !important; border: none !important;
  border-radius: var(--r-sm) !important;
  font-family: 'IBM Plex Sans Thai', sans-serif !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 8px rgba(13,71,161,0.28) !important;
}

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--grey-300); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:var(--grey-400); }
.stSpinner > div { border-top-color: var(--blue-500) !important; }
</style>    
<style>
/* ── Mobile sidebar toggle ── */
@media (max-width: 768px) {
  [data-testid="stSidebar"] {
    transform: translateX(-100%) !important;
    transition: transform 0.3s ease !important;
    position: fixed !important;
    z-index: 999 !important;
    height: 100vh !important;
  }
  [data-testid="stSidebar"].open {
    transform: translateX(0) !important;
  }
  .mobile-menu-btn {
    display: flex !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 1000 !important;
    background: #1148A8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 1.2rem !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
  }
}
@media (min-width: 769px) {
  .mobile-menu-btn { display: none !important; }
}
</style>

<button class="mobile-menu-btn" onclick="
  const sb = document.querySelector('[data-testid=stSidebar]');
  sb.classList.toggle('open');
">☰</button>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  UI COMPONENT HELPERS
# ═══════════════════════════════════════════════════════

def page_header(icon: str, title: str, subtitle: str = ""):
    sub = f"<p style='color:#64748B;font-size:0.9rem;margin:4px 0 0;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style="padding-bottom:20px;margin-bottom:24px;
                border-bottom:1px solid #E2E8F0;">
      <h1 style="margin:0 !important;">{icon}&nbsp;{title}</h1>
      {sub}
    </div>""", unsafe_allow_html=True)


def section_title(text: str, caption: str = ""):
    cap = f"<div style='font-size:0.8rem;color:#94A3B8;margin-top:2px;'>{caption}</div>" if caption else ""
    st.markdown(f"""
    <div style="margin:22px 0 12px;">
      <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:1rem;
                  font-weight:700;color:#1E293B;">{text}</div>
      {cap}
    </div>""", unsafe_allow_html=True)


def kpi_card(icon: str, label: str, value, delta: str = "", delta_ok: bool = True):
    dc = "#16A34A" if delta_ok else "#DC2626"
    d_html = f"<div style='font-size:0.78rem;font-weight:600;color:{dc};margin-top:5px;'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;
                padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,0.06);height:100%;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">
        <span style="font-size:1.2rem;">{icon}</span>
        <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.7px;color:#64748B;">{label}</span>
      </div>
      <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:1.85rem;
                  font-weight:800;color:#0F172A;line-height:1.1;">{value}</div>
      {d_html}
    </div>""", unsafe_allow_html=True)


def status_badge(label: str) -> str:
    cfg = {
        "Real":       ("#00FF5943","#000000","✓"),
        "Fake":       ("#FF0000","#991B1B","✕"),
        "Unverified": ("#FEF3C7","#92400E","?"),
    }.get(label, ("#000000","#000000","·"))
    return (f"<span style='display:inline-flex;align-items:center;gap:3px;"
            f"background:{cfg[0]};color:{cfg[1]};font-size:0.72rem;font-weight:800;"
            f"padding:3px 10px;border-radius:99px;text-transform:uppercase;"
            f"letter-spacing:0.4px;'>{cfg[2]} {label}</span>")


def time_ago(ts):
    try:
        if isinstance(ts, str):
            dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        else:
            dt = ts

        # Make dt timezone-aware in GMT+7 if it's naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_BKK)

        s = (now_bkk() - dt).total_seconds()

        if s < 60:    return "Just now"
        if s < 3600:  return f"{int(s // 60)}m ago"
        if s < 86400: return f"{int(s // 3600)}h ago"
        return f"{int(s // 86400)}d ago"
    except:
        return str(ts)


def card_wrap(content_fn, *args, **kwargs):
    """Renders content inside a white card."""
    st.markdown('<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;'
                'padding:22px 24px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:16px;">', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  ADMIN: Model Performance
# ═══════════════════════════════════════════════════════
def show_admin_dashboard_enhanced():
    stats   = db.get_dashboard_kpi()
    df_perf = db.get_model_performance_data()
   
    c1,c2,c3 = st.columns(3)
    with c1: kpi_card("🎯","Accuracy (Verified)", f"{stats['accuracy']}%",
                       f"{stats['feedback_total']} verified samples", stats['accuracy']>=70)
    with c2: kpi_card("🔍","Checks Today",        f"{stats['checks_today']:,}")
    with c3: kpi_card("👥","Active Users",         f"{stats['active_users']:,}")

    section_title("Cumulative Model Accuracy Over Time")
    if not df_perf.empty:
    # ✅ เช็คว่ามี column timestamp จริงก่อนแปลง
        if 'timestamp' in df_perf.columns:
            df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'], utc=True).dt.tz_convert("Asia/Bangkok")
        df_perf = df_perf.sort_values('timestamp')
        df_perf['cumulative_accuracy'] = df_perf['is_correct'].expanding().mean()*100
        fig = px.line(df_perf, x='timestamp', y='cumulative_accuracy',
                    line_shape='spline', color_discrete_sequence=['#1565C0'])
        fig.update_layout(
            yaxis_range=[0,100],
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#1E293B',        # ✅ เข้มขึ้น
            font=dict(size=12),
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="",
            yaxis_title="Accuracy (%)",
            xaxis=dict(tickfont=dict(color='#1E293B', size=11)),
            yaxis=dict(tickfont=dict(color='#1E293B', size=11),
                    title_font=dict(color='#1E293B')),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("ยังไม่มีข้อมูลที่ผ่านการตรวจสอบ")


def show_model_performance():
    df_all = db.get_model_performance_data()
    section_title("AI Activity Monitor","ข้อมูลการทำนายทั้งหมด (real-time)")

    if df_all.empty:
        st.info("ยังไม่มีข้อมูลการทำนาย")
        return

    total = len(df_all)
    if 'confidence' in df_all.columns:
        df_all['confidence'] = pd.to_numeric(df_all['confidence'], errors='coerce')
    avg_c = df_all['confidence'].mean()
    fc = len(df_all[df_all['prediction']=='Fake'])
    rc = len(df_all[df_all['prediction']=='Real'])

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("📄","Total Scanned",  f"{total:,}")
    with c2: kpi_card("🧠","Avg Confidence", f"{avg_c:.1f}%")
    with c3: kpi_card("🚨","Detected FAKE",  f"{fc:,}", f"{fc/total*100:.1f}% of scans", False)
    with c4: kpi_card("✅","Detected REAL",  f"{rc:,}", f"{rc/total*100:.1f}% of scans", True)

    st.markdown("<hr style='margin:28px 0 20px;'>", unsafe_allow_html=True)
    section_title("Accuracy Evaluation","เฉพาะรายการที่ Admin ตรวจสอบแล้ว")

    df_ev = db.get_evaluated_data()
    valid = False
    if not df_ev.empty and 'status' in df_ev.columns and 'prediction' in df_ev.columns:
        df_ev['status'] = df_ev['status'].astype(str).str.strip()
        df_ev = df_ev[df_ev['status'].isin(['Real','Fake'])]
        if not df_ev.empty: valid = True

    if not valid:
        st.info("ส่วนนี้จะแสดงเมื่อ Admin Review ข้อมูลในหน้า Feedback แล้ว")
        return

    st.success(f"คำนวณจาก {len(df_ev)} รายการที่ผ่านการตรวจสอบ")
    try:
        acc  = accuracy_score(df_ev['status'], df_ev['prediction'])
        prec_fake = precision_score(df_ev['status'], df_ev['prediction'], pos_label='Fake', zero_division=0)
        rec_fake  = recall_score(df_ev['status'],    df_ev['prediction'], pos_label='Fake', zero_division=0)
        f1_fake   = f1_score(df_ev['status'],        df_ev['prediction'], pos_label='Fake', zero_division=0)
        prec_real = precision_score(df_ev['status'], df_ev['prediction'], pos_label='Real', zero_division=0)
        rec_real  = recall_score(df_ev['status'],    df_ev['prediction'], pos_label='Real', zero_division=0)

        sp = lambda v: float(max(0.0, min(1.0, v)))

    # ── แถว 1: ภาพรวม ──
        section_title("📊 ภาพรวม")
        c1, c2 = st.columns(2)
        with c1: st.metric("Overall Accuracy", f"{acc*100:.1f}%"); st.progress(sp(acc))
        with c2: st.metric("F1 Score (Fake)", f"{f1_fake*100:.1f}%"); st.progress(sp(f1_fake))

    # ── แถว 2: Fake ──
        section_title("🚨 Fake News Detection")
        c1, c2 = st.columns(2)
        with c1: st.metric("Precision (Fake)", f"{prec_fake*100:.1f}%",
                        help="ที่ทายว่า Fake จริง Fake กี่%"); st.progress(sp(prec_fake))
        with c2: st.metric("Recall (Fake)", f"{rec_fake*100:.1f}%",
                        help="Fake จริงๆ จับได้กี่%"); st.progress(sp(rec_fake))

    # ── แถว 3: Real ──
        section_title("✅ Real News Detection")
        c1, c2 = st.columns(2)
        with c1: st.metric("Precision (Real)", f"{prec_real*100:.1f}%",
                        help="ที่ทายว่า Real จริง Real กี่%"); st.progress(sp(prec_real))
        with c2: st.metric("Recall (Real)", f"{rec_real*100:.1f}%",
                        help="Real จริงๆ จับได้กี่%"); st.progress(sp(rec_real))

    # ── Confusion Matrix ──
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df_ev['status'], df_ev['prediction'], labels=['Real','Fake'])
        tn, fp, fn, tp = cm.ravel()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
    <div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:18px 22px;">
      <div style="font-weight:700;font-size:0.9rem;color:#1E293B;margin-bottom:14px;">
        🔢 Confusion Matrix
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="background:#DCFCE7;border-radius:8px;padding:14px;text-align:center;">
          <div style="font-size:1.6rem;font-weight:800;color:#166534;">{tn}</div>
          <div style="font-size:0.76rem;color:#166534;font-weight:600;">True Negative<br>Real → Real ✅</div>
        </div>
        <div style="background:#FEE2E2;border-radius:8px;padding:14px;text-align:center;">
          <div style="font-size:1.6rem;font-weight:800;color:#991B1B;">{fp}</div>
          <div style="font-size:0.76rem;color:#991B1B;font-weight:600;">False Positive<br>Real → Fake ❌</div>
        </div>
        <div style="background:#FEF3C7;border-radius:8px;padding:14px;text-align:center;">
          <div style="font-size:1.6rem;font-weight:800;color:#92400E;">{fn}</div>
          <div style="font-size:0.76rem;color:#92400E;font-weight:600;">False Negative<br>Fake → Real ⚠️</div>
        </div>
        <div style="background:#DCFCE7;border-radius:8px;padding:14px;text-align:center;">
          <div style="font-size:1.6rem;font-weight:800;color:#166534;">{tp}</div>
          <div style="font-size:0.76rem;color:#166534;font-weight:600;">True Positive<br>Fake → Fake ✅</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

        with st.expander("ดูรายการที่ตรวจสอบแล้ว"):
            cols = [c for c in ['prediction','status','confidence'] if c in df_ev.columns]
            st.dataframe(df_ev[cols], width="stretch")

    except Exception as e:
        st.error(f"คำนวณไม่ได้: {e}")

# ═══════════════════════════════════════════════════════
#  ADMIN: Review Feedback
# ═══════════════════════════════════════════════════════
def show_feedback_review():
    page_header("💬","Review Feedback","ตรวจสอบ Feedback จากผู้ใช้งาน — ให้เฉลยเพื่อสอน AI")

    # Export panel
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1148A8,#0097A7);border-radius:14px;
                padding:22px 26px;margin-bottom:24px;">
      <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:1rem;font-weight:700;
                  color:#fff;margin-bottom:4px;">🧠 Export Training Dataset</div>
      <div style="font-size:0.83rem;color:rgba(255,255,255,0.75);">
        รวมข้อมูลที่ตรวจสอบแล้วพร้อม Trending News เพื่อนำไป Retrain โมเดล
      </div>
    </div>""", unsafe_allow_html=True)

    col_btn, _ = st.columns([1,2])
    with col_btn:
        if st.button("📥 ดาวน์โหลด Dataset", type="primary", width="stretch"):
            with st.spinner("กำลังรวบรวมข้อมูล..."):
                df_t = db.get_all_trending(); df_f = db.get_approved_feedbacks()
                frames = []
                if not df_t.empty and 'content' in df_t.columns: frames.append(df_t[['content','label']])
                if not df_f.empty: frames.append(df_f[['text','status']].rename(columns={'text':'content','status':'label'}))
                if frames:
                    df_final = pd.concat(frames,ignore_index=True).dropna(subset=['content','label']).drop_duplicates(subset=['content'])
                    csv = df_final.to_csv(index=False).encode('utf-8')
                    st.success(f"✅ {len(df_final)} รายการพร้อมดาวน์โหลด")
                    st.download_button("⬇️ บันทึกไฟล์ CSV", data=csv, file_name="retrain_dataset.csv", mime="text/csv", width="stretch")
                    db.log_system_event(user_id=st.session_state.get('user_id'), action="EXPORT_DATA", details=f"Exported {len(df_final)} rows", level="WARNING")
                else: st.warning("ยังไม่มีข้อมูล")

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    all_items = db.get_pending_feedbacks()

    # ✅ filter bar — แสดงเสมอ ไม่ early return
    col_f, col_s = st.columns([2, 1])
    with col_f:
        search_q = st.text_input("🔍 ค้นหา", placeholder="พิมพ์คำค้นหา...", label_visibility="collapsed")
    with col_s:
        filter_status = st.selectbox("สถานะ", ["ทั้งหมด", "pending", "Real", "Fake", "Ignored"], label_visibility="collapsed")

    # filter
    filtered = all_items
    if filter_status != "ทั้งหมด":
        filtered = [i for i in filtered if i.get('status') == filter_status]
    if search_q:
        filtered = [i for i in filtered if search_q.lower() in str(i.get('title','')).lower()
                    or search_q.lower() in str(i.get('text','')).lower()]

    # summary cards
    n_pending  = sum(1 for i in all_items if i.get('status') == 'pending')
    n_reviewed = sum(1 for i in all_items if i.get('status') != 'pending')
    n_true     = sum(1 for i in all_items if i.get('status') == 'Real')
    n_fake     = sum(1 for i in all_items if i.get('status') == 'Fake')
    n_ignored  = sum(1 for i in all_items if i.get('status') == 'Ignored')

    st.markdown(f"""
    <div style="display:flex;gap:10px;margin-bottom:18px;">
      <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
                  padding:12px 18px;flex:1;text-align:center;">
        <div style="font-size:1.4rem;font-weight:800;color:#ffc000;">{n_pending}</div>
        <div style="font-size:0.78rem;color:#ffc000;">รอตรวจสอบ</div>
      </div>
      <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;
                  padding:12px 18px;flex:1;text-align:center;">
        <div style="font-size:1.4rem;font-weight:800;color:#1148A8;">{n_reviewed}</div>
        <div style="font-size:0.78rem;color:#1148A8;">ตรวจสอบแล้ว</div>
      </div>
      <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;
                  padding:12px 18px;flex:1;text-align:center;">
        <div style="font-size:1.4rem;font-weight:800;color:#166534;">{n_true}</div>
        <div style="font-size:0.78rem;color:#16A34A;">ข่าวจริง</div>
      </div>
      <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;
                  padding:12px 18px;flex:1;text-align:center;">
        <div style="font-size:1.4rem;font-weight:800;color:#991B1B;">{n_fake}</div>
        <div style="font-size:0.78rem;color:#991B1B;">ข่าวปลอม</div>
      </div>
      <div style="background:#F0FDF4;border:1px solid #475569;border-radius:10px;
                  padding:12px 18px;flex:1;text-align:center;">
        <div style="font-size:1.4rem;font-weight:800;color:#166534;">{n_ignored}</div>
        <div style="font-size:0.78rem;color:#475569;">ถูกปฏิเสธ</div>
      </div>
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                  padding:12px 18px;flex:1;text-align:center;">
        <div style="font-size:1.4rem;font-weight:800;color:#1E293B;">{len(all_items)}</div>
        <div style="font-size:0.78rem;color:#64748B;">ทั้งหมด</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if not filtered:
        st.info("ไม่พบรายการที่ตรงกับเงื่อนไข")
        return

    st.caption(f"แสดง {len(filtered)} รายการ")

    import html
    for idx, item in enumerate(filtered):
        # ✅ badge สี ตาม status
        status_cfg = {
            'pending':  ("#000000","#92400E","⏳ รอตรวจสอบ"),
            'Real':     ("#00FF59","#166534","✅ Real"),
            'Fake':     ("#FF0000","#991B1B","❌ Fake"),
            'Ignored':  ("#69B4FF","#475569","🗑️ Ignored"),
        }.get(item.get('status','pending'), ("#F1F5F9","#475569","—"))

        ttl = (item['title'][:55]+"…") if item['title'] and len(item['title'])>55 else (item['title'] or "No Title")

        with st.expander(f"📰 {ttl}", expanded=(item.get('status')=='pending')):
            st.markdown(f"""
    <span style="background:{status_cfg[0]};color:{status_cfg[1]};
                 font-size:0.75rem;font-weight:700;padding:3px 10px;
                 border-radius:99px;">{status_cfg[2]}</span>
    """, unsafe_allow_html=True)

            c1, c2 = st.columns([3,2])
            with c1:
                st.markdown("**เนื้อหาข่าว**")
                st.markdown(f"""
                <div style="background:#1E293B;border:1px solid #334155;border-radius:8px;
                padding:13px 15px;font-size:0.9rem;line-height:1.6;
                color:#E2E8F0;max-height:300px;overflow-y:auto;">
                {html.escape(str(item['text']))}
                </div>""", unsafe_allow_html=True)
                st.caption(f"📅 {item['timestamp']}")

            with c2:
                ai_b = status_badge(item['ai_result']) if item['ai_result'] in ('Real','Fake','Unverified') else item['ai_result']
                fb = item.get('user_report') or ''
                fb_html = {
                    "Correct": "<span style='background:#DCFCE7;color:#166534;-webkit-text-fill-color:#166534;padding:3px 10px;border-radius:99px;font-weight:700;font-size:0.82rem;'>👍 AI ทายถูก</span>",
                    "Incorrect": "<span style='background:#FEE2E2;color:#991B1B;-webkit-text-fill-color:#991B1B;padding:3px 10px;border-radius:99px;font-weight:700;font-size:0.82rem;'>👎 AI ทายผิด</span>"
                }.get(fb, "<span style='color:#94A3B8;-webkit-text-fill-color:#94A3B8;font-size:0.88rem;'>— ยังไม่มี Feedback</span>")

    # ✅ แปลง confidence เป็น float และปัดทศนิยม 2 ตำแหน่ง
                try:
                    conf_display = f"{float(item['ai_confidence']):.2f}%"
                except:
                    conf_display = f"{item['ai_confidence']}%"

                st.markdown(f"""
            <div style="display:flex;flex-direction:column;gap:9px;margin-top:2px;">
            <div style="background:#2D3F55;border:1px solid #3D5068;
                  border-radius:8px;padding:11px 13px;">
                <div style="font-size:0.72rem;color:#93C5FD;font-weight:700;
                    text-transform:uppercase;margin-bottom:5px;">AI PREDICTION</div>
                {ai_b}&nbsp;
                <span style="font-size:0.83rem;color:#E2E8F0;font-weight:600;">
                {conf_display}
                </span>
            </div>
            <div style="background:#2D3320;border:1px solid #4A3B10;
                  border-radius:8px;padding:11px 13px;">
                <div style="font-size:0.72rem;color:#FCD34D;font-weight:700;
                    text-transform:uppercase;margin-bottom:4px;">💬 USER SAYS</div>
                {fb_html}
            </div>
            </div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin:14px 0 8px;font-weight:700;font-size:0.88rem;color:#1E293B;'>👨‍⚖️ Admin Decision</div>", unsafe_allow_html=True)
                fid = item['feedback_id'] if item['feedback_id'] is not None else idx
                b1, b2, b3 = st.columns(3)
                with b1:
                        if st.button("✅ Real", key=f"real_{fid}_{idx}", type="primary", width="stretch"):
                            if item['feedback_id'] is not None:
                                db.update_feedback_status(item['feedback_id'], 'Real')
                                db.log_system_event(user_id=st.session_state.get('user_id'),
                                            action="REVIEW_FEEDBACK",
                                            details=f"feedback_id={fid} → Real", level="WARNING")
                                st.rerun()
                with b2:
                        if st.button("❌ Fake", key=f"fake_{fid}_{idx}", type="primary", width="stretch"):
                            if item['feedback_id'] is not None:
                                db.update_feedback_status(item['feedback_id'], 'Fake')
                                db.log_system_event(user_id=st.session_state.get('user_id'),
                                            action="REVIEW_FEEDBACK",
                                            details=f"feedback_id={fid} → Fake", level="WARNING")
                                st.rerun()
                with b3:
                        if st.button("🗑️ Ignore", key=f"del_{fid}_{idx}", width="stretch"):
                            if item['feedback_id'] is not None:
                                db.update_feedback_status(item['feedback_id'], 'Ignored')
                                db.log_system_event(user_id=st.session_state.get('user_id'),
                                            action="REVIEW_FEEDBACK",
                                            details=f"feedback_id={fid} → Ignored", level="WARNING")
                                st.rerun()


# ═══════════════════════════════════════════════════════
#  ADMIN: Manage Trending News
# ═══════════════════════════════════════════════════════
def manage_trending_news():
    page_header("📰","Manage Trending News","เพิ่ม แก้ไข และจัดการข่าวในระบบ")

    df_exp = db.get_all_trending()
    ca, cb = st.columns([3,1])
    with ca:
        st.caption("ส่งออกรายการข่าวทั้งหมดเป็นไฟล์ .csv สำหรับเทรน AI")
    with cb:
        if not df_exp.empty:
            cols = [c for c in ['headline','content','label'] if c in df_exp.columns]
            csv_data = df_exp[cols].to_csv(index=False).encode('utf-8')

            def on_export_click():
                db.log_system_event(
                    user_id=st.session_state.get('user_id'),
                    action="EXPORT_DATA",
                    details=f"Export {len(df_exp)} rows",
                    level="WARNING"
                )

            st.download_button(
                "⬇️ Export CSV",
                data=csv_data,
                file_name="trending_dataset.csv",
                mime="text/csv",
                width="stretch",
                type="primary",
                on_click=on_export_click
            )
        else:
            st.button("ไม่มีข้อมูล", disabled=True, width="stretch")

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋  รายการข่าวปัจจุบัน","➕  เพิ่มข่าวใหม่"])

    with tab1:
        df = db.get_all_trending()
        if df.empty:
            st.info("ยังไม่มีข่าวในระบบ")
        else:
            sc, fc = st.columns([3,1])
            with sc: q = st.text_input("🔍 ค้นหา", placeholder="พิมพ์คำค้นหา...", label_visibility="collapsed")
            with fc: flt = st.selectbox("สถานะ", ["All","Real","Fake","Unverified"], label_visibility="collapsed")
            if q: df = df[df['headline'].str.contains(q,case=False,na=False)|df['content'].str.contains(q,case=False,na=False)]
            if flt!="All": df=df[df['label']==flt]
            st.caption(f"พบ {len(df)} รายการ")

            for _, row in df.iterrows():
                ekey = f"edit_{row['id']}"
                if ekey not in st.session_state:
                    st.session_state[ekey] = False
                ts = str(row.get('updated_at', '-')).replace("T", " ")[:16]

    # ── View mode ──
                if not st.session_state[ekey]:
                    label_cfg = {
                        "Fake":       ("#FEE2E2", "#991B1B", "#EF4444"),
                        "Real":       ("#DCFCE7", "#166534", "#22C55E"),
                        "Unverified": ("#FEF3C7", "#92400E", "#F59E0B"),
                    }.get(row['label'], ("#F1F5F9", "#475569", "#CBD5E1"))

                    # ✅ สร้าง image HTML ถ้ามีรูป
                    img_html = ""
                    if row.get("image_url"):
                        img_html = f"""
                        <img src="{row['image_url']}"
                            style="width:100%;max-height:180px;object-fit:cover;
                                    border-radius:8px;margin-bottom:12px;" />
                        """

                    # ✅ สร้าง category badge
                    cat = row.get('category') or 'ทั่วไป'

                    st.markdown(f"""
                    <div style="background:#fff;border:1px solid #E2E8F0;
                                border-radius:12px;padding:16px 20px;margin-bottom:10px;">

                    {img_html}

                    <div style="display:flex;align-items:flex-start;
                                justify-content:space-between;gap:12px;margin-bottom:8px;">
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.95rem;
                                    font-weight:700;color:#1E293B;flex:1;line-height:1.4;">
                        {row['headline'][:80]}
                        </div>
                        <span style="flex-shrink:0;background:{label_cfg[0]};color:{label_cfg[1]};
                                    -webkit-text-fill-color:{label_cfg[1]};
                                    border:1px solid {label_cfg[2]}44;font-size:0.7rem;font-weight:800;
                                    padding:3px 10px;border-radius:99px;text-transform:uppercase;
                                    letter-spacing:0.5px;">
                        {row['label']}
                        </span>
                    </div>

                    <!-- ✅ category badge -->
                    <div style="margin-bottom:10px;">
                        <span style="background:#EFF6FF;color:#1148A8;-webkit-text-fill-color:#1148A8;
                                    font-size:0.72rem;font-weight:700;padding:2px 10px;
                                    border-radius:99px;border:1px solid #BFDBFE;">
                        📂 {cat}
                        </span>
                    </div>

                    <div style="font-size:0.87rem;color:#475569;-webkit-text-fill-color:#475569;
                                line-height:1.65;margin-bottom:12px;">
                        {row['content'][:200]}{'…' if len(str(row['content'])) > 200 else ''}
                    </div>

                    <div style="font-size:0.74rem;color:#94A3B8;">🕒 {ts}</div>
                    </div>""", unsafe_allow_html=True)

                    ca2, cb2, _ = st.columns([1, 1, 5])
                    with ca2:
                        if st.button("✏️ แก้ไข", key=f"e_{row['id']}", use_container_width=True):
                            st.session_state[ekey] = True
                            st.rerun()
                    with cb2:
                        if st.button("🗑️ ลบ", key=f"d_{row['id']}", use_container_width=True):
                            if db.delete_trending(row['id']):
                                db.log_system_event(
                                    user_id=st.session_state.get('user_id'),
                                    action="DELETE_DATA",
                                    details=f"Deleted trending ID {row['id']}",
                                    level="WARNING"
                                )
                                st.rerun()

    # ── Edit mode ──
                else:
                    st.markdown(f"""
        <div style="background:#F0F6FF;border:1.5px solid #BFDBFE;
                    border-radius:12px;padding:16px 20px;margin-bottom:4px;">
          <div style="font-size:0.72rem;font-weight:700;color:#1148A8;
                      text-transform:uppercase;letter-spacing:0.6px;margin-bottom:12px;">
            ✏️ แก้ไขข่าว
          </div>
                    </div>""", unsafe_allow_html=True)

                    eh = st.text_input("หัวข้อข่าว", value=row['headline'], key=f"h_{row['id']}")
                    ec = st.text_area("เนื้อหา", value=row['content'], height=120, key=f"c_{row['id']}")
                    opts = ["Fake", "Real", "Unverified"]
                    el = st.selectbox(
                        "สถานะ", opts,
                        index=opts.index(row['label']) if row['label'] in opts else 0,
                        key=f"l_{row['id']}"
                    )

                    sa, sb, _ = st.columns([1, 1, 4])
                    with sa:
                        if st.button("💾 บันทึก", key=f"s_{row['id']}", type="primary", use_container_width=True):
                            if eh.strip() and ec.strip():
                                db.update_trending(row['id'], eh, ec, el)
                                db.log_system_event(
                                    user_id=st.session_state.get('user_id'),
                                    action="UPDATE_TRENDING",
                                    details=f"Updated ID {row['id']}",
                                    level="INFO"
                                )
                                st.session_state[ekey] = False
                                st.rerun()
                            else:
                                st.warning("กรุณากรอกข้อมูลให้ครบ")
                    with sb:
                        if st.button("✕ ยกเลิก", key=f"x_{row['id']}", use_container_width=True):
                            st.session_state[ekey] = False
                            st.rerun()

                    st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    with tab2:
        with st.form("add_news", clear_on_submit=True):
            st.markdown("**เพิ่มข่าวใหม่ลงระบบ**")
            nh = st.text_input("หัวข้อข่าว", placeholder="พิมพ์พาดหัวข่าว...")
            nc = st.text_area("เนื้อหา", placeholder="รายละเอียดข่าวโดยย่อ...", height=120)

            col1, col2 = st.columns(2)
            with col1:
                nl = st.selectbox("สถานะ", ["Fake","Real","Unverified"])
            with col2:
                CATEGORIES = [
                    "นโยบายรัฐบาล-ข่าวสาร","ผลิตภัณฑ์สุขภาพ","การเงิน-หุ้น","ภัยพิบัติ",
                    "ความสงบและความมั่นคง","ข่าวอื่นๆ","เศรษฐกิจ","ยาเสพติด",  
                ]
                nc_cat = st.selectbox("หมวดหมู่", CATEGORIES)

            # ✅ อัปโหลดไฟล์รูปแทน URL
            uploaded_file = st.file_uploader(
                "รูปภาพประกอบ (ถ้ามี)",
                type=["jpg","jpeg","png","webp"],
                help="ขนาดไม่เกิน 5MB"
            )

            # preview
            if uploaded_file:
                st.image(uploaded_file, width=200, caption="Preview")

            if st.form_submit_button("💾 บันทึกข่าว", type="primary", width="stretch"):
                if nh.strip() and nc.strip():
                    # ✅ อัปโหลดรูปก่อนบันทึก
                    image_url = ""
                    if uploaded_file:
                        with st.spinner("กำลังอัปโหลดรูป..."):
                            image_url = db.upload_image_to_supabase(
                                uploaded_file.read(),
                                uploaded_file.name
                            )

                    if db.create_trending(nh, nc, nl, nc_cat, image_url):
                        st.success("✅ เพิ่มข่าวเรียบร้อย")
                        time.sleep(0.7)
                        st.rerun()
                    else:
                        st.error("เกิดข้อผิดพลาด")
                else:
                    st.warning("กรุณากรอกข้อมูลให้ครบ")


# ═══════════════════════════════════════════════════════
#  ADMIN: System Analytics
# ═══════════════════════════════════════════════════════
def show_system_analytics():
    with st.spinner("กำลังโหลดข้อมูล..."):
        data = db.get_system_analytics_data()
    total_users = data["total_users"]
    df_preds    = data["df_preds"]
    df_logs     = data["df_logs"]

    total_checks = 0; avg_daily = 0; peak_hour = "—"

    if not df_preds.empty:
        total_checks = len(df_preds)
        # ✅ แปลง UTC → GMT+7
        df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'], utc=True) \
                                   .dt.tz_convert("Asia/Bangkok")
        df_preds['date'] = df_preds['timestamp'].dt.date
        df_preds['hour'] = df_preds['timestamp'].dt.hour
        mh = df_preds['hour'].mode()
        if not mh.empty: peak_hour = f"{int(mh[0]):02d}:00"

    if not df_logs.empty:
        # ✅ แปลง UTC → GMT+7
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], utc=True) \
                                  .dt.tz_convert("Asia/Bangkok")
        df_logs['date'] = df_logs['timestamp'].dt.date
        avg_daily = int(df_logs.groupby('date')['user_id'].nunique().mean() or 0)


    c1,c2,c3,c4=st.columns(4)
    with c1: kpi_card("👥","Total Users",      f"{total_users:,}")
    with c2: kpi_card("📄","Total Checks",     f"{total_checks:,}")
    with c3: kpi_card("📈","Avg Daily Users",  f"{avg_daily:,}","Last 7 days")
    with c4: kpi_card("🔥","Peak Hour",        peak_hour)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    COLORS=['#1565C0','#0097A7','#16A34A','#F59E0B']

    row1a, row1b = st.columns([3,2])
    with row1a:
        st.markdown("""<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
        padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-weight:700;font-size:0.95rem;
        color:#1E293B;">Daily Usage — Last 7 Days</div>
        <div style="font-size:0.79rem;color:#94A3B8;margin-bottom:12px;">Checks & active users per day</div>""",
        unsafe_allow_html=True)
        last7 = [(now_bkk() - timedelta(days=i)).date() for i in range(6, -1, -1)]
        dft   = pd.DataFrame({'Date': last7})
        cpd = df_preds.groupby('date').size() if not df_preds.empty else pd.Series()
        upd = df_logs.groupby('date')['user_id'].nunique() if not df_logs.empty else pd.Series()
        dft['News Checks'] = dft['Date'].map(cpd).fillna(0).astype(int)
        dft['Users']       = dft['Date'].map(upd).fillna(0).astype(int)
        dft['Date']        = pd.to_datetime(dft['Date']).dt.strftime('%b %d')
        fig1 = px.area(dft, x='Date', y=['News Checks','Users'],
               color_discrete_sequence=COLORS[:2],
               labels={'value': 'จำนวน', 'Date': 'วันที่', 'variable': 'ประเภท'})
        # ✅ Daily Usage chart
        fig1.update_layout(
            margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            font_color='#334155',
            xaxis_title="วันที่",        # ✅ เพิ่ม
            yaxis_title="จำนวน (ครั้ง)", # ✅ เพิ่ม
            legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"),
            legend_title_text=''
        )
        st.plotly_chart(fig1, width="stretch", key="trend_chart")
        st.markdown("</div>",unsafe_allow_html=True)

    with row1b:
        st.markdown("""<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
        padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-weight:700;font-size:0.95rem;
        color:#1E293B;">Classification Results</div>
        <div style="font-size:0.79rem;color:#94A3B8;margin-bottom:12px;">Real vs Fake distribution</div>""",
        unsafe_allow_html=True)
        if not df_preds.empty and 'result' in df_preds.columns:
            df_preds['result']=df_preds['result'].astype(str).str.capitalize()
            dfc=df_preds['result'].value_counts().reset_index(); dfc.columns=['Result','Count']
            dfc['Result']=dfc['Result'].replace({'Real':'Real News','Fake':'Fake News'})
            fig2=px.pie(dfc,values='Count',names='Result',hole=0.45,color='Result',
                        color_discrete_map={'Real News':'#16A34A','Fake News':'#DC2626','Error':'#94A3B8'})
            fig2.update_traces(textposition='outside',textinfo='percent+label',textfont_size=10)
            fig2.update_layout(
                margin=dict(l=10,r=10,t=10,b=10),
                showlegend=False,
                paper_bgcolor='white',
                font_color='#1E293B',        # ✅ เข้มขึ้น
                font=dict(size=12),
            )
            st.plotly_chart(fig2,width="stretch",key="pie_chart")
        else: st.info("ยังไม่มีข้อมูลเพียงพอ")
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>",unsafe_allow_html=True)
    st.markdown("""<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
    padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
    <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-weight:700;font-size:0.95rem;
    color:#1E293B;">Activity by Hour of Day</div>
    <div style="font-size:0.79rem;color:#94A3B8;margin-bottom:12px;">จำนวนการตรวจข่าวรายชั่วโมง</div>""",
    unsafe_allow_html=True)
    dft2=pd.DataFrame({'Hour':range(24)})
    dft2['Checks']=(df_preds.groupby('hour').size() if not df_preds.empty else pd.Series()).reindex(range(24),fill_value=0)
    dft2['Time']=dft2['Hour'].apply(lambda x:f"{x:02d}:00")
    # แก้ fig3 Activity by Hour
    fig3 = px.bar(dft2, x='Time', y='Checks',
                    color_discrete_sequence=[COLORS[0]],
                    labels={'Time': 'ชั่วโมง', 'Checks': 'จำนวนการตรวจ'})
    fig3.update_layout(
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis_title="ชั่วโมง",         # ✅ เพิ่ม
            yaxis_title="จำนวนการตรวจ",    # ✅ เพิ่ม
            plot_bgcolor='white', paper_bgcolor='white',
            font_color='#334155', bargap=0.35
        )
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3,width="stretch",key="hour_chart")
    st.markdown("</div>",unsafe_allow_html=True)

    # System logs
    st.markdown("<div style='height:14px;'></div>",unsafe_allow_html=True)
    st.markdown("""<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
    padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
    <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-weight:700;font-size:0.95rem;
    color:#1E293B;margin-bottom:14px;">Recent System Logs</div>""",unsafe_allow_html=True)
    logs_cont=st.container(height=300)
    with logs_cont:
        recent=db.get_system_logs(limit=100)
        if recent:
            for row in recent:
                ts, user, action, details, level = row
                try: ft = ft = pd.to_datetime(ts, utc=True) \
                       .tz_convert("Asia/Bangkok") \
                       .strftime("%m/%d %H:%M")
                except: ft = str(ts)
                bc = {"ERROR":"#DC2626","WARNING":"#D97706"}.get(level,"#1565C0")
                bg = {"ERROR":"#FFF5F5","WARNING":"#FFFBEB"}.get(level,"#F0F6FF")
                tc = {"ERROR":"#7F1D1D","WARNING":"#78350F"}.get(level,"#1E3A5F")

    # ✅ icon ตาม action
                action_icon = {
                    "PREDICT":      "🤖",
                    "USER_LOGOUT":  "🚪",
                    "USER_LOGIN":   "🔑",
                    "EXPORT_DATA":  "📥",
                    "DELETE_DATA":  "🗑️",
                    "ROLE_UPDATE":  "🛡️",
                    "API_ERROR":    "❌",
                }.get(action, "📋")

                st.markdown(f"""
    <div style="background:{bg};padding:11px 14px;border-radius:8px;
                margin-bottom:6px;border-left:3px solid {bc};">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
        <div style="display:flex;align-items:center;gap:7px;">
          <span style="font-size:1rem;">{action_icon}</span>
          <span style="font-size:0.82rem;font-weight:800;color:{bc};">{action}</span>
          <span style="background:{bc}22;color:{bc};font-size:0.67rem;font-weight:700;
                       padding:2px 7px;border-radius:99px;">{level}</span>
        </div>
        <span style="font-size:0.73rem;color:{tc};opacity:0.6;">{ft}</span>
      </div>
      <!-- ✅ แสดง user ชัดเจน -->
      <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:4px;">
        <span style="background:#1E293B;color:#93C5FD;font-size:0.72rem;font-weight:700;
                     padding:2px 9px;border-radius:99px;">👤 {user}</span>
        <span style="font-size:0.79rem;color:{tc};">
          {(details[:100]+'…') if len(details)>100 else details}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)
        else: st.info("ยังไม่มี log")
    st.markdown("</div>",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  ADMIN: Manage Users
# ═══════════════════════════════════════════════════════
def manage_users_page():
    with st.spinner("Loading..."): df=db.get_user_management_data()
    if df.empty: st.warning("ไม่พบข้อมูลผู้ใช้"); return

    df['created_at']=pd.to_datetime(df['created_at']).dt.strftime('%b %d, %Y')
    if 'last_active' in df.columns:
        df['last_active']=pd.to_datetime(df['last_active'],errors='coerce').dt.strftime('%b %d, %Y').fillna('—')
    else: df['last_active']='—'

    total=len(df); active=len(df[df['status']=='active']) if 'status' in df.columns else total
    admins=len(df[df['role']=='admin']); checks=df['checks'].sum()

    c1,c2,c3,c4=st.columns(4)
    with c1: kpi_card("👥","Total Users",  f"{total:,}")
    with c2: kpi_card("🟢","Active Users", f"{active:,}")
    with c3: kpi_card("🛡️","Admins",       f"{admins:,}")
    with c4: kpi_card("📄","Total Checks", f"{checks:,}")

    st.markdown("<div style='height:20px;'></div>",unsafe_allow_html=True)
    sc,fc,_=st.columns([3,1,1])
    with sc: sq=st.text_input("🔍 ค้นหา (Email / Username)",placeholder="พิมพ์เพื่อค้นหา...")
    with fc: rf=st.selectbox("Role",["All","user","admin"])

    dfd=df.copy()
    if sq:
        mask = pd.Series([False] * len(dfd), index=dfd.index)
        if 'email'    in dfd.columns: mask |= dfd['email'].astype(str).str.contains(sq, case=False, na=False)
        if 'username' in dfd.columns: mask |= dfd['username'].astype(str).str.contains(sq, case=False, na=False)
        dfd = dfd[mask]

    if rf!="All": dfd=dfd[dfd['role']==rf]

    dcols={k:v for k,v in {'id':'ID','username':'Username','email':'Email','role':'Role','status':'Status','checks':'Checks','created_at':'Joined','last_active':'Last Active'}.items() if k in dfd.columns}
    st.dataframe(dfd[list(dcols)].rename(columns=dcols),width="stretch",hide_index=True,
                 column_config={"ID":st.column_config.TextColumn("ID",width="small"),
                                "Checks":st.column_config.NumberColumn("Checks",format="%d"),
                                "Role":st.column_config.TextColumn("Role",width="small")})

    st.markdown("<hr style='margin:22px 0;'>",unsafe_allow_html=True)
    section_title("⚙️ User Actions","แก้ไขสิทธิ์และสถานะผู้ใช้งาน")

    with st.expander("เลือกผู้ใช้และแก้ไข",expanded=True):
        se=st.text_input("ระบุ Email หรือ ID:",placeholder="เช่น 1 หรือ user@example.com")
        if se.strip():
            tu=df[df['id']==int(se.strip())] if se.strip().isdigit() else df[df['email'].str.contains(se.strip(),case=False,na=False)]
            if tu.empty: st.warning("ไม่พบผู้ใช้งาน")
            elif len(tu)>1: st.info("พบหลายคน — กรุณาระบุให้แม่นยำขึ้น"); st.dataframe(tu[['id','email','role']],hide_index=True)
            else:
                ud=tu.iloc[0]; sid=ud['id']
                st.success(f"พบ: **{ud.get('username','—')}** | {ud['email']} (ID: {sid})")
                if sid==st.session_state.get('user_id'):
                    st.info("ไม่สามารถแก้ไขบัญชีตัวเองได้")
                else:
                    c1,c2,c3=st.columns(3)
                    with c1: nr=st.radio("Role",["user","admin"],index=0 if ud['role']=='user' else 1,key="re")
                    with c2: ns=st.radio("Status",["active","inactive"],index=0 if ud.get('status','active')=='active' else 1,key="se")
                    with c3:
                        st.write(""); st.write("")
                        if st.button("💾 บันทึก",type="primary"):
                            if db.update_user_role_status(sid,nr,ns):
                                db.log_system_event(user_id=st.session_state.get('user_id'),action="ROLE_UPDATE",details=f"User {sid}: role→{nr}, status→{ns}",level="WARNING")
                                st.success("อัปเดตสำเร็จ!"); time.sleep(0.7); st.rerun()
                            else: st.error("เกิดข้อผิดพลาด")


# ═══════════════════════════════════════════════════════
#  SESSION DEFAULTS
# ═══════════════════════════════════════════════════════
for k,v in [('logged_in',False),('user_id',None),('username',""),('role',""),
            ('reset_mode',False),('register_mode',False),('otp_sent',False),('reset_email_temp',"")]:
    if k not in st.session_state: st.session_state[k]=v


# ═══════════════════════════════════════════════════════
#  PAGE: Reset Password
# ═══════════════════════════════════════════════════════
if st.session_state['reset_mode']:
    _,mid,_=st.columns([1,1.5,1])
    with mid:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:20px;
                    padding:40px 36px;box-shadow:0 4px 24px rgba(17,72,168,0.09);margin-top:2rem;
                    text-align:center;">
          <div style="font-size:44px;margin-bottom:12px;">🔑</div>
          <h2 style="margin:0 0 6px;color:#1148A8 !important;">กู้คืนรหัสผ่าน</h2>
          <p style="color:#64748B;font-size:0.87rem;margin:0 0 24px;">ระบบจะส่ง OTP ไปยังอีเมลของคุณ</p>
        </div>""",unsafe_allow_html=True)
        if not st.session_state['otp_sent']:
            ei=st.text_input("📧 Email",placeholder="your@email.com")
            c1,c2=st.columns(2)
            with c1:
                if st.button("ส่ง OTP",type="primary",width="stretch"):
                    if ei:
                        with st.spinner("กำลังส่ง..."): ok,msg=db.send_otp_email(ei)
                        if ok: 
                            st.success(msg)
                            st.session_state.update({'otp_sent': True, 'reset_email_temp': ei})
                            st.rerun()  # ✅ เพิ่มบรรทัดนี้
                        else: st.error(msg)
                    else: st.warning("กรุณากรอกอีเมล")
            with c2:
                if st.button("← กลับ",width="stretch"):
                    st.session_state['reset_mode']=False; st.rerun()
        else:
            st.success(f"ส่ง OTP ไปที่ {st.session_state['reset_email_temp']}")
            oi=st.text_input("รหัส OTP 6 หลัก",max_chars=6)
            np=st.text_input("รหัสผ่านใหม่",type="password")
            cp=st.text_input("ยืนยันรหัสผ่านใหม่",type="password")
            # ✅ เพิ่มตรงนี้ — เหมือน Register เลย
            if np:
                checks = {
                    "อย่างน้อย 8 ตัวอักษร":   len(np) >= 8,
                    "มีตัวพิมพ์เล็ก (a-z)":    any(c.islower() for c in np),
                    "มีตัวพิมพ์ใหญ่ (A-Z)":    any(c.isupper() for c in np),
                    "มีตัวเลข (0-9)":           any(c.isdigit() for c in np),
                }
                for label, passed in checks.items():
                    icon  = "✅" if passed else "❌"
                    color = "#166534" if passed else "#991B1B"
                    bg    = "#DCFCE7" if passed else "#FEE2E2"
                    st.markdown(f"""
                    <div style="background:{bg};color:{color};font-size:0.78rem;font-weight:600;
                                padding:4px 10px;border-radius:6px;margin-bottom:3px;">
                    {icon} {label}
                    </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            
            with c1:
                if st.button("ยืนยัน", type="primary", width="stretch"):
                    if not np or not cp:
                        st.error("กรุณากรอกรหัสผ่านให้ครบ")
                    elif np != cp:
                        st.error("❌ รหัสผ่านไม่ตรงกัน")
                    elif len(np) < 8:
                        st.error("❌ รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร")
                    elif not any(c.islower() for c in np):
                        st.error("❌ รหัสผ่านต้องมีตัวพิมพ์เล็กอย่างน้อย 1 ตัว")
                    elif not any(c.isupper() for c in np):
                        st.error("❌ รหัสผ่านต้องมีตัวพิมพ์ใหญ่อย่างน้อย 1 ตัว")
                    elif not any(c.isdigit() for c in np):
                        st.error("❌ รหัสผ่านต้องมีตัวเลขอย่างน้อย 1 ตัว")
                    else:
                        ok, msg = db.verify_otp_and_reset(
                            st.session_state['reset_email_temp'], oi, np)
                        if ok:
                            st.balloons(); st.success(msg); time.sleep(2)
                            st.session_state.update({'reset_mode': False, 'otp_sent': False})
                            st.rerun()
                        else:
                            st.error(msg)
            with c2:
                if st.button("← ย้อนกลับ",width="stretch"):
                    st.session_state['otp_sent']=False; st.rerun()


# ═══════════════════════════════════════════════════════
#  PAGE: Register
# ═══════════════════════════════════════════════════════
elif st.session_state['register_mode']:
    _,mid,_=st.columns([1,1.5,1])
    with mid:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:20px;
                    padding:40px 36px;box-shadow:0 4px 24px rgba(17,72,168,0.09);margin-top:2rem;
                    text-align:center;">
          <div style="font-size:44px;margin-bottom:12px;">✨</div>
          <h2 style="margin:0 0 6px;color:#1148A8 !important;">สมัครสมาชิก</h2>
          <p style="color:#64748B;font-size:0.87rem;margin:0 0 24px;">สร้างบัญชีเพื่อเริ่มใช้งาน TrueCheck AI</p>
        </div>""",unsafe_allow_html=True)
        nu=st.text_input("👤 Username")
        ne=st.text_input("📧 Email")
        np=st.text_input("🔒 Password",type="password")
        cp=st.text_input("🔒 ยืนยัน Password",type="password")
        # ✅ แสดง password strength แบบ real-time
        if np:
            checks = {
                "อย่างน้อย 8 ตัวอักษร":     len(np) >= 8,
                "มีตัวพิมพ์เล็ก (a-z)":      any(c.islower() for c in np),
                "มีตัวพิมพ์ใหญ่ (A-Z)":      any(c.isupper() for c in np),
                "มีตัวเลข (0-9)":             any(c.isdigit() for c in np),
            }
            for label, passed in checks.items():
                icon  = "✅" if passed else "❌"
                color = "#166534" if passed else "#991B1B"
                bg    = "#DCFCE7" if passed else "#FEE2E2"
                st.markdown(f"""
                <div style="background:{bg};color:{color};font-size:0.78rem;font-weight:600;
                            padding:4px 10px;border-radius:6px;margin-bottom:3px;">
                  {icon} {label}
                </div>""", unsafe_allow_html=True)

        st.write("")
        b1,b2=st.columns(2)
        with b1:
            if st.button("✅ สมัครสมาชิก",type="primary",width="stretch"):
                # ✅ validate ก่อน
                if not all([nu,np,ne]):
                    st.warning("กรุณากรอกข้อมูลให้ครบ")
                elif np != cp:
                    st.error("รหัสผ่านไม่ตรงกัน")
                elif len(np) < 8:
                    st.error("รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร")
                elif not any(c.islower() for c in np):
                    st.error("รหัสผ่านต้องมีตัวพิมพ์เล็กอย่างน้อย 1 ตัว")
                elif not any(c.isupper() for c in np):
                    st.error("รหัสผ่านต้องมีตัวพิมพ์ใหญ่อย่างน้อย 1 ตัว")
                elif not any(c.isdigit() for c in np):
                    st.error("รหัสผ่านต้องมีตัวเลขอย่างน้อย 1 ตัว")
                elif db.create_user(nu,np,ne):
                    db.log_system_event(user_id=None, action="USER_REGISTER",
                        details=f"New user registered: {nu} ({ne})", level="INFO")
                    st.success("สมัครสำเร็จ!"); time.sleep(1.2)
                    st.session_state['register_mode']=False; st.rerun()
                else:
                    st.error("Username หรือ Email นี้มีผู้ใช้งานแล้ว")
        with b2:
            if st.button("← กลับ Login",width="stretch"):
                st.session_state['register_mode']=False; st.rerun()


# ═══════════════════════════════════════════════════════
#  PAGE: Login
# ═══════════════════════════════════════════════════════
elif not st.session_state['logged_in']:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D47A1 0%,#1565C0 55%,#00838F 100%);
                border-radius:20px;padding:56px 40px 52px;margin-bottom:40px;
                box-shadow:0 8px 40px rgba(13,71,161,0.22);position:relative;overflow:hidden;">
      <div style="position:absolute;top:-50px;right:-50px;width:260px;height:260px;
                  border-radius:50%;background:rgba(255,255,255,0.04);pointer-events:none;"></div>
      <div style="position:absolute;bottom:-70px;right:100px;width:180px;height:180px;
                  border-radius:50%;background:rgba(255,255,255,0.03);pointer-events:none;"></div>
      <div style="position:relative;z-index:1;text-align:center;">
        <div style="font-size:54px;margin-bottom:14px;">🛡️</div>
        <h1 style="color:#FFFFFF !important;font-family:'IBM Plex Sans Thai',sans-serif !important;
                   font-size:2.25rem !important;font-weight:800 !important;
                   letter-spacing:-0.5px !important;margin:0 0 10px;">TrueCheck AI</h1>
        <p style="color:rgba(255,255,255,0.80);font-size:1rem;margin:0 0 32px;line-height:1.5;">
          ตรวจสอบความน่าเชื่อถือของข่าวด้วย AI<br>รวดเร็ว แม่นยำ โปร่งใส
        </p>
      </div>
    </div>""", unsafe_allow_html=True)

    _,mid,_=st.columns([1,1.35,1])
    with mid:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:20px;
                    padding:36px 32px;box-shadow:0 4px 20px rgba(17,72,168,0.08);">
          <h3 style="margin:0 0 22px;text-align:center;color:#0F172A !important;
                     font-family:'IBM Plex Sans Thai',sans-serif !important;font-weight:700 !important;">
            เข้าสู่ระบบ</h3>
        </div>""", unsafe_allow_html=True)

        u_in=st.text_input("Username",placeholder="กรอก Username",label_visibility="visible")
        p_in=st.text_input("Password",type="password",placeholder="กรอก Password",label_visibility="visible")
        st.markdown("<div style='height:6px;'></div>",unsafe_allow_html=True)

        if st.button("เข้าสู่ระบบ  →", type="primary", width="stretch"):
            udata = db.authenticate_user(u_in, p_in)
            if udata:
                st.session_state.update({
                    'logged_in': True,
                    'user_id': udata[0],
                    'username': udata[1],
                    'role': udata[2],
                    'need_to_save_cookie': True
                })
                db.log_system_event(user_id=udata[0], action="USER_LOGIN",
                details=f"{udata[1]} เข้าสู่ระบบสำเร็จ (role: {udata[2]})", level="INFO")
                st.rerun()
            else:
                try:
                    db.get_supabase()
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
                    # ✅ log login fail
                    db.log_system_event(user_id=None, action="LOGIN_FAILED",
                    details=f"Login failed for username: '{u_in}'", level="WARNING")
                except Exception:
                    st.error("⚠️ ไม่สามารถเชื่อมต่อฐานข้อมูลได้ — กรุณาตรวจสอบอินเทอร์เน็ต")

        st.markdown("<div style='height:8px;'></div>",unsafe_allow_html=True)
        ca,_,cb=st.columns([1,0.1,1])
        with ca:
            if st.button("🔑 ลืมรหัสผ่าน?",width="stretch"):
                st.session_state['reset_mode']=True; st.rerun()
        with cb:
            if st.button("✨ สมัครสมาชิก",width="stretch"):
                st.session_state['register_mode']=True; st.rerun()

        st.markdown("""
        <div style="margin-top:22px;padding-top:16px;border-top:1px solid #F1F5F9;
                    display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
          <span style="font-size:0.76rem;color:#94A3B8;display:flex;align-items:center;gap:4px;">🔒 Secure Login</span>
          <span style="font-size:0.76rem;color:#94A3B8;display:flex;align-items:center;gap:4px;">🛡️ Data Protected</span>
          <span style="font-size:0.76rem;color:#94A3B8;display:flex;align-items:center;gap:4px;">🤖 AI Powered</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  AUTHENTICATED APP
# ═══════════════════════════════════════════════════════
else:
    if 'active_menu' not in st.session_state:
        st.session_state.active_menu="🏠 หน้าหลัก"

    # ──────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="padding:22px 16px 14px;border-bottom:1px solid rgba(255,255,255,0.06);
                    margin-bottom:8px;display:flex;align-items:center;gap:10px;">
          <span style="font-size:26px;">🛡️</span>
          <div>
            <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-weight:800;
                        font-size:0.97rem;color:#fff;line-height:1.2;">TrueCheck AI</div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.35);
                        text-transform:uppercase;letter-spacing:1.2px;">Fake News Detector</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # User card
        role  = (st.session_state.get('role') or 'user').upper()
        uname = st.session_state.get('username','User')
        rc    = "#60A5FA" if role=="ADMIN" else "#64748B"
        st.markdown(f"""
        <div style="margin:6px 10px 14px;background:rgba(255,255,255,0.055);
                    border:1px solid rgba(255,255,255,0.08);border-radius:10px;
                    padding:11px 13px;display:flex;align-items:center;gap:10px;">
          <div style="width:32px;height:32px;border-radius:50%;background:rgba(30,136,229,0.30);
                      display:flex;align-items:center;justify-content:center;
                      font-size:15px;flex-shrink:0;">👤</div>
          <div style="min-width:0;">
            <div style="font-weight:600;font-size:0.86rem;color:#fff;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{uname}</div>
            <div style="font-size:0.68rem;font-weight:700;color:{rc};
                        text-transform:uppercase;letter-spacing:0.7px;">{role}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='padding:2px 14px 5px;font-size:0.67rem;font-weight:700;color:rgba(255,255,255,0.28);text-transform:uppercase;letter-spacing:1.2px;'>MENU</div>", unsafe_allow_html=True)

        nav=[("🏠 หน้าหลัก","🏠  หน้าหลัก"),
             ("📜 ประวัติการตรวจสอบ","📜  ประวัติการตรวจสอบ"),
             ("🔥 ข่าวที่เป็นกระแส","🔥  ข่าวที่เป็นกระแส"),
             ("👤 ข้อมูลส่วนตัว","👤  ข้อมูลส่วนตัว")]

        for key,label in nav:
            active=st.session_state.active_menu==key
            if st.button(label,key=f"nav_{key}",width="stretch",
                         type="primary" if active else "secondary"):
                st.session_state.active_menu=key; st.rerun()

        if st.session_state.get('role')=='admin':
            st.markdown("<div style='margin:10px 14px 0;padding-top:10px;border-top:1px solid rgba(255,255,255,0.07);font-size:0.67rem;font-weight:700;color:rgba(255,255,255,0.28);text-transform:uppercase;letter-spacing:1.2px;'>ADMIN PANEL</div>", unsafe_allow_html=True)
            admin_nav=[("📊 Dashboard","📊  Dashboard"),
                       ("📈 Model Performance","📈  Model Performance"),
                       ("📰 Manage News","📰  Manage News"),
                       ("💬 Review Feedback","💬  Review Feedback"),
                       ("🔬 System Analytics","🔬  System Analytics"),
                       ("👥 Manage Users","👥  Manage Users")]
            for key,label in admin_nav:
                active=st.session_state.active_menu==key
                if st.button(label,key=f"adm_{key}",width="stretch",
                             type="primary" if active else "secondary"):
                    st.session_state.active_menu=key; st.rerun()

        st.markdown("<hr style='margin:14px 0;'>",unsafe_allow_html=True)
        if st.button("  🚪  ออกจากระบบ",key="logout_btn",width="stretch"):
            db.log_system_event(user_id=st.session_state.get('user_id'),action="USER_LOGOUT",
                details=f"{uname} logged out",level="INFO")
            st.session_state['do_logout']=True; st.rerun()

    menu=st.session_state.active_menu

    # ══════════════════════════════════════
    # 🏠 HOME
    # ══════════════════════════════════════
    # --- เริ่มส่วน UI ---
    if menu == "🏠 หน้าหลัก":
        page_header("🔍","ตรวจสอบข่าว","วิเคราะห์เนื้อหาข่าวด้วย AI ")

        # แก้ Warning: ใส่ข้อความลงใน label แต่สั่งซ่อนไว้
        check_mode = st.radio(
            label="เลือกโหมดการตรวจสอบ", 
            options=["📝  พิมพ์ / วางเนื้อหา", "🔗  URL ลิงก์ข่าว"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("<div style='height:6px;'></div>",unsafe_allow_html=True)

        input_url = ""
        input_text = ""

        if check_mode == "🔗  URL ลิงก์ข่าว":
            # initialise session state key for URL
            if 'input_url' not in st.session_state:
                st.session_state['input_url'] = ""

            _url_col, _clr_url_col = st.columns([5, 1])
            with _url_col:
                st.text_input(
                    label="🔗 URL ของข่าว",
                    placeholder="https://www.example.com/news/...",
                    label_visibility="collapsed",
                    key="input_url"
                )
            with _clr_url_col:
                st.button("🗑️ ล้างข้อความ", type="secondary",
                          on_click=clear_url, use_container_width=True)
            input_url = st.session_state['input_url']
        else:
            with st.expander("💡 ลองใช้ข่าวตัวอย่าง (Demo)"):
                m1, m2 = st.columns(2)

                if m1.button("🚨 ตัวอย่างข่าวปลอม", use_container_width=True, type="primary"):
                    st.session_state['input_text'] = "ด่วนที่สุด! ครม. อนุมัติแล้ว แจกเงินช่วยเหลือเยียวยาพิเศษให้ประชาชนทุกคน คนละ 5,000 บาท สามารถเช็คสิทธิ์และลงทะเบียนรับเงินเข้าบัญชีได้ทันทีผ่านเว็บไซต์นี้"

                if m2.button("✅ ตัวอย่างข่าวจริง", use_container_width=True):
                    st.session_state['input_text'] = "คณะกรรมการนโยบายการเงิน (กนง.) ธนาคารแห่งประเทศไทย มีมติปรับขึ้นอัตราดอกเบี้ยนโยบาย 0.25% ต่อปี จาก 2.25% เป็น 2.50% ต่อปี โดยให้มีผลทันที เพื่อเป็นการดูแลอัตราเงินเฟ้อให้อยู่ในกรอบเป้าหมายอย่างยั่งยืน และสอดคล้องกับแนวโน้มเศรษฐกิจที่กำลังฟื้นตัว"

            # ปุ่มล้างข้อความ อยู่นอก expander
            _, _clr_col = st.columns([5, 1])
            with _clr_col:
                st.button("🗑️ ล้างข้อความ", type="secondary", on_click=clear_text,
                          use_container_width=True)

            # เพิ่มกล่องพิมพ์ข้อความ
            st.text_area(
                label="กรอกเนื้อหาข่าว",
                height=180,
                placeholder="วางหรือพิมพ์เนื้อหาข่าวที่ต้องการตรวจสอบที่นี่...",
                label_visibility="collapsed",
                key="input_text"
            )
            # ดึงข้อความจากหน้าเว็บมาเก็บในตัวแปร เพื่อเตรียมส่งให้ AI ตรวจสอบ
            input_text = st.session_state['input_text']

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        if st.button("🚀  วิเคราะห์ข่าวนี้", type="primary", width="stretch"):
            clean = ""
            if check_mode=="🔗  URL ลิงก์ข่าว":
                if not input_url: st.warning("กรุณาวาง URL ก่อนกด"); st.stop()
                with st.spinner("กำลังดึงข้อมูลจากลิงก์..."):
                    db.log_system_event(user_id=st.session_state.get('user_id'), action="URL_FETCH",
                    details=f"Fetched: {input_url}", level="INFO")
                    title,content=get_content_from_url(input_url)
                if title and not str(content).startswith("Error"):
                    clean=f"{title}\n\n{content}"
                else:
                    st.error(f"ดึงข้อมูลไม่ได้: {content}")
                    db.log_system_event(user_id=st.session_state.get('user_id'),action="API_ERROR",
                                        details=f"URL fetch failed: {input_url}",level="ERROR"); st.stop()
            else:
                clean=str(input_text).strip()
                if not clean: st.warning("กรุณาใส่เนื้อหาข่าว"); st.stop()

            with st.spinner("🧠 AI กำลังวิเคราะห์..."):
                try:
        # ✅ โหลด pipeline ก่อน แล้วส่งเข้าไปด้วย
                    pipeline = ai.get_pipeline()
                    result = ai.predict_news(re.sub(r'\s+',' ', clean).strip(), pipeline)
        
                    if result:
                        time.sleep(0.35)
                        rl, rc = result.get('result'), result.get('confidence')
                        uname = st.session_state.get('username', 'Unknown')
                        db.log_system_event(user_id=st.session_state.get('user_id'), action="PREDICT",
                        details=f"[{uname}] ทำนาย: '{clean[:50]}' → {rl} ({rc}%)", level="INFO")
                        pid = db.create_prediction(st.session_state.get('user_id'), clean[:50]+"…",
                                                clean, input_url or None, rl, rc)
                        st.session_state.update({'current_result': result, 'current_pred_id': pid, 'feedback_given': False,'current_text':  clean})
                        st.rerun()
                    else:
                        st.error("AI ไม่ตอบสนอง")
                        db.log_system_event(user_id=st.session_state.get('user_id'), action="API_ERROR",
                                details="None result", level="ERROR")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")

        # Result
        if 'current_result' in st.session_state:
            res   = st.session_state['current_result']
            label = res['result']
            conf  = float(res['confidence'])
            cat   = res.get('category', 'ไม่ระบุ')

            # ✅ ดึงจาก current_text แทน input_text
            raw_text = st.session_state.get('current_text', '')
            preview  = re.sub(r'\s+', ' ', str(raw_text)).strip()[:150]
            preview  = (preview + "…") if len(str(raw_text)) > 150 else preview

            if conf < 70:
                cfg = dict(bg="#FFFBEB", border="#F59E0B", bc="#92400E", bbg="#FEF3C7",
                        icon="⚠️", verdict="UNVERIFIED", bar="#F59E0B",
                        desc="AI ยังไม่มีความมั่นใจเพียงพอ — ข้อมูลอาจไม่ครบถ้วน ควรตรวจสอบจากแหล่งอื่นด้วย")
            elif label == "Fake":
                cfg = dict(bg="#FFF5F5", border="#EF4444", bc="#991B1B", bbg="#FEE2E2",
                        icon="🚨", verdict="FAKE NEWS", bar="#EF4444",
                        desc="เนื้อหานี้มีลักษณะเป็นข่าวปลอมหรือข้อมูลบิดเบือน — กรุณาตรวจสอบแหล่งที่มาก่อนแชร์")
            else:
                cfg = dict(bg="#F0FDF4", border="#22C55E", bc="#14532D", bbg="#DCFCE7",
                        icon="✅", verdict="REAL NEWS", bar="#22C55E",
                        desc="เนื้อหาดูน่าเชื่อถือและสมเหตุสมผล — ควรอ้างอิงแหล่งข้อมูลหลักเสมอ")

            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

            st.markdown(f"""
        <div style="background:{cfg['bg']};border:1.5px solid {cfg['border']};
                    border-radius:16px;padding:26px 28px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <div style="display:flex;align-items:flex-start;gap:15px;margin-bottom:18px;">
            <span style="font-size:2rem;line-height:1;flex-shrink:0;">{cfg['icon']}</span>
            <div style="flex:1;">
            <span style="display:inline-block;background:{cfg['bbg']};color:{cfg['bc']};
                        font-family:'IBM Plex Sans Thai',sans-serif;font-weight:800;
                        font-size:1.15rem;padding:5px 16px;border-radius:8px;
                        letter-spacing:-0.2px;">{cfg['verdict']}</span>
            <div style="margin-top:9px;font-size:0.9rem;color:#475569;line-height:1.55;">
                {cfg['desc']}
            </div>
            </div>
        </div>
        
        {'<div style="background:rgba(0,0,0,0.04);border-radius:10px;padding:12px 16px;margin-bottom:14px;"><div style="font-size:0.72rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">📄 จากเนื้อหาข่าว</div><div style="font-size:0.88rem;color:#334155;line-height:1.6;font-style:italic;">&quot;' + preview + '&quot;</div></div>' if preview else ''}
        <div style="margin-bottom:14px;">
            <span style="font-size:0.75rem;font-weight:600;color:#64748B;">📂 หมวดหมู่ข่าว</span>
            &nbsp;
            <span style="background:#EFF6FF;color:#1148A8;font-size:0.82rem;font-weight:700;
                        padding:4px 12px;border-radius:99px;border:1px solid #BFDBFE;">
            {cat}
            </span>
        </div>
        <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:0.8rem;font-weight:600;color:#64748B;">AI Confidence</span>
            <span style="font-size:0.88rem;font-weight:800;color:{cfg['bc']};">{conf:.1f}%</span>
            </div>
            <div style="background:rgba(0,0,0,0.07);border-radius:99px;height:7px;overflow:hidden;">
            <div style="width:{conf}%;height:100%;background:{cfg['bar']};
                        border-radius:99px;"></div>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

# Feedback
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
            padding:18px 20px 10px;">
  <div style="font-weight:700;font-size:0.9rem;color:#1E293B;">
    💬 AI ทายถูกหรือเปล่า?
  </div>
  <div style="font-size:0.8rem;color:#94A3B8;margin:3px 0 12px;">
    Feedback ของคุณช่วยให้ AI แม่นยำขึ้น
  </div>
</div>""", unsafe_allow_html=True)
            if not st.session_state.get('feedback_given'):
                fc1,fc2=st.columns(2)
                with fc1:
                    if st.button("👍  ถูกต้อง — AI ทายถูก",type="primary",width="stretch"):
                        db.save_feedback(st.session_state['current_pred_id'], "Correct")
                        db.log_system_event(user_id=st.session_state.get('user_id'), action="FEEDBACK",
                        details=f"[{st.session_state.get('username')}] Feedback: Correct (pred_id: {st.session_state['current_pred_id']})", level="INFO")

                        st.session_state['feedback_given']=True; st.toast("ขอบคุณ!"); time.sleep(0.7); st.rerun()
                with fc2:
                    if st.button("👎  ไม่ถูกต้อง — AI ทายผิด",width="stretch"):
                        db.save_feedback(st.session_state['current_pred_id'],"Incorrect")
                        db.log_system_event(user_id=st.session_state.get('user_id'), action="FEEDBACK",
                        details=f"[{st.session_state.get('username')}] Feedback: Incorrect (pred_id: {st.session_state['current_pred_id']})", level="INFO")
                        st.session_state['feedback_given']=True; st.toast("ขอบคุณ! เราจะนำไปปรับปรุง"); time.sleep(0.7); st.rerun()
            else:
                st.success("✅ ส่ง Feedback แล้ว — ขอบคุณที่ช่วยพัฒนา AI!")

    # ══════════════════════════════════════
    # 📜 HISTORY
    # ══════════════════════════════════════
    elif menu=="📜 ประวัติการตรวจสอบ":
        page_header("📜","ประวัติการตรวจสอบ","ข่าวทุกรายการที่คุณเคยวิเคราะห์")
        uid=st.session_state.get('user_id')
        if uid:
            hist=db.get_user_history(uid)
            if hist:
                df=pd.DataFrame(hist); df.columns=[c.lower() for c in df.columns]
                sq = st.text_input("ค้นหาหัวข้อข่าว", "", placeholder="🔍 ค้นหาหัวข้อข่าว...", label_visibility="collapsed")
                if sq: df=df[df['title'].str.contains(sq,case=False,na=False)]
                if not df.empty:
                    if 'timestamp' in df.columns:
                        df['timestamp']=pd.to_datetime(df['timestamp']).dt.strftime('%d %b %Y, %H:%M')
                    rmap={'title':'หัวข้อข่าว','result':'ผลลัพธ์','confidence':'ความมั่นใจ (%)','timestamp':'วันที่-เวลา'}
                    vc=[c for c in rmap if c in df.columns]
                    dfd = df[vc].rename(columns=rmap)
                    dfd.index = pd.RangeIndex(1, len(dfd) + 1)
                    st.caption(f"พบ {len(dfd)} รายการ")
                    st.dataframe(dfd, width="stretch")
                else:
                    st.warning(f"ไม่พบผลลัพธ์สำหรับ '{sq}'")
            else:
                st.markdown("""
                <div style="text-align:center;padding:64px 20px;">
                  <div style="font-size:52px;margin-bottom:16px;">📭</div>
                  <div style="font-size:1.05rem;font-weight:700;color:#334155;margin-bottom:8px;">
                    ยังไม่มีประวัติการตรวจสอบ
                  </div>
                  <div style="font-size:0.88rem;color:#94A3B8;">
                    กลับไปหน้าหลักเพื่อเริ่มตรวจสอบข่าวรายการแรก
                  </div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════
    # 🔥 TRENDING
    # ══════════════════════════════════════
    elif menu == "🔥 ข่าวที่เป็นกระแส":
            page_header("🔥", "ข่าวที่เป็นกระแส", "ข่าวสารที่ถูกพูดถึงและผ่านการตรวจสอบโดยทีมงาน")
            df = db.get_all_trending()
            if df.empty:
                st.info("ℹ️ ยังไม่มีข่าวที่เป็นกระแสในขณะนี้")
            else:
                # ✅ เพิ่ม filter bar
                col_s, col_f = st.columns([1, 2])
                with col_s:
                    filter_label = st.selectbox(
                        "กรองตามประเภท",
                        ["ทั้งหมด", "Real", "Fake"],
                        label_visibility="collapsed"
                    )
                with col_f:
                    search_q = st.text_input(
                        "ค้นหา", placeholder="🔍 ค้นหาข่าว...",
                        label_visibility="collapsed"
                    )

                # ✅ filter ข้อมูล
                df_filtered = df.copy()
                if filter_label != "ทั้งหมด":
                    df_filtered = df_filtered[df_filtered['label'] == filter_label]
                if search_q:
                    df_filtered = df_filtered[
                        df_filtered['headline'].str.contains(search_q, case=False, na=False) |
                        df_filtered['content'].str.contains(search_q, case=False, na=False)
                    ]

                # ✅ summary badges
                n_real       = len(df[df['label'] == 'Real'])
                n_fake       = len(df[df['label'] == 'Fake'])
                

                st.markdown(f"""
                <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;">
                <span style="background:#F1F5F9;color:#475569;font-size:0.78rem;font-weight:700;
                            padding:4px 12px;border-radius:99px;cursor:pointer;">
                    📰 ทั้งหมด {len(df)}
                </span>
                <span style="background:#DCFCE7;color:#166534;font-size:0.78rem;font-weight:700;
                            padding:4px 12px;border-radius:99px;">
                    ✅ Real {n_real}
                </span>
                <span style="background:#FEE2E2;color:#991B1B;font-size:0.78rem;font-weight:700;
                            padding:4px 12px;border-radius:99px;">
                    🚨 Fake {n_fake}
                </span>
            
                </div>""", unsafe_allow_html=True)

                st.caption(f"แสดง {len(df_filtered)} รายการ")

                if df_filtered.empty:
                    st.info("ไม่พบข่าวที่ตรงกับเงื่อนไข")
                else:
                    lcfg = {
                        "Fake":       ("#FEE2E2","#991B1B","#EF4444","🚨"),
                        "Real":       ("#DCFCE7","#166534","#22C55E","✅"),
                
                    }
                    for _, row in df_filtered.iterrows():
                        lc = lcfg.get(row['label'], ("#F1F5F9","#475569","#CBD5E1","📰"))
                        ts = str(row.get('updated_at', '-')).replace("T", " ")[:16]
                        st.markdown(f"""
                        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
                                    padding:20px 24px;margin-bottom:12px;
                                    box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                        <div style="display:flex;align-items:flex-start;
                                    justify-content:space-between;gap:12px;margin-bottom:10px;">
                            <h4 style="margin:0;color:#1E293B !important;font-size:0.97rem !important;
                                    line-height:1.45;flex:1;">{lc[3]}&nbsp;{row['headline']}</h4>
                            <span style="flex-shrink:0;background:{lc[0]};color:{lc[1]};
                                        font-size:0.71rem;font-weight:800;padding:3px 11px;
                                        border-radius:99px;text-transform:uppercase;letter-spacing:0.5px;
                                        border:1px solid {lc[2]}44;white-space:nowrap;">{row['label']}</span>
                        </div>
                        <p style="color:#475569;font-size:0.88rem;line-height:1.6;margin:0 0 10px;">
                            {row['content']}
                        </p>
                        <div style="font-size:0.76rem;color:#94A3B8;">🕒 อัปเดตเมื่อ {ts}</div>
                        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════
    # 👤 PROFILE
    # ══════════════════════════════════════
    elif menu == "👤 ข้อมูลส่วนตัว":
        for k, v in [('username', "ผู้ใช้งานทั่วไป"), ('email', ""), ('edit_email_mode', False)]:
            if k not in st.session_state:
                st.session_state[k] = v
        page_header("👤", "ข้อมูลส่วนตัว", "จัดการบัญชีและการตั้งค่า")
        uid = st.session_state.get('user_id')
        check_count = 0
        if uid:
            h=db.get_user_history(uid)
            if h:
                df=pd.DataFrame(h); df.columns=[c.lower() for c in df.columns]; check_count=len(df)
            if 'email' not in st.session_state or not st.session_state.email:
                st.session_state.email=db.get_user_email(uid)
        

        # Profile banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1148A8 0%,#1565C0 60%,#0097A7 100%);
                    border-radius:16px;padding:28px 26px;margin-bottom:20px;
                    display:flex;align-items:center;gap:20px;
                    box-shadow:0 4px 20px rgba(17,72,168,0.18);">
          <div style="width:62px;height:62px;border-radius:50%;
                      background:rgba(255,255,255,0.16);display:flex;
                      align-items:center;justify-content:center;
                      font-size:28px;flex-shrink:0;">👤</div>
          <div style="flex:1;min-width:0;">
            <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:1.35rem;
                        font-weight:800;color:#fff;">{st.session_state.username}</div>
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.60);margin-top:2px;">
              สมาชิก TrueCheck AI
            </div>
          </div>
          <div style="text-align:center;background:rgba(255,255,255,0.12);
                      border-radius:12px;padding:13px 22px;flex-shrink:0;">
            <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-size:1.7rem;
                        font-weight:800;color:#fff;">{check_count}</div>
            <div style="font-size:0.7rem;color:rgba(255,255,255,0.55);
                        text-transform:uppercase;letter-spacing:0.5px;margin-top:1px;">ข่าวที่ตรวจสอบ</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Settings
        st.markdown("""<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;
        padding:22px 26px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-family:'IBM Plex Sans Thai',sans-serif;font-weight:700;font-size:0.95rem;
        color:#1E293B;margin-bottom:16px;">ข้อมูลบัญชี</div>""",unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:12px 0;border-bottom:1px solid #F1F5F9;">
          <div>
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.6px;color:#94A3B8;margin-bottom:3px;">USERNAME</div>
            <div style="font-size:0.93rem;font-weight:600;color:#1E293B;">
              {st.session_state.username}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="padding-top:13px;">
        <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.6px;color:#94A3B8;margin-bottom:7px;">EMAIL</div>
        </div>""", unsafe_allow_html=True)

        ec,eb=st.columns([4,1])
        new_email=""
        with ec:
            if st.session_state.edit_email_mode:
                new_email=st.text_input("",value=st.session_state.email,
                                         label_visibility="collapsed",placeholder="your@email.com")
            else:
                v=st.session_state.email or "ยังไม่ได้ระบุ"
                co="#1E293B" if st.session_state.email else "#94A3B8"
                st.markdown(f"<div style='font-size:0.92rem;font-weight:600;color:{co};padding:10px 0;'>{v}</div>",unsafe_allow_html=True)
        with eb:
            if st.session_state.edit_email_mode:
                if st.button("Save",key="save_email",type="primary",width="stretch"):
                    if not new_email.strip(): st.warning("กรุณากรอกอีเมล")
                    elif db.update_user_email(uid,new_email):
                        st.session_state.email=new_email; st.session_state.edit_email_mode=False; st.rerun()
                    else: st.error("บันทึกไม่ได้")
                if st.button("✕",key="cancel_email",width="stretch"):
                    st.session_state.edit_email_mode=False; st.rerun()
            else:
                lbl="✏️ แก้ไข" if st.session_state.email else "➕ เพิ่ม"
                if st.button(lbl,key="edit_email",width="stretch"):
                    st.session_state.edit_email_mode=True; st.rerun()

        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("<div style='height:14px;'></div>",unsafe_allow_html=True)
        if st.button("🚪 ออกจากระบบ (Logout)",type="primary"):
            db.log_system_event(user_id=uid,action="USER_LOGOUT",
                                details=f"{st.session_state.get('username')} logged out",level="INFO")
            st.session_state['do_logout']=True; st.rerun()

    # ══════════════════════════════════════
    # ADMIN ROUTES
    # ══════════════════════════════════════
    admin_keys={"📊 Dashboard","📈 Model Performance","📰 Manage News",
                "💬 Review Feedback","🔬 System Analytics","👥 Manage Users"}

    if menu in admin_keys:
        if st.session_state.get('role')!='admin':
            st.error("⛔ Access Denied — หน้านี้สำหรับ Admin เท่านั้น"); st.stop()

        if menu=="📊 Dashboard":
            page_header("📊","Admin Dashboard","สรุปภาพรวมประสิทธิภาพระบบ (Real-time)")
            stats=db.get_dashboard_kpi() or {"checks_today":0,"active_users":0,"accuracy":0.0,"feedback_total":0}
            c1,c2,c3,c4=st.columns(4)
            with c1: kpi_card("🔍","Checks Today",       f"{stats.get('checks_today',0):,}")
            with c2: kpi_card("👥","Active Users (24h)", f"{stats.get('active_users',0):,}")
            with c3:
                acc=stats.get('accuracy',0.0)
                kpi_card("🎯","Model Accuracy Today",f"{acc}%","✅ ปกติ" if acc>=50 else "⚠️ ต่ำกว่าเกณฑ์",acc>=50)
            with c4: kpi_card("💬","Feedback Total Today",f"{stats.get('feedback_total',0):,}")

            st.markdown("<div style='height:24px;'></div>",unsafe_allow_html=True)
            section_title("⏱️ Recent Activity","การใช้งานล่าสุด")
            logs=db.get_system_logs(limit=10)
            if logs:
                for row in logs:
                    ts,user,action,details,level=row
                    icon="🛡️" if "admin" in str(user).lower() else "👤"
                    lbg={"ERROR":"#FFF5F5","WARNING":"#FFFBEB"}.get(level,"#F8FAFC")
                    lborder={"ERROR":"#FECACA","WARNING":"#FDE68A"}.get(level,"#E2E8F0")
                    st.markdown(f"""
                    <div style="background:{lbg};border:1px solid {lborder};border-radius:10px;
                                padding:13px 16px;margin-bottom:7px;
                                display:flex;align-items:center;gap:13px;">
                      <span style="font-size:1.2rem;">{icon}</span>
                      <div style="flex:1;min-width:0;">
                        <div style="font-weight:700;font-size:0.87rem;color:#1E293B;">{user}</div>
                        <div style="font-size:0.8rem;color:#64748B;white-space:nowrap;
                                    overflow:hidden;text-overflow:ellipsis;">
                          {action} · {(details[:80]+'…') if len(details)>80 else details}
                        </div>
                      </div>
                      <div style="font-size:0.76rem;color:#94A3B8;white-space:nowrap;">{time_ago(ts)}</div>
                    </div>""", unsafe_allow_html=True)
            else: st.info("ยังไม่มีประวัติ")

        elif menu=="📈 Model Performance":
            page_header("📈","Model Performance","ความแม่นยำและการทำนายของ AI Model")
            show_model_performance()

        elif menu=="💬 Review Feedback":
            show_feedback_review()

        elif menu=="📰 Manage News":
            manage_trending_news()

        elif menu=="🔬 System Analytics":
            page_header("🔬","System Analytics","วิเคราะห์การใช้งานและแนวโน้มของระบบ")
            show_admin_dashboard_enhanced()
            st.markdown("<hr style='margin:24px 0;'>",unsafe_allow_html=True)
            show_system_analytics()

        elif menu=="👥 Manage Users":
            page_header("👥","Manage Users","จัดการบัญชีผู้ใช้งาน สิทธิ์ และสถานะ")
            manage_users_page()