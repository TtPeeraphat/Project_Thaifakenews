from supabase import create_client
import httpx
import sys
import os
from config import config


SUPABASE_URL = config.database.supabase_url
SUPABASE_KEY = config.database.supabase_key
print("กำลังทดสอบเชื่อมต่อ...")

try:
    # ทดสอบยิง Request แบบพื้นฐานที่สุด
    response = httpx.get(SUPABASE_URL)
    print(f"✅ 1. การเชื่อมต่ออินเทอร์เน็ตผ่าน Python: ผ่าน! (Status: {response.status_code})")
    
    # ทดสอบสร้าง Supabase Client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ 2. การสร้าง Supabase Client: ผ่าน!")
    
except Exception as e:
    print(f"❌ พัง!: {e}")
    
from config import config

# ใช้ repr() เพื่อตีแผ่พวกช่องว่าง หรือเครื่องหมายคำพูดที่ซ่อนอยู่
print("👉 URL จาก Config คือ:", repr(config.database.supabase_url))