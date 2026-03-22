from supabase import create_client
import httpx
import sys
import os


# 1. ใส่ URL และ KEY ตรงๆ เพื่อตัดปัญหาไฟล์ .env
SUPABASE_URL = "https://orxtfxdernqmpkfmsijj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9yeHRmeGRlcm5xbXBrZm1zaWpqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzEyMDQ5OTgsImV4cCI6MjA4Njc4MDk5OH0.6dDVQio5hQpTQj6jnnS6yZBqR2GBReqFwazza6TqolQ" 

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