# คู่มือการติดตั้งและการใช้งานโปรเจค (Project Setup & Installation)

คู่มือนี้ครอบคลุมขั้นตอนการติดตั้ง การตั้งค่าระบบ การขอ Credentials และการรันโปรเจค Fake News Detection

---

## 1. การติดตั้งเครื่องมือพื้นฐาน

### การติดตั้ง Python
1. ดาวน์โหลด Python ได้จาก [python.org/downloads](https://www.python.org/downloads/) (แนะนำเวอร์ชัน 3.10 ขึ้นไป)
2. เปิดไฟล์ติดตั้ง และ **ทำเครื่องหมายถูกที่ "Add Python to PATH"** ก่อนกดคลิกตั้ง
3. ตรวจสอบการติดตั้งใน Terminal หรือ Command Prompt:
```bash
python --version
การติดตั้งแพ็กเกจที่จำเป็นเบื้องต้นเปิด Terminal หรือ Command Prompt แล้วพิมพ์คำสั่งต่อไปนี้เพื่อติดตั้งแพ็กเกจพื้นฐาน:Bashpip install streamlit fastapi uvicorn
ตรวจสอบการติดตั้ง Streamlit:Bashstreamlit --version
2. การติดตั้งโปรเจค2.1 การ Clone โค้ดจาก GitHubเปิด Terminal และดำเนินการ Clone โค้ดลงมาที่เครื่องของคุณ:Bashgit clone [https://github.com/TtPeeraphat/Project_Thaifakenews.git](https://github.com/TtPeeraphat/Project_Thaifakenews.git)
2.2 การติดตั้ง Dependencies ทั้งหมดเปิดโฟลเดอร์โปรเจคใน VS Code จากนั้นเปิด Terminal ภายในโปรเจคแล้วรันคำสั่ง:Bashpip install -r requirements.txt
3. การตั้งค่า Credentialsระบบต้องการค่า Credentials 6 ตัวสำหรับการทำงาน ดังตารางต่อไปนี้:Variableวัตถุประสงค์การใช้งานSUPABASE_URLURL ของ Supabase project สำหรับเชื่อมต่อฐานข้อมูลSUPABASE_KEYAPI key สำหรับเชื่อมต่อ Supabase (anon public key)HF_TOKENHugging Face token สำหรับโหลดโมเดล WangchanBERTaAPIFY_API_TOKENToken สำหรับ Apify bot ดึงเนื้อหาข่าวจาก URLGMAIL_EMAILอีเมล Gmail ที่ใช้ส่งรหัส OTP แก่ผู้ใช้งานGMAIL_APP_PASSWORDApp Password ของ Gmail (ไม่ใช่รหัสผ่าน Gmail ปกติ)กรณีที่ 1: รันบนเครื่องตัวเอง (Local Development)สร้างไฟล์ .env ที่ root ของโปรเจค (โฟลเดอร์หลัก) แล้วกรอกค่าดังต่อไปนี้:Code snippetSUPABASE_URL=[https://xxxxxxxxxxxx.supabase.co](https://xxxxxxxxxxxx.supabase.co)
SUPABASE_KEY=your_supabase_anon_key_here
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
APIFY_API_TOKEN=apify_api_xxxxxxxxxxxxxxxxxxxxxxxx
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
คำเตือน: ห้าม commit ไฟล์ .env เข้า Git เด็ดขาด ตรวจสอบให้แน่ใจว่าไฟล์ .gitignore มีบรรทัด .env อยู่แล้วก่อนเสมอกรณีที่ 2: Deploy บน Streamlit CloudStreamlit Cloud ไม่รองรับไฟล์ .env จึงต้องตั้งค่าผ่าน Secrets ใน Dashboardไปที่ share.streamlit.io แล้ว Loginเลือก App ที่ต้องการ คลิกเมนูจุดสามจุด (⋮) และเลือก Settingsคลิกแท็บ Secrets แล้วกรอกค่าในรูปแบบต่อไปนี้:Ini, TOMLSUPABASE_URL = "[https://xxxxxxxxxxxx.supabase.co](https://xxxxxxxxxxxx.supabase.co)"
SUPABASE_KEY = "your_supabase_anon_key_here"
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
APIFY_API_TOKEN = "apify_api_xxxxxxxxxxxxxxxxxxxxxxxx"
GMAIL_EMAIL = "your_email@gmail.com"
GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"
คลิก Save แล้วรอ App restart อัตโนมัติ (ระบบจะเข้าถึงค่าเหล่านี้ผ่าน st.secrets["KEY"] โดยไม่ต้องแก้โค้ดเพิ่ม)
4. วิธีการขอ Credentials จาก Services ต่างๆSupabase (SUPABASE_URL, SUPABASE_KEY)ไปที่ Supabase และ Sign in หรือ Sign upคลิก New Project กรอกชื่อและตั้ง Database passwordไปที่ Project Settings -> APIคัดลอก Project URL และ anon public keyHugging Face Token (HF_TOKEN)ไปที่ Hugging Face Settingsคลิก New token ตั้งชื่อ (เช่น truecheckAI) และเลือก Role: Readคลิก Generate แล้วคัดลอก Token ทันที (Token จะแสดงให้เห็นเพียงครั้งเดียวเท่านั้น)Apify API Token (APIFY_API_TOKEN)ไปที่ Apify Consoleคัดลอก Personal API TokenGmail App Password (GMAIL_APP_PASSWORD)ไปที่ Google Account Securityเปิดใช้งาน 2-Step Verification ให้เรียบร้อยค้นหาเมนู App passwords กรอกชื่อแอป (เช่น TrueCheckAI) และคลิก Createนำรหัส 16 หลักที่ได้มาใช้งาน (ห้ามใช้รหัสผ่าน Gmail ปกติเด็ดขาด)
5. การรันระบบข้อสำคัญ: ต้องเปิด Terminal 2 หน้าต่างพร้อมกัน และรัน Backend ก่อน Frontend เสมอTerminal หน้าต่างที่ 1 — รัน FastAPI Backend:Bashuvicorn api:app --reload --port 8000
หมายเหตุ: รอให้เห็นข้อความ Application startup complete ก่อนเปิด FrontendTerminal หน้าต่างที่ 2 — รัน Streamlit Frontend:Bashstreamlit run frontend.py
เปิดเว็บเบราว์เซอร์และไปที่ http://localhost:8501 เพื่อเริ่มต้นใช้งาน
6. การตั้งค่า Jupyter Notebook สำหรับนักพัฒนาหากต้องการพัฒนาโมเดลต่อหรือดูผลการทดลอง สามารถตั้งค่า Jupyter Notebook ใน VS Code ได้ดังนี้:การสร้าง Virtual Environment และติดตั้ง Jupyterสร้างโฟลเดอร์ Environment ผ่าน Terminal:Bashpython -m venv .venv
ติดตั้ง Jupyter:Bashpip install jupyter
การใช้งาน Notebookสร้างไฟล์ใหม่โดยให้มีนามสกุล .ipynb (เช่น new_main.ipynb)เมื่อกด Run cell ครั้งแรก ให้คลิก Install ipykernel ตามที่ VS Code แจ้งเตือนตรวจสอบให้แน่ใจว่า Kernel ที่มุมขวาบนของ VS Code ถูกเลือกเป็น .venv (Python 3.x.x) ก่อนรันเสมอไฟล์ Notebook ที่มีในโปรเจค:ไฟล์วัตถุประสงค์main_progress.ipynbติดตามความคืบหน้าและผลลัพธ์การ Training โมเดลcompare.ipynbเปรียบเทียบประสิทธิภาพของโมเดลแต่ละแบบ
