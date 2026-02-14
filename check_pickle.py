# ไฟล์ check_pickle.py
import pickle

try:
    with open('artifacts.pkl', 'rb') as f:
        data = pickle.load(f)
        print("✅ เปิดไฟล์สำเร็จ!")
        print("🔑 Keys ทั้งหมดที่มีในไฟล์:", data.keys())
        
        # ลองเช็คว่ามันเก็บ Vectorizer ไว้ชื่ออะไร
        if 'vectorizer' in data:
            print("พบ vectorizer (ถูกต้อง)")
        else:
            print("❌ ไม่พบ 'vectorizer'! คุณอาจจะตั้งชื่ออื่น?")
except Exception as e:
    print(f"❌ อ่านไฟล์ไม่ได้: {e}")