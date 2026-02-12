import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. โหลด Artifacts ครั้งเดียว (Singleton) เพื่อความเร็ว
# ไม่ต้องโหลดทุกครั้งที่มีคนกดปุ่ม จะทำให้เว็บช้า
try:
    with open('artifacts.pkl', 'rb') as f:
        data = pickle.load(f)
        
    vectorizer = data['vectorizer'] # ตัวแปลงภาษา
    x_db = data['x_np']             # ฐานข้อมูลข่าวเก่า (Vector)
    y_db = data['y_label_np']       # เฉลยข่าวเก่า
    id2label = data['id2label']     # {0: จริง, 1: ปลอม}
    k_neighbors = data.get('k', 5)  # จำนวนเพื่อนบ้านที่จะหา
    
    print("✅ AI Engine Loaded Successfully!")

except Exception as e:
    print(f"⚠️ Error loading artifacts: {e}")
    # สร้าง Dummy ไว้กัน Error (เผื่อไฟล์ยังไม่มา)
    vectorizer = None

# ==========================================
# ฟังก์ชันหลัก: ทำนายข่าว (Predict)
# ==========================================
def predict_news(text):
    if vectorizer is None:
        return {"status": "Error", "message": "Model not loaded"}

    # 1. แปลงข้อความใหม่ เป็น Vector
    # (ต้องใช้ transform นะครับ ห้าม fit_transform ใหม่)
    input_vec = vectorizer.transform([text]).toarray() 

    # 2. คำนวณความเหมือน (Cosine Similarity) กับข่าวเก่าทั้งหมด
    # เพื่อดูว่าข่าวนี้น่าจะเป็นพวกเดียวกับกลุ่มไหน
    similarities = cosine_similarity(input_vec, x_db)
    
    # 3. หา 5-10 อันดับที่เหมือนที่สุด (Neighbors)
    # indices คือลำดับของข่าวที่เหมือนที่สุด
    top_k_indices = similarities[0].argsort()[-k_neighbors:][::-1]
    
    # 4. ดูว่าเพื่อนบ้านส่วนใหญ่เป็นข่าวจริงหรือปลอม (Voting)
    neighbor_labels = y_db[top_k_indices]
    fake_count = np.sum(neighbor_labels == 1) # สมมติ 1 = Fake
    
    # คำนวณ % ความมั่นใจ
    confidence = (fake_count / k_neighbors) * 100
    
    # สรุปผล
    if confidence > 50:
        result = "Fake"
        final_conf = confidence
    else:
        result = "Real"
        final_conf = 100 - confidence

    # 5. ดึงข้อความของข่าวที่คล้ายกันมาโชว์ (Explainability)
    # (สมมติว่าคุณเก็บ Text ข่าวเก่าไว้ด้วย ถ้าไม่ได้เก็บก็ข้ามส่วนนี้)
    # similar_news = [text_db[i] for i in top_k_indices] 

    return {
        "result": result,           # "Fake" หรือ "Real"
        "confidence": final_conf,   # เช่น 85.5
        "neighbor_ids": top_k_indices.tolist() # ส่ง ID ข่าวที่คล้ายกลับไป (เผื่อใช้)
    }