# frontend.py
import streamlit as st
import requests
from collections import Counter

# Setup หน้าเว็บ
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

st.title("🕵️ Fake News Detection System")
st.write("Architecture: Streamlit (Frontend) <--> FastAPI (Backend)")

# ช่องรับข้อมูล
news_text = st.text_area("วางเนื้อหาข่าวที่ต้องการตรวจสอบ:", height=150)

# URL ของ API (ยิงเข้าหลังบ้านตัวเอง)
API_URL = "http://localhost:8000/predict"

if st.button("ตรวจสอบความถูกต้อง"):
    if not news_text:
        st.warning("กรุณาใส่เนื้อหาข่าวก่อนครับ")
    else:
        with st.spinner('กำลังส่งข้อมูลไปประมวลผลที่ API...'):
            try:
                # 🔥 หัวใจสำคัญ: ส่ง Request ไปหา API
                payload = {"text": news_text}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json() # แกะกล่อง JSON
                    
                    # แสดงผล (Code ส่วนแสดงผลเดิมของคุณ เป๊ะๆ)
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['label'] == 'ข่าวจริง': 
                            st.success(f"## ✅ {result['label']}")
                        else:
                            st.error(f"## 🚨 {result['label']}")
                        
                        st.metric("ความมั่นใจ (Confidence)", f"{result['probability']*100:.2f}%")
                    
                    with col2:
                        st.info(f"**หมวดหมู่หลัก:** {result['category']}")
                        # st.write("**🕵️ เพื่อนบ้าน 10 อันดับแรก:**")
                        
                        # neighbor_cats = result.get('neighbor_cats', [])
                        # if neighbor_cats:
                        #     for i, cat in enumerate(neighbor_cats):
                        #         st.markdown(f"**{i+1}.** <span style='color:gray'>(หมวด: {cat})</span>", unsafe_allow_html=True)
                        # else:
                        #     st.write("- ไม่พบข้อมูล")

                    # Debug
                    with st.expander("🔍 API Response Debug"):
                        st.json(result)

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("❌ เชื่อมต่อ API ไม่ได้")
                st.warning("💡 อย่าลืมรันไฟล์ api.py ใน Terminal อีกจอนึงนะครับ!")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")