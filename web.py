import streamlit as st
import torch
import numpy as np
import pickle
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from collections import Counter
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
st.set_page_config(page_title="AI ข่าวปลอม Detector", page_icon="🕵️")

@st.cache_resource
def load_resources():
    # A) โหลด BERT (ใช้เวลาแป๊บนึง)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    lm_model = AutoModel.from_pretrained('xlm-roberta-base')
    
    # B) โหลดข้อมูลประกอบ
    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # C) สร้างระบบค้นหา (KNN) ขึ้นมาใหม่จากข้อมูลที่เซฟมา
    nbrs = NearestNeighbors(n_neighbors=artifacts['k'], metric='cosine')
    nbrs.fit(artifacts['x_np'])
    
    return tokenizer, lm_model, artifacts, nbrs

# เรียกใช้ฟังก์ชันโหลด
tokenizer, lm_model, artifacts, nbrs = load_resources()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. DEFINE MODEL CLASS (⚠️ สำคัญ: ต้องเหมือนที่เทรนเป๊ะๆ)
# ==========================================
# 🔴 ก๊อปปี้ Class GCN(...) ของคุณมาวางตรงนี้ครับ 🔴
# ตัวอย่าง (ถ้าของคุณเป็นแบบนี้):
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch) # ถ้ามี pooling
        x = self.lin(x)
        return x

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================
# กำหนดค่า Parameter ให้ตรงกับตอนเทรน (เช่น 768, 64, 2)
model = GCN(in_channels=768, hidden_channels=64, out_channels=2) 
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = lm_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def predict(content):
    # 1. Embed
    emb = get_bert_embedding(content)
    emb_norm = normalize(emb.reshape(1, -1), axis=1, norm='l2')[0]
    
    # 2. KNN Search
    dists, idxs = nbrs.kneighbors(emb_norm.reshape(1, -1))
    idxs = idxs[0]
    
    # 3. Category Logic
    neighbor_cats = [artifacts['id2cat'][artifacts['y_cat_np'][i]] for i in idxs]
    most_common_cat = Counter(neighbor_cats).most_common(1)[0][0]
    
    # 4. Build Graph
    topn = artifacts['k']
    X_new = np.vstack([emb_norm, artifacts['x_np'][idxs]])
    center = 0
    neighbors = np.arange(1, topn + 1)
    edge_index = np.concatenate([
        np.stack([np.full(topn, center), neighbors]),
        np.stack([neighbors, np.full(topn, center)])
    ], axis=1)
    edge_attr = np.concatenate([1 - dists[0], 1 - dists[0]])
    
    data_new = Data(
        x=torch.tensor(X_new, dtype=torch.float, device=device),
        edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float, device=device),
        batch=torch.zeros(X_new.shape[0], dtype=torch.long, device=device) # เผื่อต้องใช้
    )
    
    # 5. Predict
    with torch.no_grad():
        logits = model(data_new)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))
    
    return {
        'label': artifacts['id2label'][pred_id],
        'prob': probs[pred_id],
        'category': most_common_cat,
        'neighbors': neighbor_cats
    }

# ==========================================
# 5. UI (ส่วนหน้าเว็บ)
# ==========================================
st.title("🕵️ Fake News Detection System")
st.write("ระบบตรวจสอบข่าวปลอมด้วย GNN + BERT")

news_text = st.text_area("วางเนื้อหาข่าวที่ต้องการตรวจสอบ:", height=150)

if st.button("ตรวจสอบความถูกต้อง"):
    if not news_text:
        st.warning("กรุณาใส่เนื้อหาข่าวก่อนครับ")
    else:
        with st.spinner('AI กำลังวิเคราะห์...'):
            try:
                result = predict(news_text)
                
                # แสดงผล
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['label'] == 'ข่าวจริง': # ปรับ String ให้ตรงกับ id2label ของคุณ
                        st.success(f"## ✅ {result['label']}")
                    else:
                        st.error(f"## 🚨 {result['label']}")
                    st.metric("ความมั่นใจ (Confidence)", f"{result['prob']*100:.2f}%")
                
                with col2:
                    st.info(f"**หมวดหมู่:** {result['category']}")
                    st.write("**วิเคราะห์จากข่าวใกล้เคียง:**")
                    st.write(result['neighbors'])
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")