import streamlit as st
import torch
import numpy as np
import pickle
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize 
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import re

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
st.set_page_config(page_title="AI ข่าวปลอม Detector", page_icon="🕵️")

@st.cache_resource
def load_resources():
    # A) โหลด BERT
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    lm_model = AutoModel.from_pretrained('xlm-roberta-base')
    
    # B) โหลดข้อมูลประกอบ
    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # C) สร้างระบบค้นหา (KNN)
    nbrs = NearestNeighbors(n_neighbors=artifacts['k'], metric='cosine')
    nbrs.fit(artifacts['x_np'])
    
    return tokenizer, lm_model, artifacts, nbrs

# เรียกใช้ฟังก์ชันโหลด
tokenizer, lm_model, artifacts, nbrs = load_resources()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. DEFINE MODEL CLASS
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        return x

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================
model = GCN(in_channels=768, hidden_channels=256, out_channels=2) 
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def get_bert_embeddings_batch(texts, tokenizer, model, device, max_length=256, batch_size=32, use_mean_pool=True):
    model.eval()
    all_embeddings = []
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        batch_texts = ["" if (isinstance(t, float) and np.isnan(t)) else str(t) for t in batch_texts]
        
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            
            if use_mean_pool:
                attn = inputs['attention_mask'].unsqueeze(-1)
                summed = (last_hidden * attn).sum(dim=1)
                denom = attn.sum(dim=1).clamp(min=1)
                emb = (summed / denom).cpu().numpy()
            else:
                emb = last_hidden[:, 0, :].cpu().numpy()
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)
def clean_text(text):
    # ลบ URL
    text = re.sub(r'http\S+', '', text)
    # ลบตัวอักษรพิเศษ (เหลือแค่ ไทย อังกฤษ ตัวเลข)
    text = re.sub(r'[^a-zA-Z0-9\u0E00-\u0E7F\s]', '', text)
    # ลบช่องว่างซ้ำๆ
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(content):
    content = clean_text(content)
    # 1. Embed (ใช้ค่าดิบๆ ไม่ต้อง Normalize)
    emb = get_bert_embeddings_batch([content], tokenizer, lm_model, device, max_length=256, use_mean_pool=True)
    emb_vec = emb[0] 
    
    # ⚠️ แก้ไขจุดสำคัญ: ใช้ vector ดิบๆ เลย ไม่ต้อง Normalize จะได้ตรงกับ Training Data
    emb_norm = normalize(emb_vec.reshape(1, -1), axis=1, norm='l2')[0]
    
    # 2. KNN Search
    dists, idxs = nbrs.kneighbors(emb_norm.reshape(1, -1))
    idxs = idxs[0]
    
    # 3. Logic หาหมวดหมู่
    neighbor_cats = [artifacts['id2cat'][artifacts['y_cat_np'][i]] for i in idxs]
    most_common_cat = Counter(neighbor_cats).most_common(1)[0][0]
    
    # 4. สร้าง Graph
    topn = artifacts['k']
    X_new = np.vstack([emb_norm, artifacts['x_np'][idxs]]) # รวมโหนดข่าวใหม่กับเพื่อนบ้าน
    
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
        batch=torch.zeros(X_new.shape[0], dtype=torch.long, device=device)
    )
    
    # 5. เข้า Model
    with torch.no_grad():
        logits = model(data_new)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))
    
    return {
        'label': artifacts['id2label'][pred_id],
        'prob': probs[pred_id],
        'category': most_common_cat,
        'neighbors': neighbor_cats,
        'pred_id': pred_id,
        'raw_probs': probs.tolist()
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
                # เรียกใช้ฟังก์ชันแค่ครั้งเดียวพอ
                result = predict(news_text)
                
                # แสดงผล
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['label'] == 'ข่าวจริง': 
                        st.success(f"## ✅ {result['label']}")
                    else:
                        st.error(f"## 🚨 {result['label']}")
                    st.metric("ความมั่นใจ (Confidence)", f"{result['prob']*100:.2f}%")
                
                with col2:
                    st.info(f"**หมวดหมู่:** {result['category']}")
                    st.write("**วิเคราะห์จากข่าวใกล้เคียง:**")
                    st.write(result['neighbors'])
                    
                # ส่วน Debug
                with st.expander("🛠️ Debug Information"):
                    st.write(f"**Predicted ID:** {result['pred_id']}")
                    st.write(f"**Label Mapping:** {artifacts['id2label']}")
                    st.write(f"**Raw Probabilities:** {result['raw_probs']}")
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")