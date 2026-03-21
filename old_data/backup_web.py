import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from collections import Counter

# ==========================================
# 1. Config & Device
# ==========================================
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Define Model Architecture (ต้องเหมือนตอนเทรนเป๊ะๆ)
# ==========================================
class GCNNet(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=256, dropout_rate=0.4):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# ==========================================
# 3. Define Prediction Function (ตัวที่ Error เมื่อกี้)
# ==========================================
def predict_news(content, topn, x_np, label2id, id2label, y_cat_np, id2cat, device, nbrs, model_gnn, embed_fn):  
    # 1) Embedding
    emb_batch = embed_fn([content])
    content_emb = emb_batch[0]
    emb = normalize(content_emb.reshape(1, -1), axis=1, norm='l2')[0]

    # 2) KNN Search
    dists, idxs = nbrs.kneighbors(emb.reshape(1, -1), n_neighbors=topn)
    idxs = idxs[0]
    
    # Predict Category
    pred_category = "ไม่ระบุ"
    neighbor_cats = []
    if y_cat_np is not None and id2cat is not None:
        neighbor_cat_ids = y_cat_np[idxs]
        neighbor_cats = [id2cat[cid] for cid in neighbor_cat_ids]
        most_common = Counter(neighbor_cats).most_common(1)
        if most_common:
            pred_category = most_common[0][0]

    # 3) Build Graph
    X_new = np.vstack([emb, x_np[idxs]])
    center = 0
    neighbors = np.arange(1, topn + 1)
    
    edge_index_new = np.concatenate([
        np.stack([np.full(topn, center), neighbors]),
        np.stack([neighbors, np.full(topn, center)])
    ], axis=1)
    
    edge_weight_new = np.concatenate([1 - dists[0], 1 - dists[0]])

    data_new = Data(
        x=torch.tensor(X_new, dtype=torch.float, device=device),
        edge_index=torch.tensor(edge_index_new, dtype=torch.long, device=device),
        edge_attr=torch.tensor(edge_weight_new, dtype=torch.float, device=device),
    )

    # 4) Predict Real/Fake
    model_gnn.eval()
    with torch.no_grad():
        logits = model_gnn(data_new)
        probas = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probas)) 
        label_pred = id2label[pred_id]

    return {
        'label': label_pred,           
        'probability': float(probas[pred_id]), 
        'proba_all': probas.tolist(),  
        'category': pred_category,     
        'neighbor_cats': neighbor_cats, # 🔥 ตัวสำคัญที่ต้องมี
        'pred_id': pred_id             
    }

# ==========================================
# 4. Load Resources (Cache เพื่อให้โหลดครั้งเดียว)
# ==========================================
@st.cache_resource
def load_resources():
    print("🔄 Loading resources...")
    
    # 1. Load Artifacts
    if not os.path.exists('artifacts.pkl'):
        st.error("❌ ไม่พบไฟล์ artifacts.pkl")
        return None
        
    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    # 2. Build KNN Engine (สดๆ หน้าเว็บ)
    # หมายเหตุ: n_neighbors ต้องไม่เกินจำนวนข้อมูลที่มี
    k = min(10, len(artifacts['x_np']))
    nbrs_engine = NearestNeighbors(n_neighbors=k, metric='cosine').fit(artifacts['x_np'])

    # 3. Load SentenceBERT
    bert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embed_fn = lambda x: bert_model.encode(x)

    # 4. Load GCN Model
    model = GCNNet(num_node_features=artifacts['x_np'].shape[1], num_classes=2).to(device)
    
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("✅ Model loaded.")
    else:
        st.warning("⚠️ ไม่พบไฟล์ best_model.pth (จะใช้โมเดลเปล่าๆ)")

    return artifacts, nbrs_engine, embed_fn, model

# เรียกใช้ฟังก์ชันโหลดของ
resources = load_resources()

if resources:
    artifacts, nbrs_engine, embed_fn, model = resources
    x_bal = artifacts['x_np']
    y_cat_bal = artifacts.get('y_cat_np') # ใช้ .get กัน error
    id2label = artifacts['id2label']
    id2cat = artifacts['id2cat']
else:
    st.stop() # หยุดทำงานถ้าโหลดของไม่ได้

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
                # เรียกใช้ฟังก์ชัน predict_news ที่ประกาศไว้ด้านบน
                result = predict_news(
                    content=news_text,
                    topn=10,
                    x_np=x_bal,
                    label2id=None,
                    id2label=id2label,
                    y_cat_np=y_cat_bal,
                    id2cat=id2cat,
                    device=device,
                    nbrs=nbrs_engine,
                    model_gnn=model,
                    embed_fn=embed_fn
                )
                
                # แสดงผล
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
                    st.write("**🕵️ เพื่อนบ้าน 10 อันดับแรก:**")
                    
                    neighbor_cats = result.get('neighbor_cats', [])
                    if neighbor_cats:
                        for i, cat in enumerate(neighbor_cats):
                            st.markdown(f"**{i+1}.** <span style='color:gray'>(หมวด: {cat})</span>", unsafe_allow_html=True)
                    else:
                        st.write("- ไม่พบข้อมูล")

                # Debug
                with st.expander("🔍 Debug Information"):
                    st.write(f"Predicted ID: {result.get('pred_id')}")
                    st.write("Neighbors Categories:", neighbor_cats)
                    from collections import Counter
                    if neighbor_cats:
                        st.write("Count:", Counter(neighbor_cats))

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
                st.write(e)