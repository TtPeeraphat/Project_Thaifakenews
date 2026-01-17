# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
from collections import Counter

# --- 🔥 เปลี่ยน: ใช้ Transformers แทน SentenceTransformer ---
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 1. Config & Model Architecture
# ==========================================
app = FastAPI(title="Fake News Detection API (WangchanBERTa)")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# 2. Global Resources (Load Once at Startup)
# ==========================================
print("⏳ Loading Models & Artifacts...")

resources = {}

try:
    # 2.1 Load Artifacts
    if not os.path.exists('artifacts.pkl'):
        raise FileNotFoundError("❌ ไม่พบไฟล์ artifacts.pkl")
    
    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
        resources['artifacts'] = artifacts

    # 2.2 Build KNN Engine
    k = min(10, len(artifacts['x_np']))
    resources['nbrs_engine'] = NearestNeighbors(n_neighbors=k, metric='cosine').fit(artifacts['x_np'])

    # 2.3 Load WangchanBERTa 
    print("   ... Loading WangchanBERTa Model")
    MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
    resources['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_NAME)
    resources['bert_model'] = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # 2.4 Load GCN Model
    model = GCNNet(num_node_features=artifacts['x_np'].shape[1], num_classes=2).to(device)
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval() # Set eval mode
        resources['model_gnn'] = model
        print("✅ Models Loaded Successfully!")
    else:
        raise FileNotFoundError("❌ ไม่พบไฟล์ best_model.pth")

except Exception as e:
    print(f"🔥 Error loading resources: {e}")

# ==========================================
# 3. Define Input/Output Format
# ==========================================
class NewsRequest(BaseModel):
    text: str

# ==========================================
# 4. API Endpoints
# ==========================================
@app.get("/")
def home():
    return {"message": "Fake News Detection API (WangchanBERTa) is Running"}

@app.post("/predict")
def predict(req: NewsRequest):
    try:
        content = req.text
        topn = 10
        
        # ดึงของจาก Global Resources
        tokenizer = resources['tokenizer']
        bert_model = resources['bert_model']
        nbrs = resources['nbrs_engine']
        model_gnn = resources['model_gnn']
        artifacts = resources['artifacts']
        
        x_np = artifacts['x_np']
        id2label = artifacts['id2label']
        id2cat = artifacts['id2cat']
        y_cat_np = artifacts.get('y_cat_np')

        # --- 1) Embedding with WangchanBERTa (Updated: Match Training Logic) ---
        
        # 1.1 Tokenize (แก้ max_length เป็น 256 ให้เท่าตอนเทรน)
        inputs = tokenizer(
            [content], 
            padding=True, 
            truncation=True, 
            max_length=256,   # ⚠️ ต้องเท่ากับตอน Train (256)
            return_tensors="pt"
        ).to(device)
        
        # 1.2 Pass through Model
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        # 1.3 Mean Pooling (แบบละเอียด: ใช้ Attention Mask เหมือนตอน Train)
        last_hidden = outputs.last_hidden_state  # Shape: (1, Seq_Len, 768)
        attn = inputs['attention_mask'].unsqueeze(-1)  # Shape: (1, Seq_Len, 1)
        
        # สูตรเดียวกับ get_bert_embeddings_batch
        summed = (last_hidden * attn).sum(dim=1)       # ผลรวมเฉพาะคำจริง
        denom = attn.sum(dim=1).clamp(min=1)           # จำนวนคำจริง (ไม่นับ Padding)
        content_emb = (summed / denom).cpu().numpy()[0] # ค่าเฉลี่ยที่ถูกต้อง
        
        # 1.4 Normalize (คงไว้ เพราะตอน Train ปกติจะทำก่อนเข้า KNN)
        emb = normalize(content_emb.reshape(1, -1), axis=1, norm='l2')[0]

        # --- 2) KNN Search (เหมือนเดิม) ---
        dists, idxs = nbrs.kneighbors(emb.reshape(1, -1), n_neighbors=topn)
        idxs = idxs[0]
        
        # Find Category Neighbors
        neighbor_cats = []
        pred_category = "ไม่ระบุ"
        if y_cat_np is not None and id2cat is not None:
            neighbor_cat_ids = y_cat_np[idxs]
            neighbor_cats = [id2cat[cid] for cid in neighbor_cat_ids]
            most_common = Counter(neighbor_cats).most_common(1)
            if most_common:
                pred_category = most_common[0][0]

        # --- 3) Build Graph ---
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

        # --- 4) Predict Real/Fake ---
        with torch.no_grad():
            logits = model_gnn(data_new)
            probas = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_id = int(np.argmax(probas)) 
            label_pred = id2label[pred_id]

        # ส่งผลลัพธ์กลับเป็น JSON
        return {
            "status": "success",
            "label": label_pred,
            "probability": float(probas[pred_id]),
            "category": pred_category,
            "neighbor_cats": neighbor_cats,
            "pred_id": pred_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))