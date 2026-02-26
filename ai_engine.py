import torch
import torch.nn.functional as F
import numpy as np
import pickle
import re
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# ==========================================
# 1. CONFIG & MODEL CLASS
# ==========================================
MODEL_PATH = "best_model.pth"
ARTIFACTS_PATH = "artifacts.pkl"
BERT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=2):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # ในโหมด Inference นี้เราใช้ edge_attr (น้ำหนักเส้นเชื่อม) ด้วยเพื่อให้แม่นยำเหมือนตอนเทรน
        edge_weight = getattr(data, 'edge_attr', None)
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔄 Loading AI Engine (Device: {device})...")

try:
    # 1. โหลด Artifacts (ข้อมูลข่าวเก่าสำหรับเทียบ kNN)
    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    
    x_database = artifacts['x_np']       # Vector ข่าวทั้งหมดในฐานข้อมูล
    id2label = artifacts['id2label']     # {0: 'ข่าวจริง', 1: 'ข่าวปลอม'}
    k_neighbors = artifacts.get('k', 10) # จำนวนเพื่อนบ้าน (ปกติคือ 10)
    
    # 2. เตรียม kNN Searcher
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine').fit(x_database)
    
    # 3. โหลด BERT
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()
    
    # 4. โหลด GNN Model
    model = GCNNet(hidden_channels=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("✅ AI Engine & kNN Database Loaded!")
    
except Exception as e:
    print(f"❌ Error Loading Resources: {e}")
    model = None

# ==========================================
# 3. PREDICTION FUNCTION (kNN-based)
# ==========================================
def predict_news(text: str):
    if model is None:
        return {"result": "System Error", "confidence": 0.0}

    try:
        # A. BERT Embedding (สร้าง Vector ให้ข่าวใหม่)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Mean Pooling เพื่อให้ได้ Vector ขนาด 768
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            emb = normalize(emb, axis=1, norm='l2') # Normalize ให้เป็นหน่วยมาตรฐาน

        # B. kNN Search (หาข่าวที่คล้ายกันในฐานข้อมูล)
        dists, idxs = nbrs.kneighbors(emb, n_neighbors=k_neighbors)
        idxs = idxs[0]
        
        # C. Construct Small Graph (1 ข่าวใหม่ + k เพื่อนบ้าน)
        # Node Features: ข่าวใหม่ (index 0) และเพื่อนบ้าน (index 1..k)
        x_features = np.vstack([emb, x_database[idxs]])
        x_tensor = torch.tensor(x_features, dtype=torch.float).to(device)
        
        # Edges: สร้างเส้นเชื่อมแบบ Star Graph (ใหม่ <-> เก่า)
        center = 0
        neighbors = np.arange(1, k_neighbors + 1)
        # เชื่อมไป-กลับ
        edge_index = torch.tensor(np.concatenate([
            np.stack([np.full(k_neighbors, center), neighbors]),
            np.stack([neighbors, np.full(k_neighbors, center)])
        ], axis=1), dtype=torch.long).to(device)
        
        # Edge Weights: ยิ่งใกล้ (Dist น้อย) น้ำหนักยิ่งเยอะ
        weights = np.concatenate([1 - dists[0], 1 - dists[0]])
        edge_attr = torch.tensor(weights, dtype=torch.float).to(device)

       # D. GCN Prediction
        # ปรับการย้ายไป device ให้เป็น string เพื่อเอาใจ Pylance
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr).to(str(device))
        
        with torch.no_grad():
            out = model(data)
            
            # แก้ปัญหาดัชนี [0] โดยการระบุให้ชัดเจนว่าเป็นดัชนีแถวแรก
            # และใช้ out[0, :] เพื่อระบุพิกัด 2 มิติ (แถวแรก, ทุกคอลัมน์)
            probs = F.softmax(out[0, :], dim=0) 
            
            pred_idx = torch.argmax(probs).item()
            # มั่นใจว่าเป็น int แน่นอน
            pred_idx = int(pred_idx) 
            
            confidence = float(probs[pred_idx].item()) * 100
            label = id2label[pred_idx]
            
            # แปลง label 'ข่าวจริง'/'ข่าวปลอม' เป็น 'Real'/'Fake' เพื่อความง่ายของระบบหลังบ้าน
            result_text = "Real" if pred_idx == 0 else "Fake"

        return {
            "result": result_text,
            "confidence": round(confidence, 2),
            "thai_label": label
        }

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"result": "Error", "confidence": 0.0}