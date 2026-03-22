# api.py — แก้ทุก error
import os
import sys
import pickle
import logging
from collections import Counter
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
from model_def import GCNNet


from embed_utils import embed_text
from validators import InputValidator
from text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

app    = FastAPI(title="Fake News Detection API (WangchanBERTa)")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# ✅ นิยาม NewsRequest ครั้งเดียว — ลบตัวซ้ำออก
# ============================================================
class NewsRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ข้อความต้องไม่ว่างเปล่า")
        return v.strip()


# ============================================================
# ✅ นิยาม resources และ artifacts ก่อน try block
# ============================================================
resources: Dict[str, Any] = {}
artifacts: Dict[str, Any] = {}

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

print("⏳ Loading Models & Artifacts...")
try:
    if not os.path.exists('artifacts.pkl'):
        raise FileNotFoundError("ไม่พบ artifacts.pkl")

    sys.modules['__main__'].GCNNet = GCNNet   

    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    resources['artifacts'] = artifacts

    # kNN
    k = int(artifacts.get('k', 10))
    k = min(k, len(artifacts['x_np']))
    resources['k'] = k
    resources['nbrs_engine'] = NearestNeighbors(
        n_neighbors=k, metric='cosine'
    ).fit(artifacts['x_np'])

    # BERT
    resources['tokenizer']  = AutoTokenizer.from_pretrained(MODEL_NAME)
    resources['bert_model'] = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


    # ✅ GCN — ใช้ parameter ที่ถูกต้องจาก model_def.py
    if not os.path.exists('best_model.pth'):
        raise FileNotFoundError("ไม่พบ best_model.pth")

        model = GCNNet(in_channels=int(artifacts['x_np'].shape[1]),
                    hidden_channels=256, out_channels=2, dropout_rate=0.4).to(device)
        sd = torch.load('best_model.pth', map_location=device, weights_only=False)

        gcn_keys    = set(model.state_dict().keys())
        sd_filtered = {k: v for k, v in sd.items() if k in gcn_keys}

        model.load_state_dict(sd_filtered, strict=True)
        print(f"โหลด {len(sd_filtered)} keys จากทั้งหมด {len(sd)} keys")
        model.eval()
    if isinstance(_raw, dict):
        model.load_state_dict(_raw)
    else:
        model.load_state_dict(_raw.state_dict())
    model.eval()
    resources['model_gnn'] = model
    print("✅ Models Loaded Successfully!")

except Exception as e:
    print(f"🔥 Error loading resources: {e}")


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Fake News Detection API is Running"}


@app.post("/predict")
def predict(req: NewsRequest) -> Dict[str, Any]:
    # ✅ Validate
    validation = InputValidator.validate_text(req.text, require_thai=True)
    if not validation.is_valid:
        raise HTTPException(status_code=422, detail=validation.error_message)

    # ✅ Preprocess — ใช้ method ที่ถูกต้อง
    cleaned, is_valid, reason = TextPreprocessor.preprocess(req.text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=reason)

    content = cleaned

    try:
        tokenizer:  Any          = resources['tokenizer']
        bert_model: Any          = resources['bert_model']
        nbrs:       Any          = resources['nbrs_engine']
        model_gnn:  GCNNet = resources['model_gnn']
        arts:       Dict[str, Any] = resources['artifacts']

        x_np:     np.ndarray    = arts['x_np']
        id2label: Dict[int,str] = arts['id2label']
        id2cat:   Optional[Dict] = arts.get('id2cat')
        y_cat_np: Optional[np.ndarray] = arts.get('y_cat_np')
        topn = resources.get('k', 10)

        emb = embed_text(content, tokenizer, bert_model, device)
        dists, idxs_2d = nbrs.kneighbors(
            emb.reshape(1, -1), n_neighbors=topn
        )
        idxs: np.ndarray = idxs_2d[0]

        pred_category = "ไม่ระบุ"
        neighbor_cats: list = []
        if y_cat_np is not None and id2cat is not None:
            neighbor_cat_ids = y_cat_np[idxs]
            neighbor_cats = [id2cat[int(cid)] for cid in neighbor_cat_ids]
            most_common = Counter(neighbor_cats).most_common(1)
            if most_common:
                pred_category = most_common[0][0]

        # Build graph + self-loop
        X_new        = np.vstack([emb, x_np[idxs]])
        center_node  = 0
        nbr_nodes    = np.arange(1, topn + 1)
        self_loop_np = np.array([[center_node], [center_node]])

        edge_index_np = np.concatenate([
            np.stack([np.full(topn, center_node), nbr_nodes]),
            np.stack([nbr_nodes, np.full(topn, center_node)]),
            self_loop_np
        ], axis=1)

        w = np.clip(1 - dists[0], 0.0, 1.0)
        edge_w = np.concatenate([w, w, np.array([1.0])])

        graph_data = Data(
            x=torch.tensor(X_new, dtype=torch.float, device=device),
            edge_index=torch.tensor(
                edge_index_np, dtype=torch.long, device=device
            ),
            edge_attr=torch.tensor(edge_w, dtype=torch.float, device=device),
        )

        with torch.no_grad():
            logits: torch.Tensor = model_gnn(graph_data)
            probas = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_id   = int(np.argmax(probas))
            label_out = id2label[pred_id]

        return {
            "status":        "success",
            "label":         label_out,
            "probability":   float(probas[pred_id]),
            "category":      pred_category,
            "neighbor_cats": neighbor_cats,
            "pred_id":       pred_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))