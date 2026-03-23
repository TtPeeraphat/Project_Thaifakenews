import sys
import os
import pickle
import logging
from collections import Counter
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
from model_def import GCNNet


from graph_utils import build_star_graph

from validators import InputValidator
from text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

app    = FastAPI(title="Fake News Detection API (WangchanBERTa)")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME     = "airesearch/wangchanberta-base-att-spm-uncased"
MODEL_PATH     = "best_model.pth"
ARTIFACTS_PATH = "artifacts.pkl"


class NewsRequest(BaseModel):
    text: str
    title: str = ""      # optional — ส่งมาเมื่อดึงจาก URL scraping
    content: str = ""    # optional — เนื้อหาเต็มจาก scraper

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ข้อความต้องไม่ว่างเปล่า")
        return v.strip()


resources: Dict[str, Any] = {}
artifacts: Dict[str, Any] = {}

print("⏳ Loading Models & Artifacts...")
try:
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(f"ไม่พบ {ARTIFACTS_PATH}")

    sys.modules['__main__'].GCNNet = GCNNet

    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)

    resources['artifacts'] = artifacts

    k_neighbors = int(artifacts.get('k', 10))
    k_neighbors = min(k_neighbors, len(artifacts['x_np']))
    resources['k'] = k_neighbors
    resources['nbrs_engine'] = NearestNeighbors(
        n_neighbors=k_neighbors, metric='cosine'
    ).fit(artifacts['x_np'])

    resources['tokenizer']  = AutoTokenizer.from_pretrained(MODEL_NAME)
    resources['bert_model'] = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ไม่พบ {MODEL_PATH}")

    model = GCNNet(
        in_channels     = int(artifacts['x_np'].shape[1]),
        hidden_channels = 256,
        out_channels    = 2,
        dropout_rate    = 0.4
    ).to(device)

    sd          = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    gcn_keys    = set(model.state_dict().keys())
    sd_filtered = {key: val for key, val in sd.items() if key in gcn_keys}

    model.load_state_dict(sd_filtered, strict=True)
    model.eval()
    resources['model_gnn'] = model
    print(f"✅ โหลด {len(sd_filtered)} keys จากทั้งหมด {len(sd)} keys")

except Exception as e:
    print(f"🔥 Error loading resources: {e}")


@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Fake News Detection API is Running"}


@app.post("/predict")
def predict(req: NewsRequest) -> Dict[str, Any]:
    validation = InputValidator.validate_text(req.text, require_thai=True)
    if not validation.is_valid:
        raise HTTPException(status_code=422, detail=validation.error_message)

    cleaned, is_valid, reason = TextPreprocessor.preprocess(req.text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=reason)

    try:
        tokenizer  = resources['tokenizer']
        bert_model = resources['bert_model']
        nbrs       = resources['nbrs_engine']
        model_gnn  = resources['model_gnn']
        arts       = resources['artifacts']

        x_np     = arts['x_np']
        id2label = arts['id2label']
        id2cat   = arts.get('id2cat')
        y_cat_np = arts.get('y_cat_np')
        topn     = resources.get('k', 10)


        if req.title.strip() and req.content.strip():
            emb = embed_combined(
                title=req.title,
                content=req.content,
                tokenizer=tokenizer,
                bert_model=bert_model,
                device=device,
            )
        else:
            emb = embed_text(cleaned, tokenizer, bert_model, device)

        dists, idxs_2d = nbrs.kneighbors(emb.reshape(1, -1), n_neighbors=topn)
        idxs = idxs_2d[0]

        # Category prediction
        pred_category = "ไม่ระบุ"
        neighbor_cats = []
        if y_cat_np is not None and id2cat is not None:
            neighbor_cats = [id2cat[int(cid)] for cid in y_cat_np[idxs]]
            most_common   = Counter(neighbor_cats).most_common(1)
            if most_common:
                pred_category = most_common[0][0]


        graph_data = build_star_graph(
            query_emb      = emb,
            neighbor_embs  = x_np[idxs],
            neighbor_dists = dists[0],
            device         = device,
        )

        with torch.no_grad():
            logits  = model_gnn(graph_data)
            # logits shape: (k+1, 2) — node 0 คือ query
            probas  = torch.softmax(logits[0], dim=0).cpu().numpy()
            pred_id = int(np.argmax(probas))

        return {
            "status":        "success",
            "label":         id2label[pred_id],
            "probability":   float(probas[pred_id]),
            "category":      pred_category,
            "neighbor_cats": neighbor_cats,
            "pred_id":       pred_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))