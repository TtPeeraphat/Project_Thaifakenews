import sys
import os
import pickle
import logging
from collections import Counter          # ✅ Counter จาก collections
from typing import Dict, Any, Optional   # ✅ ลบ Counter ออกจาก typing

import numpy as np
import torch
import torch.nn.functional as F          # ✅ เพิ่ม F
import streamlit as st
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors


from text_preprocessor import TextPreprocessor   # ✅ import class
from embed_utils import embed_text
from model_def import GCNNet

logger = logging.getLogger(__name__)

BERT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
MODEL_PATH      = "best_model.pth"
ARTIFACTS_PATH  = "artifacts.pkl"

CATEGORY_RULES: Dict[str, list] = {
    "นโยบายรัฐบาล-ข่าวสาร": ["รัฐบาล","ครม.","นายกฯ","กระทรวง","นโยบาย","รัฐสภา","พรรค"],
    "ผลิตภัณฑ์สุขภาพ":      ["ยา","อาหารเสริม","สุขภาพ","รักษา","โรค","หมอ","โรงพยาบาล"],
    "การเงิน-หุ้น":          ["หุ้น","ตลาด","ลงทุน","เงิน","ธนาคาร","บาท","กำไร","ขาดทุน"],
    "ภัยพิบัติ":             ["น้ำท่วม","แผ่นดินไหว","พายุ","ไฟไหม้","ภัย","อพยพ"],
    "ความสงบและความมั่นคง":  ["ตำรวจ","ทหาร","จับกุม","ความมั่นคง","อาชญากรรม","ยิง"],
    "เศรษฐกิจ":              ["เศรษฐกิจ","GDP","เงินเฟ้อ","ส่งออก","นำเข้า","การค้า"],
    "ยาเสพติด":              ["ยาเสพติด","ยาบ้า","โคเคน","จับยา","ปราบปราม"],
}


def classify_category_by_keyword(text: str) -> str:
    scores = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in CATEGORY_RULES.items()
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "ข่าวอื่นๆ"


# ============================================================
# ✅ load_model_pipeline — มีเพียงตัวเดียวเท่านั้น
# ============================================================
@st.cache_resource
def load_model_pipeline() -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    try:
        logger.info("🔄 Loading ML Pipeline...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(ARTIFACTS_PATH):
            raise FileNotFoundError(f"ไม่พบ: {ARTIFACTS_PATH}")

        with open(ARTIFACTS_PATH, 'rb') as f:
            artifacts = pickle.load(f)

        x_database:  np.ndarray          = artifacts['x_np']
        id2label:    Dict[int, str]       = artifacts['id2label']
        id2cat:      Optional[Dict]       = artifacts.get('id2cat')
        y_cat_np:    Optional[np.ndarray] = artifacts.get('y_cat_np')
        k_neighbors: int                  = int(artifacts.get('k', 10))

        nbrs = NearestNeighbors(
            n_neighbors=k_neighbors, metric='cosine'
        ).fit(x_database)

        tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(
            BERT_MODEL_NAME
        ).to(device).eval()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ไม่พบ: {MODEL_PATH}")

        model = GCNNet(
                in_channels=768,
                hidden_channels=256,
                out_channels=2,
                dropout_rate=0.4
            ).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # ✅ return อยู่ใน try — Pylance เห็น return ทุก path
        return {
            'model':       model,
            'tokenizer':   tokenizer,
            'bert_model':  bert_model,
            'nbrs':        nbrs,
            'x_database':  x_database,
            'id2label':    id2label,
            'device':      device,
            'k_neighbors': k_neighbors,
            'id2cat':      id2cat,
            'y_cat_np':    y_cat_np,
        }

    except Exception as e:
        logger.error("❌ โหลด pipeline ล้มเหลว: %s", e, exc_info=True)
        # ✅ raise ทุก path — Pylance รู้ว่าไม่มีทาง return None
        raise RuntimeError(f"Model loading failed: {str(e)}") from e


def get_pipeline() -> Dict[str, Any]:
    return load_model_pipeline()


# ============================================================================
# 4. ✅ IMPROVED: PREDICTION WITH PROPER ERROR HANDLING
# ============================================================================


def predict_news(text: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        return {'result': 'Error', 'confidence': 0.0, 'thai_label': 'Error', 'error': 'Invalid text input'}
    cleaned, is_valid, reason = TextPreprocessor.preprocess(text)
    if not is_valid:
        return {
            'result': 'Error',
            'confidence': 0.0,
            'thai_label': 'Error',
            'error': reason
        }
    text = cleaned  # ใช้ text ที่ clean แล้ว
    try:
        model       = pipeline['model']
        tokenizer   = pipeline['tokenizer']
        bert_model  = pipeline['bert_model']
        nbrs        = pipeline['nbrs']
        x_database  = pipeline['x_database']
        id2label    = pipeline['id2label']
        device      = pipeline['device']
        k_neighbors = pipeline['k_neighbors']
 
        # A. Embed ด้วย BERT (CLS token)
        logger.debug("Tokenizing text (length: %d)", len(text))
        
        emb = embed_text(text, tokenizer, bert_model, device)  # shape: (768,)
        emb = emb.reshape(1, -1)   # ปรับ shape สำหรับ kNN
 
       
 
        logger.debug("Embedding shape: %s", emb.shape)
 
        # B. kNN search
        logger.debug("Searching %d nearest neighbors...", k_neighbors)
        dists, idxs = nbrs.kneighbors(emb, n_neighbors=k_neighbors)
        idxs = idxs[0]
 
        # C. ทำนาย category (ครั้งเดียว — ลบบล็อกซ้ำออกแล้ว)
        id2cat   = pipeline.get('id2cat')
        y_cat_np = pipeline.get('y_cat_np')
        pred_category = "ข่าวอื่นๆ"
 
        logger.debug("id2cat keys: %s", list(id2cat.keys())[:5] if id2cat else None)
        logger.debug("y_cat_np sample: %s", y_cat_np[:5] if y_cat_np is not None else None)
        logger.debug("neighbor idxs: %s", idxs)
 
        if y_cat_np is not None and id2cat is not None:
            try:
                neighbor_cat_ids = y_cat_np[idxs]
                neighbor_cats    = [id2cat[cid] for cid in neighbor_cat_ids]
                most_common      = Counter(neighbor_cats).most_common(1)
                if most_common and most_common[0][0] != "ไม่ระบุ":
                    pred_category = most_common[0][0]
                else:
                    pred_category = classify_category_by_keyword(text)
            except Exception:
                pred_category = classify_category_by_keyword(text)
        else:
            pred_category = classify_category_by_keyword(text)
 
        logger.debug("final pred_category: %s", pred_category)
 
        # D. สร้าง graph
        logger.debug("Building computation graph...")

        # รวม embedding ของ query กับ neighbor
        x_features: np.ndarray = np.vstack(
            [emb, x_database[idxs]]
        )  # shape: (k+1, 768)
        x_tensor = torch.tensor(
            x_features, dtype=torch.float
        ).to(device)

        center_node = 0
        neighbor_nodes = np.arange(1, k_neighbors + 1)

        # Bidirectional edges: center↔neighbor
        forward_edges  = np.stack(
            [np.full(k_neighbors, center_node), neighbor_nodes]
        )
        backward_edges = np.stack(
            [neighbor_nodes, np.full(k_neighbors, center_node)]
        )

        # ✅ เพิ่ม self-loop สำหรับ center node
        self_loop = np.array([[center_node], [center_node]])

        edge_index_np = np.concatenate(
            [forward_edges, backward_edges, self_loop], axis=1
        )
        graph_edge_index = torch.tensor(
            edge_index_np, dtype=torch.long
        ).to(device)

        # Edge weights
        edge_weights_np = np.concatenate(
            [1 - dists[0], 1 - dists[0], np.array([1.0])]
        )
        graph_edge_attr = torch.tensor(
            edge_weights_np, dtype=torch.float
        ).to(device)

        # E. GCN predict
        logger.debug("Running GCN inference...")

        # ✅ ใช้ชื่อ graph_data แทน data เพื่อไม่ชนกับตัวแปรอื่น
        graph_data = Data(
            x=x_tensor,
            edge_index=graph_edge_index,
            edge_attr=graph_edge_attr
        ).to(device)

        with torch.no_grad():
            out: torch.Tensor = model(graph_data)
            # out shape: (k+1, num_classes) — เอาเฉพาะ node 0 (query)
            query_logits: torch.Tensor = out[0, :]
            probs = F.softmax(query_logits, dim=0)
            pred_idx    = int(torch.argmax(probs).item())
            confidence  = float(probs[pred_idx].item()) * 100
            label       = id2label[pred_idx]
            result_text = "Real" if pred_idx == 0 else "Fake"
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        logger.info("Prediction: %s (%.1f%%)", result_text, confidence)
 
        return {
            'result':     result_text,
            'confidence': round(confidence, 2),
            'thai_label': label,
            'category':   pred_category,
            'error':      None
        }
 
    except RuntimeError as e:
        logger.error("Model error: %s", e)
        return {'result': 'Error', 'confidence': 0.0, 'thai_label': 'Error', 'error': f'Model error: {str(e)[:100]}'}
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        return {'result': 'Error', 'confidence': 0.0, 'thai_label': 'Error', 'error': f'Unexpected error: {str(e)[:100]}'}



# ============================================================================
# 5. GPU CLEANUP (Optional)
# ============================================================================
def cleanup_gpu():
    """
    Clear GPU memory if using CUDA.
    Safe to call even if not using GPU.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ GPU memory cleared")

# เพิ่มฟังก์ชันนี้ก่อน for loop ใน show_category_analysis()

