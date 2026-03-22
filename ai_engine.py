import sys
import os
import pickle
import logging
import re
from collections import Counter
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors

from text_preprocessor import TextPreprocessor
from embed_utils import embed_combined, embed_text
from model_def import GCNNet

logger = logging.getLogger(__name__)

BERT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
MODEL_PATH     = os.path.join(_model_dir, 'best_model.pth')
ARTIFACTS_PATH = os.path.join(_model_dir, 'artifacts.pkl')
print(f"_model_dir    : {_model_dir}")
print(f"ARTIFACTS_PATH: {ARTIFACTS_PATH}")
print(f"exists        : {os.path.exists(ARTIFACTS_PATH)}")
# ─────────────────────────────────────────────────────────────────────────────
# Category keyword rules (fallback เมื่อ neighbor ไม่ตัดสินได้)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_RULES: Dict[str, list] = {
    "นโยบายรัฐบาล-ข่าวสาร": ["รัฐบาล", "ครม.", "นายกฯ", "กระทรวง", "นโยบาย"],
    "ผลิตภัณฑ์สุขภาพ":      ["ยา", "อาหารเสริม", "สุขภาพ", "รักษา", "โรค"],
    "การเงิน-หุ้น":          ["หุ้น", "ตลาด", "ลงทุน", "เงิน", "ธนาคาร"],
    "ภัยพิบัติ":             ["น้ำท่วม", "แผ่นดินไหว", "พายุ", "ไฟไหม้"],
    "ความสงบและความมั่นคง":  ["ตำรวจ", "ทหาร", "จับกุม", "ความมั่นคง"],
    "เศรษฐกิจ":              ["เศรษฐกิจ", "GDP", "เงินเฟ้อ", "ส่งออก"],
    "ยาเสพติด":              ["ยาเสพติด", "ยาบ้า", "โคเคน", "จับยา"],
}


def classify_category_by_keyword(text: str) -> str:
    scores = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in CATEGORY_RULES.items()
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "ข่าวอื่นๆ"


def _build_star_graph(
    query_emb: np.ndarray,
    neighbor_embs: np.ndarray,
    neighbor_dists: np.ndarray,
    device: torch.device,
) -> Data:
    k = len(neighbor_embs)

    # Node features: [query, neighbor_1, ..., neighbor_k]
    x_np = np.vstack([query_emb.reshape(1, -1), neighbor_embs])  # (k+1, 768)
    x = torch.tensor(x_np, dtype=torch.float32).to(device)

    center = 0
    neighbors = np.arange(1, k + 1)

    # Forward edges: center → each neighbor
    fwd = np.stack([np.full(k, center), neighbors])       # (2, k)
    # Backward edges: each neighbor → center
    bwd = np.stack([neighbors, np.full(k, center)])       # (2, k)
    # Self-loop บน center node
    slf = np.array([[center], [center]])                   # (2, 1)

    edge_index_np = np.concatenate([fwd, bwd, slf], axis=1)  # (2, 2k+1)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

    # Edge weights: [FIX M3] clip ให้อยู่ใน [0, 1]
    w = np.clip(1.0 - neighbor_dists, 0.0, 1.0)           # (k,)
    weights_np = np.concatenate([w, w, np.array([1.0])])  # (2k+1,)
    edge_attr = torch.tensor(weights_np, dtype=torch.float32).to(device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


# =============================================================================
# โหลด ML Pipeline (cached ด้วย Streamlit)
# =============================================================================

@st.cache_resource
def load_model_pipeline() -> Dict[str, Any]:
    """
    โหลด model และ artifacts ทั้งหมด (ทำครั้งเดียว cache ไว้)

    [FIX M2] id2label ดึงจาก artifacts.pkl เสมอ — ไม่ hardcode
    เหตุผล: ถ้า label index เปลี่ยน (เช่น 0=fake, 1=real แทน 0=real, 1=fake)
            hardcode จะทำให้ผลพลิกกลับหมด

    [NEW] Auto-detect model type:  GCNNet
    model เก่า (GCNNet) 
    """
    logger.info("🔄 Loading ML Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── โหลด artifacts ──
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(f"ไม่พบ artifacts: {ARTIFACTS_PATH}")

    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    x_database:  np.ndarray    = artifacts["x_np"]
    id2label:    Dict[int, str] = artifacts["id2label"]    # [FIX M2]
    id2cat:      Optional[Dict] = artifacts.get("id2cat")
    y_cat_np:    Optional[np.ndarray] = artifacts.get("y_cat_np")
    k_neighbors: int            = int(artifacts.get("k", 10))

    logger.info("Database size: %d samples, k=%d", len(x_database), k_neighbors)

    # ── kNN ──
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors, metric="cosine"
    ).fit(x_database)

    # ── WangchanBERTa ──
    tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModelForCausalLMForCausalLM.from_pretrained(BERT_MODEL_NAME).to(device).eval()

    # ── GNN model — detect  GCNNet ──
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ไม่พบ model: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
        # GCNConv — model เดิม
    model = GCNNet(
            in_channels=768, hidden_channels=256, out_channels=2, dropout_rate=0.4
        ).to(device)
    logger.info(" โหลด GCNNet ")

    model.load_state_dict(state_dict)
    model.eval()
    
    return {
        "model":       model,
        "tokenizer":   tokenizer,
        "bert_model":  bert_model,
        "nbrs":        nbrs,
        "x_database":  x_database,
        "id2label":    id2label,
        "id2cat":      id2cat,
        "y_cat_np":    y_cat_np,
        "device":      device,
        "k_neighbors": k_neighbors,
    }


def get_pipeline() -> Dict[str, Any]:
    return load_model_pipeline()


# =============================================================================
# predict_news — ฟังก์ชันหลักสำหรับทำนาย
# =============================================================================
def _strip_html(text: str) -> str:
            return re.sub(r'<[^>]+>', '', str(text)).strip()

def predict_news(
    text: str,
    pipeline: Dict[str, Any],
    content: str = "",          # [FIX C2] รับ content เพิ่มถ้ามี
) -> Dict[str, Any]:
    """
    ทำนายว่าข่าวเป็น จริง หรือ ปลอม

    Args:
        text:     ข้อความข่าว (หัวข้อ หรือ หัวข้อ+เนื้อหารวม)
        pipeline: dict จาก load_model_pipeline()
        content:  เนื้อหาข่าว (optional) — ถ้ามีจะรวมกับ text ก่อน embed

    Returns:
        dict: {
            'result':     'Real' | 'Fake' | 'Error'
            'confidence': float (0-100)
            'thai_label': str
            'category':   str
            'error':      str | None
        }
    """
    # ── Validation ──
    if not text or not isinstance(text, str):
        return _error("Invalid text input")

    cleaned, is_valid, reason = TextPreprocessor.preprocess(text)
    if not is_valid:
        return _error(reason)

    try:
        model       = pipeline["model"]
        tokenizer   = pipeline["tokenizer"]
        bert_model  = pipeline["bert_model"]
        nbrs        = pipeline["nbrs"]
        x_database  = pipeline["x_database"]
        id2label    = pipeline["id2label"]
        id2cat      = pipeline["id2cat"]
        y_cat_np    = pipeline["y_cat_np"]
        device      = pipeline["device"]
        k_neighbors = pipeline["k_neighbors"]

        # ── A. Embed ──
        # [FIX C2] ถ้ามี content จาก frontend ให้รวมก่อน embed
        #          เพื่อให้ embedding ตรงกับ training (combined_text = title + content)
        if content and content.strip():
            emb = embed_combined(
                title=cleaned, content=content,
                tokenizer=tokenizer, bert_model=bert_model, device=device,
            )
        else:
            emb = embed_text(
                cleaned, tokenizer=tokenizer, bert_model=bert_model, device=device,
            )
        # emb shape: (768,)

        # ── B. kNN Search ──
        dists, idxs = nbrs.kneighbors(emb.reshape(1, -1), n_neighbors=k_neighbors)
        dists = dists[0]  # (k,)
        idxs  = idxs[0]   # (k,)

        # ── C. Category prediction (Majority Vote จาก neighbors) ──
        pred_category = _predict_category(idxs, y_cat_np, id2cat, cleaned)

        # ── D. สร้าง Star Graph + GNN Inference ──
        # [FIX C3 + C4] ใช้ _build_star_graph ที่มี self-loop และ bidirectional edges
        neighbor_embs = x_database[idxs]
        graph_data = _build_star_graph(emb, neighbor_embs, dists, device)

        with torch.no_grad():
            out: torch.Tensor = model(graph_data)       # (k+1, 2)
            query_logits = out[0, :]                    # (2,)  — node 0 คือ query
            probs = F.softmax(query_logits, dim=0)      # softmax บน class dimension
            pred_idx   = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item()) * 100

        # [FIX M2] ดึง label จาก id2label ใน artifacts (ไม่ hardcode)
        label       = id2label[pred_idx]
        result_text = "Real" if "จริง" in label else "Fake"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Prediction: %s (%.1f%%)", result_text, confidence)
    
        return {
            "result":     result_text,
            "confidence": round(confidence, 2),
            "thai_label": label,
            'category':   _strip_html(pred_category),
            "error":      None,
        }

    except RuntimeError as e:
        logger.error("Model error: %s", e)
        return _error(f"Model error: {str(e)[:100]}")
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        return _error(f"Unexpected error: {str(e)[:100]}")


# =============================================================================
# Helper functions
# =============================================================================

def _error(msg: str) -> Dict[str, Any]:
    return {
        "result": "Error", "confidence": 0.0,
        "thai_label": "Error", "category": "", "error": msg,
    }


def _predict_category(
    idxs: np.ndarray,
    y_cat_np: Optional[np.ndarray],
    id2cat: Optional[Dict],
    text: str,
) -> str:
    """Majority vote จาก neighbors, fallback เป็น keyword matching"""
    if y_cat_np is not None and id2cat is not None:
        try:
            neighbor_cat_ids = y_cat_np[idxs]
            neighbor_cats    = [id2cat[cid] for cid in neighbor_cat_ids]
            most_common      = Counter(neighbor_cats).most_common(1)
            if most_common and most_common[0][0] != "ไม่ระบุ":
                return most_common[0][0]
        except Exception:
            pass
    return classify_category_by_keyword(text)


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ GPU memory cleared")
