# ✅ REFACTORED: Model Caching & Loading
# Location: ai_cache.py
# This file fixes Issue 2.1 (No Model Caching) and Issue 2.2 (Thread Safety)

import sys

import torch
import numpy as np
import pickle
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import logging
from typing import Counter, Dict, Any
from text_preprocessor import TextPreprocessor
from embed_utils import embed_text

__all__ = [
    "get_pipeline",
    "predict_news", 
    "load_model_pipeline",
    "cleanup_gpu",
    "GCNNet",
]


logger = logging.getLogger(__name__) 

# Model configuration
BERT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
MODEL_PATH = "best_model.pth"
ARTIFACTS_PATH = "artifacts.pkl"
pred_category: str = "ข่าวอื่นๆ"

def classify_category_by_keyword(text: str) -> str:
    rules = {
        "นโยบายรัฐบาล-ข่าวสาร": ["รัฐบาล","ครม.","นายกฯ","กระทรวง","นโยบาย","รัฐสภา","พรรค"],
        "ผลิตภัณฑ์สุขภาพ":      ["ยา","อาหารเสริม","สุขภาพ","รักษา","โรค","หมอ","โรงพยาบาล"],
        "การเงิน-หุ้น":          ["หุ้น","ตลาด","ลงทุน","เงิน","ธนาคาร","บาท","กำไร","ขาดทุน"],
        "ภัยพิบัติ":             ["น้ำท่วม","แผ่นดินไหว","พายุ","ไฟไหม้","ภัย","อพยพ"],
        "ความสงบและความมั่นคง":  ["ตำรวจ","ทหาร","จับกุม","ความมั่นคง","อาชญากรรม","ยิง"],
        "เศรษฐกิจ":              ["เศรษฐกิจ","GDP","เงินเฟ้อ","ส่งออก","นำเข้า","การค้า"],
        "ยาเสพติด":              ["ยาเสพติด","ยาบ้า","โคเคน","จับยา","ปราบปราม"],
    }
   
    scores: dict[str, int] = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in rules.items()
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "ข่าวอื่นๆ"
    
# ============================================================================
# 1. GCN MODEL DEFINITION (same as before)
# ============================================================================
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNet(torch.nn.Module):
    """Graph Convolutional Network for news classification."""
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=2):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

        
# ============================================================================
# 2. ✅ CRITICAL FIX: Model Caching with @st.cache_resource
# ============================================================================

@st.cache_resource
def load_model_pipeline() -> Dict[str, Any]:
    try:
        logger.info("🔄 Loading ML Pipeline (First time only)...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"📍 Using device: {device}")
        
        # 1. Load artifacts (kNN database)
        logger.info("📂 Loading artifacts...")

        # ✅ แก้ปัญหา GCNNet ไม่เจอตอน pickle.load
        setattr(sys.modules['__main__'], 'GCNNet', GCNNet)

        with open(ARTIFACTS_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        
        x_database = artifacts['x_np']
        id2label = artifacts['id2label']
        id2cat      = artifacts.get('id2cat')       # ✅ เพิ่ม
        y_cat_np    = artifacts.get('y_cat_np')     # ✅ เพิ่ม
        k_neighbors = artifacts.get('k', 10)
        logger.info(f"✅ Loaded {len(x_database)} news vectors for kNN")
        
        # 2. Load kNN searcher
        logger.info("🔍 Setting up kNN searcher...")
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine').fit(x_database)
        
        # 3. Load BERT tokenizer & model
        logger.info("🤖 Loading BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        logger.info("🧠 Loading BERT model...")
        bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()
        
        # 4. Load GCN model
        logger.info("📊 Loading GCN model...")
        model = GCNNet(hidden_channels=256).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        logger.info("✅ All models loaded successfully!")
        
        # Return immutable pipeline dictionary
        return {
            'model': model,
            'tokenizer': tokenizer,
            'bert_model': bert_model,
            'nbrs': nbrs,
            'x_database': x_database,
            'id2label': id2label,
            'device': device,
            'k_neighbors': k_neighbors,
            'id2cat':      id2cat,       # ✅ เพิ่ม
            'y_cat_np':    y_cat_np,     # ✅ เพิ่ม
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {str(e)}")


# ============================================================================
# 3. GET PIPELINE (for use in frontend)
# ============================================================================
def get_pipeline() -> Dict[str, Any]:
    """
    Get cached pipeline. Safe to call multiple times.
    Streamlit handles caching automatically.
    """
    return load_model_pipeline()


# ============================================================================
# 4. ✅ IMPROVED: PREDICTION WITH PROPER ERROR HANDLING
# ============================================================================
from torch_geometric.data import Data

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
        x_features = np.vstack([emb, x_database[idxs]])
        x_tensor   = torch.tensor(x_features, dtype=torch.float).to(device)
 
        center     = 0
        neighbors  = np.arange(1, k_neighbors + 1)
        edge_index = torch.tensor(
            np.concatenate([
                np.stack([np.full(k_neighbors, center), neighbors]),
                np.stack([neighbors, np.full(k_neighbors, center)])
            ], axis=1),
            dtype=torch.long
        ).to(device)
 
        weights   = np.concatenate([1 - dists[0], 1 - dists[0]])
        edge_attr = torch.tensor(weights, dtype=torch.float).to(device)
 
        # E. GCN predict
        logger.debug("Running GCN inference...")
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr).to(device)
 
        with torch.no_grad():
            out      = model(data)
            probs    = F.softmax(out[0, :], dim=0)
            pred_idx = int(torch.argmax(probs).item())
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
def guess_category(text: str) -> str:
    rules = {
        "นโยบายรัฐบาล-ข่าวสาร": ["รัฐบาล","ครม.","นายกฯ","กระทรวง","นโยบาย","รัฐสภา","พรรค"],
        "ผลิตภัณฑ์สุขภาพ":      ["ยา","อาหารเสริม","สุขภาพ","รักษา","โรค","หมอ","โรงพยาบาล"],
        "การเงิน-หุ้น":          ["หุ้น","ตลาด","ลงทุน","เงิน","ธนาคาร","บาท","กำไร","ขาดทุน"],
        "ภัยพิบัติ":             ["น้ำท่วม","แผ่นดินไหว","พายุ","ไฟไหม้","ภัย","อพยพ"],
        "ความสงบและความมั่นคง":  ["ตำรวจ","ทหาร","จับกุม","ความมั่นคง","อาชญากรรม","ยิง"],
        "เศรษฐกิจ":              ["เศรษฐกิจ","GDP","เงินเฟ้อ","ส่งออก","นำเข้า","การค้า"],
        "ยาเสพติด":              ["ยาเสพติด","ยาบ้า","โคเคน","จับยา","ปราบปราม"],
    }
    scores: dict[str, int] = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in rules.items()
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "ข่าวอื่นๆ"
