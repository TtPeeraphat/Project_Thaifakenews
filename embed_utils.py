import torch
import numpy as np
from sklearn.preprocessing import normalize
 
 
def embed_text(text: str, tokenizer, bert_model, device, max_length: int = 256) -> np.ndarray:
    """
    แปลงข้อความเป็น embedding vector ด้วย Mean Pooling
    
    ใช้ฟังก์ชันนี้เพียงที่เดียว แล้ว import ทั้งใน ai_engine.py และ api.py
    เพื่อให้ embedding สอดคล้องกัน 100%
    
    Args:
        text:        ข้อความที่ต้องการแปลง
        tokenizer:   WangchanBERTa tokenizer
        bert_model:  WangchanBERTa model
        device:      torch device (cpu / cuda)
        max_length:  ความยาวสูงสุดของ token (ต้องตรงกับตอน train)
    
    Returns:
        numpy array shape (768,) ที่ normalize แล้ว
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
 
    with torch.inference_mode():
        outputs = bert_model(**inputs)
 
    # ── Mean Pooling: คำนวณค่าเฉลี่ยเฉพาะ token จริง (ไม่นับ padding) ──
    last_hidden = outputs.last_hidden_state          # (1, seq_len, 768)
    attn_mask   = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
 
    summed = (last_hidden * attn_mask).sum(dim=1)    # (1, 768)
    denom  = attn_mask.sum(dim=1).clamp(min=1)       # (1, 1)
    mean_pooled = (summed / denom).cpu().numpy()[0]  # (768,)
 
    # ── L2 Normalize ──
    emb = normalize(mean_pooled.reshape(1, -1), axis=1, norm="l2")[0]
    return emb  # shape: (768,)