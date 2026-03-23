import torch
import numpy as np
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# ค่าคงที่ที่ใช้ในทั้ง training และ inference — ต้องเหมือนกัน 100%
# ─────────────────────────────────────────────────────────────────────────────
MAX_LENGTH: int = 256    # จำนวน token สูงสุด (ตรงกับ WangchanBERTa ที่เทรนมา)
TITLE_CONTENT_SEP: str = " "  # separator ระหว่าง title กับ content


def _mean_pool_and_normalize(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> np.ndarray:
    """
    Mean Pooling + L2 Normalize
    ─────────────────────────────
    ทำไมใช้ Mean Pooling แทน CLS Token?
    - WangchanBERTa เทรนด้วย Masked Language Modeling (MLM)
    - CLS token ไม่ได้ถูก fine-tune ให้เป็น sentence representation
    - Mean pooling รวม semantic ของทุก token → ดีกว่าสำหรับ sentence-level task

    ทำไมต้อง L2 Normalize?
    - cosine similarity = dot product ของ L2-normalized vectors
    - การ normalize ทำให้ kNN (cosine) ทำงานบน unit sphere → ระยะห่างมีความหมาย
    """
    # last_hidden: (batch, seq_len, 768)
    # attention_mask: (batch, seq_len)
    mask = attention_mask.unsqueeze(-1).float()       # (batch, seq_len, 1)
    summed = (last_hidden * mask).sum(dim=1)          # (batch, 768)
    denom  = mask.sum(dim=1).clamp(min=1e-9)          # (batch, 1)
    pooled = (summed / denom).cpu().numpy()           # (batch, 768)

    # L2 normalize ทุก row ให้อยู่บน unit sphere
    normalized = normalize(pooled, axis=1, norm='l2') # (batch, 768)
    return normalized


def embed_text(
    text: str,
    tokenizer,
    bert_model,
    device,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    Embed ข้อความเดี่ยวด้วย WangchanBERTa (ใช้ใน inference)

    Args:
        text:       ข้อความที่ต้องการ embed (หลัง preprocess แล้ว)
        tokenizer:  WangchanBERTa tokenizer
        bert_model: WangchanBERTa model
        device:     torch device
        max_length: ความยาว token สูงสุด (ต้องตรงกับ training)

    Returns:
        numpy array shape (768,) ที่ L2-normalized แล้ว
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

    emb = _mean_pool_and_normalize(
        outputs.last_hidden_state,
        inputs["attention_mask"],
    )
    return emb[0]  # shape (768,)


def embed_combined(
    title: str,
    content: str,
    tokenizer,
    bert_model,
    device,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    [FIX C2] Embed จาก title + content รวมกัน — เหมือนกับที่ training ทำ

    ทำไมถึงสำคัญ?
    - Training: embed จาก "หัวข้อข่าว + เนื้อหา" (combined_text)
    - Inference เดิม: embed จาก text ที่ user พิมพ์อย่างเดียว (อาจสั้นกว่ามาก)
    - ผลลัพธ์: embedding space ต่างกัน → kNN หา neighbor ผิดทิศทาง

    วิธีแก้: รับ title + content แยกกัน แล้วรวมก่อน embed
    ถ้าไม่มี content ก็ embed แค่ title (graceful fallback)

    Returns:
        numpy array shape (768,) ที่ L2-normalized แล้ว
    """
    # รวม title + content แบบเดียวกับ training pipeline
    if content and content.strip():
        combined = f"{title.strip()} {content.strip()}"
    else:
        combined = title.strip()

    return embed_text(combined, tokenizer, bert_model, device, max_length)


def embed_batch(
    texts: list,
    tokenizer,
    bert_model,
    device,
    max_length: int = MAX_LENGTH,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed ข้อความหลายอัน (ใช้ใน training pipeline)

    Args:
        texts:      list of strings
        batch_size: ขนาด batch (32 เหมาะกับ GPU 8GB)

    Returns:
        numpy array shape (N, 768) ที่ L2-normalized แล้ว
    """
    bert_model.eval()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        # ป้องกัน NaN หรือ non-string
        batch = [
            "" if (isinstance(t, float) and np.isnan(t)) else str(t)
            for t in batch
        ]

        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = bert_model(**inputs)

        batch_emb = _mean_pool_and_normalize(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )
        all_embeddings.append(batch_emb)

    return np.vstack(all_embeddings)  # (N, 768)
