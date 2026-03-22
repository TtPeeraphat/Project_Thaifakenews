"""
train_utils.py  (UPDATED)
=========================
CHANGES:
  - ลบ _build_batch_star_graphs() ออก (เดิมไม่มี self-loop)
  - import build_batch_star_graphs จาก graph_utils แทน
  - ทุกอย่างอื่นเหมือนเดิม 100%
"""
import numpy as np
import torch
from torch_geometric.data import Batch

# ✅ [FIX CRITICAL] import shared function แทนโค้ดซ้ำ
from graph_utils import build_batch_star_graphs


def train_epoch_inductive(model, optimizer, criterion,
                           x_query, y_query,
                           x_support, support_nbrs,
                           k, device, batch_size=32):
    model.train()
    total_loss, n_correct = 0.0, 0
    n_total  = len(x_query)
    indices  = np.random.permutation(n_total)

    for start in range(0, n_total, batch_size):
        batch_idx = indices[start: start + batch_size]
        optimizer.zero_grad()

        # ✅ ใช้ shared function → structure เหมือน inference ทุกประการ
        batch_graph = build_batch_star_graphs(
            x_query[batch_idx], x_support, support_nbrs, k, device
        )
        labels = torch.tensor(y_query[batch_idx], dtype=torch.long, device=device)

        logits = model(batch_graph)                              # (batch*(k+1), 2)
        query_idx = torch.arange(len(batch_idx), device=device) * (k + 1)
        logits_q  = logits[query_idx]                           # (batch_size, 2)

        loss = criterion(logits_q, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_idx)
        n_correct  += (logits_q.argmax(dim=1) == labels).sum().item()

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def eval_epoch_inductive(model, x_query, y_query,
                          x_support, support_nbrs,
                          k, device, batch_size=64):
    model.eval()
    y_true, y_pred = [], []

    for start in range(0, len(x_query), batch_size):
        batch_idx   = np.arange(start, min(start + batch_size, len(x_query)))

        # ✅ ใช้ shared function เช่นกัน
        batch_graph = build_batch_star_graphs(
            x_query[batch_idx], x_support, support_nbrs, k, device
        )
        logits    = model(batch_graph)
        query_idx = torch.arange(len(batch_idx), device=device) * (k + 1)
        preds     = logits[query_idx].argmax(dim=1).cpu().numpy()

        y_true.extend(y_query[batch_idx].tolist())
        y_pred.extend(preds.tolist())

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return acc, y_true, y_pred