"""
graph_utils.py  ─ Shared Star Graph Builder
============================================
ไฟล์ใหม่ที่เป็น "single source of truth" สำหรับการสร้าง star graph

ทำไมต้องแยกเป็นไฟล์ใหม่?
- train_utils.py   ใช้สร้าง batch graph (training)
- ai_engine.py     ใช้สร้าง single graph (Streamlit inference)
- api.py           ใช้สร้าง single graph (FastAPI inference)
- evaluate_inductive.py ใช้สร้าง single graph (evaluation)

เดิม: โค้ดซ้ำ 3 ที่ และ self-loop ไม่ตรงกัน
แก้:  ทุกที่ import จากที่เดียว → guaranteed consistent

Structure ที่เลือก (มี self-loop บน center node):
  - Bidirectional edges: center→neighbor และ neighbor→center
  - Self-loop ที่ center (node 0) ด้วย weight=1.0
  - Edge weight = clip(1 - cosine_distance, 0, 1)

เหตุผลที่เลือก self-loop:
  GCNConv รวม feature ของ neighbor มาอัปเดต node ตัวเอง
  ถ้าไม่มี self-loop center node จะสูญเสีย original feature บางส่วน
  ในงาน node classification แบบ inductive, self-loop ช่วยรักษา
  query representation ไว้ได้ดีกว่า
"""
import numpy as np
import torch
from torch_geometric.data import Data, Batch


# =============================================================================
# SINGLE-SAMPLE  (ใช้ใน inference และ evaluation)
# =============================================================================

def build_star_graph(
    query_emb:     np.ndarray,   # shape (D,)
    neighbor_embs: np.ndarray,   # shape (k, D)
    neighbor_dists: np.ndarray,  # shape (k,)  cosine distances จาก kNN
    device:        torch.device,
) -> Data:
    """
    สร้าง star graph สำหรับ 1 ตัวอย่าง

    Graph structure:
        Node 0  = query (center)
        Node 1..k = neighbors

    Edges (2k+1 edges รวม):
        [center→n1, center→n2, ..., center→nk]   forward
        [n1→center, n2→center, ..., nk→center]   backward
        [center→center]                            self-loop (weight=1.0)

    Args:
        query_emb:     embedding ของ query  (D,)
        neighbor_embs: embeddings ของ k neighbors  (k, D)
        neighbor_dists: cosine distances ของ k neighbors  (k,)
        device:        torch device

    Returns:
        torch_geometric.data.Data พร้อมส่งเข้า GCNNet ได้ทันที
    """
    k = len(neighbor_embs)

    # ─── Node features ───────────────────────────────────────────────────────
    # รวม query + neighbors → shape (k+1, D)
    x_np = np.vstack([query_emb.reshape(1, -1), neighbor_embs])
    x    = torch.tensor(x_np, dtype=torch.float32).to(device)

    # ─── Edge index ──────────────────────────────────────────────────────────
    center    = 0
    neighbors = np.arange(1, k + 1)

    fwd = np.stack([np.full(k, center), neighbors])     # center → each neighbor
    bwd = np.stack([neighbors, np.full(k, center)])     # each neighbor → center
    slf = np.array([[center], [center]])                # self-loop on center

    edge_index_np = np.concatenate([fwd, bwd, slf], axis=1)   # (2, 2k+1)
    edge_index    = torch.tensor(edge_index_np, dtype=torch.long).to(device)

    # ─── Edge weights ─────────────────────────────────────────────────────────
    # cosine similarity = 1 - cosine_distance, clipped to [0, 1]
    w          = np.clip(1.0 - neighbor_dists, 0.0, 1.0)           # (k,)
    weights_np = np.concatenate([w, w, np.array([1.0])])           # (2k+1,)
    edge_attr  = torch.tensor(weights_np, dtype=torch.float32).to(device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


# =============================================================================
# BATCH  (ใช้ใน training และ batch evaluation)
# =============================================================================

def build_batch_star_graphs(
    query_embs:  np.ndarray,          # (B, D)
    support_embs: np.ndarray,         # (N, D)   training database
    support_nbrs,                     # fitted NearestNeighbors
    k:           int,
    device:      torch.device,
) -> Batch:
    """
    สร้าง star graph สำหรับทั้ง batch แล้วรวมเป็น PyG Batch เดียว

    *** ใช้ build_star_graph() ภายใน → guaranteed ใช้ structure เดียวกัน ***

    Note: kneighbors(k+1) แล้วตัด index แรก (ตัวเอง) ออก
    เพราะ query อาจอยู่ใน support set ด้วย (transductive setting)

    Args:
        query_embs:   embeddings ของ B queries  (B, D)
        support_embs: embeddings ของ training database  (N, D)
        support_nbrs: sklearn NearestNeighbors fitted บน support_embs
        k:            จำนวน neighbors
        device:       torch device

    Returns:
        torch_geometric.data.Batch ของ B star graphs
    """
    # kneighbors(k+1) แล้วตัดตัวแรกออก (ป้องกัน query=neighbor)
    dists_all, idxs_all = support_nbrs.kneighbors(query_embs, n_neighbors=k + 1)

    graphs = []
    for i in range(len(query_embs)):
        neighbor_idxs  = idxs_all[i][1:]    # ตัด self ออก
        neighbor_dists = dists_all[i][1:]

        graph = build_star_graph(
            query_emb      = query_embs[i],
            neighbor_embs  = support_embs[neighbor_idxs],
            neighbor_dists = neighbor_dists,
            device         = device,
        )
        graphs.append(graph)

    return Batch.from_data_list(graphs).to(device)