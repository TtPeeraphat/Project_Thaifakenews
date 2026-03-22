import numpy as np
import torch
from torch_geometric.data import Data, Batch

def _build_batch_star_graphs(query_embs, support_embs, support_nbrs, k, device):
    """สร้าง star graph ทั้ง batch แล้วรวมเป็น graph เดียว"""
    dists_all, idxs_all = support_nbrs.kneighbors(query_embs, n_neighbors=k + 1)
    graphs = []
    for i in range(len(query_embs)):
        idxs  = idxs_all[i][1:]
        dists = dists_all[i][1:]
        x_nodes    = np.vstack([query_embs[i], support_embs[idxs]])
        neighbors  = np.arange(1, k + 1)
        src        = np.concatenate([np.zeros(k, dtype=np.int64), neighbors])
        dst        = np.concatenate([neighbors, np.zeros(k, dtype=np.int64)])
        edge_weight = np.concatenate([1.0 - dists, 1.0 - dists])
        graphs.append(Data(
            x          = torch.tensor(x_nodes,              dtype=torch.float),
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long),
            edge_attr  = torch.tensor(edge_weight,          dtype=torch.float),
        ))
    return Batch.from_data_list(graphs).to(device)


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

        # ── สร้างทั้ง batch ครั้งเดียว ──
        batch_graph = _build_batch_star_graphs(
            x_query[batch_idx], x_support, support_nbrs, k, device
        )
        labels = torch.tensor(y_query[batch_idx], dtype=torch.long, device=device)

        # ── forward ครั้งเดียว ──
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
        batch_graph = _build_batch_star_graphs(
            x_query[batch_idx], x_support, support_nbrs, k, device
        )
        logits    = model(batch_graph)
        query_idx = torch.arange(len(batch_idx), device=device) * (k + 1)
        preds     = logits[query_idx].argmax(dim=1).cpu().numpy()

        y_true.extend(y_query[batch_idx].tolist())
        y_pred.extend(preds.tolist())

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return acc, y_true, y_pred