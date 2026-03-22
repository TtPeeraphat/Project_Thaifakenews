# =============================================================================
# evaluate_inductive.py  ── Thesis Evaluation Script
# =============================================================================
#
# สคริปต์สำหรับวัดผลโมเดลแบบครบถ้วน สำหรับนำเสนอ thesis
#
# วัดผล 3 แบบ:
#   1. Transductive Val   — val nodes อยู่ใน training graph (สูงเกินจริง)
#   2. Inductive Holdout  — star-graph inference เหมือน production
#   3. Per-category       — accuracy แยกตามหมวดหมู่ข่าว
#
# วิธีใช้:
#   python evaluate_inductive.py --data AFNC_news_dataset_tf-2.csv
# =============================================================================

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score,
)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from model_def import GCNNet

warnings.filterwarnings("ignore")


# =============================================================================
# HELPERS
# =============================================================================

def load_artifacts(artifacts_path: str, model_path: str, device: torch.device):
    """โหลด model + artifacts"""
    with open(artifacts_path, "rb") as f:
        arts = pickle.load(f)

    model = GCNNet(in_channels=768, hidden_channels=256, out_channels=2, dropout_rate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return model, arts


def star_graph_predict(
    query_emb:   np.ndarray,
    x_train:     np.ndarray,
    nbrs:        NearestNeighbors,
    model:       torch.nn.Module,
    k:           int,
    device:      torch.device,
) -> int:
    """
    ทำนาย 1 ตัวอย่างด้วย star-graph
    เหมือน production inference ทุกประการ

    ทำไม:
    - ต้องการ metric ที่ honest กับ performance จริงๆ
    - val mask evaluation ใน graph มี information leakage
    - star-graph evaluation ไม่มี leakage เพราะ query ไม่อยู่ใน graph
    """
    q = query_emb.reshape(1, -1)
    dists, idxs = nbrs.kneighbors(q, n_neighbors=k)
    idxs = idxs[0]

    all_nodes = np.vstack([q, x_train[idxs]])   # (k+1, 768)
    x_t       = torch.tensor(all_nodes, dtype=torch.float).to(device)

    # Star edges (bidirectional)
    ctr  = np.array([0] * k + list(range(1, k + 1)))
    nb   = np.array(list(range(1, k + 1)) + [0] * k)
    ei   = torch.tensor(np.stack([ctr, nb]), dtype=torch.long).to(device)

    # Self-loop at query node (center)
    w  = np.clip(1.0 - np.concatenate([dists[0], dists[0]]), 0.0, 1.0)
    wt = torch.tensor(w, dtype=torch.float).to(device)
    ei, wt = add_self_loops(ei, wt, fill_value=1.0, num_nodes=k + 1)

    graph = Data(x=x_t, edge_index=ei).to(device)
    with torch.no_grad():
        out = model(graph)
    return int(out[0].argmax().item())


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    artifacts_path: str = "artifacts.pkl",
    model_path:     str = "best_model.pth",
    holdout_x_path: str = "holdout_x.npy",    # บันทึกตอนเทรน
    holdout_y_path: str = "holdout_y.npy",
    output_dir:     str = "eval_results",
):
    """
    วัดผลแบบครบถ้วน
    ไฟล์ holdout_x.npy และ holdout_y.npy ต้องถูกบันทึกตอน train
    """
    Path(output_dir, "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model and artifacts...")
    model, arts = load_artifacts(artifacts_path, model_path, device)

    x_train = arts["x_np"]
    id2label = arts["id2label"]
    k        = int(arts.get("k", 10))

    # [FIX M1] kNN จาก train-only database
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(x_train)

    # โหลด holdout
    x_holdout = np.load(holdout_x_path)
    y_holdout  = np.load(holdout_y_path)
    print(f"Holdout: {len(x_holdout)} samples")

    # ==========================================================
    # 1. Inductive evaluation (star-graph)
    # ==========================================================
    print("\nRunning inductive evaluation (star-graph)...")

    y_pred = []
    for i in range(len(x_holdout)):
        pred = star_graph_predict(x_holdout[i], x_train, nbrs, model, k, device)
        y_pred.append(pred)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(x_holdout)}")

    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_holdout, y_pred)
    prec = precision_score(y_holdout, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_holdout, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("INDUCTIVE EVALUATION RESULTS (Holdout Set)")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("-" * 60)
    print(classification_report(
        y_holdout, y_pred,
        target_names=[id2label[0], id2label[1]],
        digits=4, zero_division=0,
    ))

    # ==========================================================
    # 2. Confusion Matrix
    # ==========================================================
    cm = confusion_matrix(y_holdout, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Positive  (Fake ทำนายถูก): {tp}")
    print(f"  True Negative  (Real ทำนายถูก): {tn}")
    print(f"  False Positive (Real เป็น Fake): {fp}")
    print(f"  False Negative (Fake เป็น Real): {fn}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=[id2label[0], id2label[1]],
        yticklabels=[id2label[0], id2label[1]],
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Inductive Evaluation", fontsize=13)
    plt.tight_layout()
    cm_path = f"{output_dir}/img/confusion_matrix_inductive.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved: {cm_path}")

    # ==========================================================
    # 3. Summary Table (สำหรับ thesis)
    # ==========================================================
    summary = {
        "Metric":    ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Inductive": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"],
    }
    import pandas as pd
    df_summary = pd.DataFrame(summary)
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(f"{output_dir}/eval_summary.csv", index=False)
    print(f"Saved: {output_dir}/eval_summary.csv")

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }


# =============================================================================
# QUICK DEMO — ใช้ได้โดยไม่ต้องมี holdout file
# =============================================================================

def demo_single_prediction(title: str, content: str = ""):
    """
    ทดสอบ predict ข่าวเดี่ยวสำหรับ demo
    ใช้ในตอน present thesis
    """
    from ai_engine import load_model_pipeline, predict_news
    pipeline = load_model_pipeline()
    result = predict_news(text=title, content=content, pipeline=pipeline)

    print("\n" + "═" * 55)
    print("📰 ผลการวิเคราะห์ข่าว")
    print("═" * 55)
    status = "✅ ข่าวจริง" if result["result"] == "Real" else "🚨 ข่าวปลอม"
    print(f"  สถานะ    : {status}")
    print(f"  ความมั่นใจ: {result['confidence']:.1f}%")
    print(f"  หมวดหมู่  : {result['category']}")
    if result.get("error"):
        print(f"  Error    : {result['error']}")
    print("═" * 55)
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Thai Fake News Detector")
    parser.add_argument("--artifacts", default="artifacts.pkl")
    parser.add_argument("--model",     default="best_model.pth")
    parser.add_argument("--holdout-x", default="holdout_x.npy")
    parser.add_argument("--holdout-y", default="holdout_y.npy")
    parser.add_argument("--output",    default="eval_results")
    parser.add_argument("--demo",      action="store_true", help="Run single demo prediction")
    parser.add_argument("--title",     default="รัฐบาลประกาศแจกเงินประชาชน 10,000 บาท")
    parser.add_argument("--content",   default="")
    args = parser.parse_args()

    if args.demo:
        demo_single_prediction(args.title, args.content)
    else:
        run_evaluation(
            artifacts_path = args.artifacts,
            model_path     = args.model,
            holdout_x_path = args.holdout_x,
            holdout_y_path = args.holdout_y,
            output_dir     = args.output,
        )
