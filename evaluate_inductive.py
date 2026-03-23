"""
evaluate_inductive.py  (UPDATED)
=================================
CHANGES:
  - ลบ star_graph_predict() ที่ใช้ add_self_loops จาก PyG ออก
    (เดิม: structure ต่างจาก training เพราะใช้ add_self_loops)
  - import build_star_graph จาก graph_utils แทน
  - ทุกอย่างอื่นเหมือนเดิม 100% — ตัวเลข metrics ที่ได้ตอนนี้
    honest กับ production performance จริงๆ
"""
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

from model_def import GCNNet

# ✅ [FIX CRITICAL] import shared function — ลบ add_self_loops ที่ทำให้ structure ต่างกัน
from graph_utils import build_star_graph

warnings.filterwarnings("ignore")


def load_artifacts(artifacts_path: str, model_path: str, device: torch.device):
    """โหลด model + artifacts"""
    with open(artifacts_path, "rb") as f:
        arts = pickle.load(f)

    model = GCNNet(in_channels=768, hidden_channels=256, out_channels=2, dropout_rate=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return model, arts





def run_evaluation(
    artifacts_path: str = "artifacts.pkl",
    model_path:     str = "best_model.pth",
    holdout_x_path: str = "holdout_x.npy",
    holdout_y_path: str = "holdout_y.npy",
    output_dir:     str = "eval_results",
):
    Path(output_dir, "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model and artifacts...")
    model, arts = load_artifacts(artifacts_path, model_path, device)

    x_train  = arts["x_np"]
    id2label = arts["id2label"]
    k        = int(arts.get("k", 10))

    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(x_train)

    x_holdout = np.load(holdout_x_path)
    y_holdout  = np.load(holdout_y_path)
    print(f"Holdout: {len(x_holdout)} samples")

    # ==========================================================
    # Inductive evaluation (star-graph)
    # ==========================================================
    print("\nRunning inductive evaluation (star-graph)...")

    y_pred = []
    for i in range(len(x_holdout)):
        q = x_holdout[i].reshape(1, -1)
        dists, idxs = nbrs.kneighbors(q, n_neighbors=k)

        # ✅ [FIX CRITICAL] ใช้ build_star_graph → structure เหมือน training/inference ทุกประการ
        graph = build_star_graph(
            query_emb      = x_holdout[i],
            neighbor_embs  = x_train[idxs[0]],
            neighbor_dists = dists[0],
            device         = device,
        )

        with torch.no_grad():
            out = model(graph)
        pred = int(out[0].argmax().item())
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

    # Confusion Matrix
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

    # Summary
    import pandas as pd
    df_summary = pd.DataFrame({
        "Metric":    ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Inductive": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"],
    })
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(f"{output_dir}/eval_summary.csv", index=False)
    print(f"Saved: {output_dir}/eval_summary.csv")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def demo_single_prediction(title: str, content: str = ""):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Thai Fake News Detector")
    parser.add_argument("--artifacts", default="artifacts.pkl")
    parser.add_argument("--model",     default="best_model.pth")
    parser.add_argument("--holdout-x", default="holdout_x.npy")
    parser.add_argument("--holdout-y", default="holdout_y.npy")
    parser.add_argument("--output",    default="eval_results")
    parser.add_argument("--demo",      action="store_true")
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