from __future__ import annotations

"""run_paper_cnn_pca.py（）

核心流程（CNN + PCA）：
  1) 讀取 pm25_dataset.npz 的 y_train / y_test
  2) 讀取由 train_cnn_feature.py 產生的 features_train.npy / features_test.npy
  3) 在 train features 上用 StratifiedKFold CV 選出最佳 PCA 參數
  4) 用最佳 PCA 參數 fit 最終模型，並在 test 上評估
  5) 輸出最終 artifacts（scaler、PCA、centroid）、summary.json、meta.json

規範：
  - features 必須先由 train_cnn_feature.py 產生；本檔不會、也不應另外再訓練 CNN
  - 所有 scaler/調參只用 train features；test 僅最後評估一次（避免資料洩漏）

輸入：
  - (必）--data : pm25_dataset.npz（包含 X_train, y_train, X_test, y_test；本檔只用 y）
  - (選）--out_dir : 輸出資料夾（預設 ./results）
  - (選）--feat_tag : train_cnn_feature.py 的 tag（預設 baseline）（可選 baseline_hist_eq、baseline_hist_eq_affine 等）
  - (選）--seed : 隨機種子（預設 42）

輸出（方便重現）：
  - out_dir/cnn_pca/<feat_tag>/cnn_pca_feat_scaler.joblib : 最終 feature 標準化器（train-only fit）
  - out_dir/cnn_pca/<feat_tag>/cnn_pca_pca.joblib         : 最終 PCA
  - out_dir/cnn_pca/<feat_tag>/cnn_pca_centroid.joblib    : 最終 Nearest Centroid
  - out_dir/cnn_pca/<feat_tag>/summary.json                        : train/test accuracy
  - out_dir/cnn_pca/<feat_tag>/meta.json                           : 參數、CV 最佳參數、

執行範例：
  python run_paper_cnn_pca.py --data pm25_dataset.npz
  python run_paper_cnn_pca.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq
  python run_paper_cnn_pca.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq_affine
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Logging / Seed
# =============================================================================

def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# 資料讀取（只拿 y，不碰 X；features 由 train_cnn_feature.py 負責）
# =============================================================================

@dataclass
class Dataset:
    y_train: np.ndarray  # (N_train,)
    y_test: np.ndarray   # (N_test,)


def load_pm25_labels_npz(path: str) -> Dataset:
    """讀取附件資料庫 .npz 的 y_train / y_test。"""
    data = np.load(path)
    if "y_train" not in data or "y_test" not in data:
        raise KeyError("npz 必須包含 y_train / y_test。")

    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    return Dataset(y_train=y_train, y_test=y_test)


# =============================================================================
# Features I/O
# =============================================================================

def resolve_feature_dir(out_dir: str, feat_tag: str) -> str:
    return os.path.join(out_dir, "cnn_feature", feat_tag)


def load_features(feat_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    train_path = os.path.join(feat_dir, "features_train.npy")
    test_path = os.path.join(feat_dir, "features_test.npy")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到 features_train.npy：{train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到 features_test.npy：{test_path}")
    f_train = np.load(train_path)
    f_test = np.load(test_path)
    return f_train.astype(np.float32), f_test.astype(np.float32)


def try_load_feature_meta(feat_dir: str) -> Optional[dict]:
    meta_path = os.path.join(feat_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================================
# PCA head：Nearest Centroid in PCA space
# =============================================================================

def _fit_centroids(z_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在 PCA 空間計算每個類別的 centroid。

    Returns:
        classes: (C,) 類別 id
        centroids: (C, D) 每類別中心
    """
    classes = np.unique(y_train)
    centroids = []
    for c in classes:
        centroids.append(z_train[y_train == c].mean(axis=0))
    return classes, np.stack(centroids, axis=0)


def _predict_nearest_centroid(z: np.ndarray, classes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """用歐氏距離找最近的 centroid。"""
    # z: (N, D), centroids: (C, D) -> d2: (N, C)
    d2 = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    return classes[idx]


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# =============================================================================
# CV：選 PCA 參數（train-only）
# =============================================================================

PCAChoice = Dict[str, Union[str, int, float]]


def build_pca_candidates(n_train: int, n_feat: int) -> List[PCAChoice]:
    """PCA 參數候選（依 report.txt 的小樣本精神：n_components ≤ n_train-1）。"""
    max_k = max(1, min(n_feat, n_train - 1))

    fixed_dims = [16, 32, 64, 128, 256]
    fixed_dims = [int(min(k, max_k)) for k in fixed_dims]
    fixed_dims = sorted(list(dict.fromkeys([k for k in fixed_dims if k >= 1])))

    var_ratios = [0.80, 0.90, 0.95, 0.98]

    cands: List[PCAChoice] = []
    for k in fixed_dims:
        cands.append({"type": "n_components", "value": int(k)})
    for r in var_ratios:
        cands.append({"type": "var_ratio", "value": float(r)})
    return cands


def cv_select_pca(
    f_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """用 StratifiedKFold 在 train features 上選出最佳 PCA 設定。

    每個 fold 都會各自 fit scaler + PCA（避免洩漏）。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_train, n_feat = f_train.shape
    candidates = build_pca_candidates(n_train=n_train, n_feat=n_feat)

    rows: List[Dict[str, Any]] = []

    for cand in candidates:
        fold_accs: List[float] = []
        fold_dims: List[int] = []  # 記錄實際維度（var_ratio 會自動決定）
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(f_train, y_train), start=1):
            X_tr, X_va = f_train[tr_idx], f_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)  # train-only fit
            X_va_s = scaler.transform(X_va)

            if cand["type"] == "n_components":
                n_comp = int(cand["value"])
                pca = PCA(n_components=n_comp, random_state=seed)
            else:
                r = float(cand["value"])
                pca = PCA(n_components=r, svd_solver="full", random_state=seed)

            Z_tr = pca.fit_transform(X_tr_s)  # train-only fit
            Z_va = pca.transform(X_va_s)

            classes, centroids = _fit_centroids(Z_tr, y_tr)
            y_hat = _predict_nearest_centroid(Z_va, classes, centroids)
            acc = _acc(y_va, y_hat)

            fold_accs.append(acc)
            fold_dims.append(int(Z_tr.shape[1]))

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs, ddof=0))
        rows.append(
            {
                "cand_type": cand["type"],
                "cand_value": cand["value"],
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "fold_accs": fold_accs,
                "effective_dims": fold_dims,
            }
        )

    # best：先看 mean_acc，再看 std_acc（穩定性），最後偏好較小維度
    def _key(r: Dict[str, Any]) -> Tuple[float, float, float]:
        mean_acc = float(r["mean_acc"])
        std_acc = float(r["std_acc"])
        eff_dim_med = float(np.median(r["effective_dims"]))
        return (mean_acc, -std_acc, -eff_dim_med)

    best = sorted(rows, key=_key, reverse=True)[0]
    return {
        "candidates": rows,
        "best": best,
    }


# =============================================================================
# Final fit + evaluate（test 只用一次）
# =============================================================================

def fit_final_and_eval(
    f_train: np.ndarray,
    y_train: np.ndarray,
    f_test: np.ndarray,
    y_test: np.ndarray,
    best: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(f_train)
    X_te_s = scaler.transform(f_test)

    if best["cand_type"] == "n_components":
        pca = PCA(n_components=int(best["cand_value"]), random_state=seed)
    else:
        pca = PCA(n_components=float(best["cand_value"]), svd_solver="full", random_state=seed)

    Z_tr = pca.fit_transform(X_tr_s)
    Z_te = pca.transform(X_te_s)

    classes, centroids = _fit_centroids(Z_tr, y_train)

    y_tr_hat = _predict_nearest_centroid(Z_tr, classes, centroids)
    y_te_hat = _predict_nearest_centroid(Z_te, classes, centroids)

    return {
        "artifacts": {
            "scaler": scaler,
            "pca": pca,
            "classes": classes,
            "centroids": centroids,
        },
        "final": {
            "train_acc": _acc(y_train, y_tr_hat),
            "test_acc": _acc(y_test, y_te_hat),
            "pca_dim": int(Z_tr.shape[1]),
        },
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="資料檔 .npz 路徑（只讀 y_train/y_test）")
    p.add_argument("--out_dir", type=str, default="./results", help="輸出資料夾（features 也在這底下）")
    p.add_argument("--feat_tag", type=str, default="baseline", help="train_cnn_feature.py 的 tag")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    set_global_seed(args.seed)

    # 1) labels
    ds = load_pm25_labels_npz(args.data)
    logging.info("Labels loaded: y_train=%s y_test=%s", str(ds.y_train.shape), str(ds.y_test.shape))
    logging.info(
        "Label range: train [%d, %d], test [%d, %d]",
        int(ds.y_train.min()),
        int(ds.y_train.max()),
        int(ds.y_test.min()),
        int(ds.y_test.max()),
    )

    # 2) features
    feat_dir = resolve_feature_dir(args.out_dir, args.feat_tag)
    f_train, f_test = load_features(feat_dir)
    logging.info("Features loaded: train=%s test=%s", str(f_train.shape), str(f_test.shape))

    if f_train.shape[0] != ds.y_train.shape[0]:
        raise ValueError(f"features_train N={f_train.shape[0]} 與 y_train N={ds.y_train.shape[0]} 不一致")
    if f_test.shape[0] != ds.y_test.shape[0]:
        raise ValueError(f"features_test N={f_test.shape[0]} 與 y_test N={ds.y_test.shape[0]} 不一致")

    start = time.time()

    # 3) CV 選 PCA
    cv_res = cv_select_pca(f_train, ds.y_train, seed=args.seed, n_splits=5)
    best = cv_res["best"]
    logging.info(
        "[PCA-CV] best=%s:%s | mean_acc=%.4f ± %.4f",
        best["cand_type"],
        str(best["cand_value"]),
        float(best["mean_acc"]),
        float(best["std_acc"]),
    )

    # 4) final fit + test evaluate（test 只用一次）
    final_res = fit_final_and_eval(f_train, ds.y_train, f_test, ds.y_test, best, seed=args.seed)
    elapsed = time.time() - start

    train_acc = float(final_res["final"]["train_acc"])
    test_acc = float(final_res["final"]["test_acc"])
    pca_dim = int(final_res["final"]["pca_dim"])
    logging.info("[PCA] train_acc=%.4f | test_acc=%.4f | pca_dim=%d | time=%.1fs", train_acc, test_acc, pca_dim, elapsed)

    # 5) save artifacts（輸出獨立資料夾，避免跟 features 混在一起）
    outdir = os.path.join(args.out_dir, "cnn_pca", args.feat_tag)
    os.makedirs(outdir, exist_ok=True)

    scaler_path = os.path.join(outdir, "cnn_pca_feat_scaler.joblib")
    pca_path = os.path.join(outdir, "cnn_pca_pca.joblib")
    head_path = os.path.join(outdir, "cnn_pca_centroid.joblib")

    joblib.dump(final_res["artifacts"]["scaler"], scaler_path)
    joblib.dump(final_res["artifacts"]["pca"], pca_path)
    joblib.dump(
        {
            "classes": final_res["artifacts"]["classes"],
            "centroids": final_res["artifacts"]["centroids"],
        },
        head_path,
    )

    # summary.json（簡潔：成績）
    summary = {
        "feat_tag": args.feat_tag,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "pca_dim": pca_dim,
        "best_pca": {
            "type": best["cand_type"],
            "value": best["cand_value"],
            "cv_mean_acc": float(best["mean_acc"]),
            "cv_std_acc": float(best["std_acc"]),
            "effective_dims": best["effective_dims"],
        },
    }
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # meta.json（詳細：可重現、參數、features 來源）
    feat_meta = try_load_feature_meta(feat_dir)
    meta = {
        "seed": int(args.seed),
        "data": os.path.abspath(args.data),
        "out_dir": os.path.abspath(args.out_dir),
        "feat_tag": args.feat_tag,
        "feat_dir": os.path.abspath(feat_dir),
        "features": {
            "train_shape": list(f_train.shape),
            "test_shape": list(f_test.shape),
        },
        "pca_cv": cv_res,
        "final": final_res["final"],
        "saved": {
            "scaler": os.path.abspath(scaler_path),
            "pca": os.path.abspath(pca_path),
            "centroid": os.path.abspath(head_path),
            "summary": os.path.abspath(summary_path),
        },
        "feature_stage_meta_json": feat_meta,
        "note": "PCA 參數以 train features 的 5-fold CV 選出；test 僅最後評估一次。分類器為 PCA 空間的 Nearest Centroid。",
    }
    meta_path = os.path.join(outdir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 另外輸出一份 CV 表格（方便貼投影片）
    csv_path = os.path.join(outdir, "cv_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cand_type,cand_value,mean_acc,std_acc,effective_dims\n")
        for r in cv_res["candidates"]:
            f.write(
                f"{r['cand_type']},{r['cand_value']},{r['mean_acc']:.6f},{r['std_acc']:.6f},\"{r['effective_dims']}\"\n"
            )

    logging.info("Artifacts saved to: %s", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
