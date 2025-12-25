from __future__ import annotations

"""run_paper_cnn_svm.py（讀取 train_cnn_feature 產生的 features → SVM）

核心流程（CNN + SVM）：
  1) 只讀 pm25_dataset.npz 的 y_train / y_test（本檔不再訓練 CNN）
  2) 讀取 out_dir/cnn_feature/<feat_tag>/features_train.npy 與 features_test.npy
  3) 用 train features 做 5-fold Stratified CV 選 SVM 超參數（每個 fold 各自 fit StandardScaler）
  4) 用全 train features fit 最終 StandardScaler + SVM
  5) test 只評一次（不參與調參）

規範：
  - features 必須先由 train_cnn_feature.py 產生；本檔不會、也不應另外再訓練 CNN
  - 所有 scaler/調參只用 train features；test 僅最後評估一次（避免資料洩漏）

輸入：
  - (必）--data : pm25_dataset.npz（包含 X_train, y_train, X_test, y_test；本檔只用 y）
  - (選）--out_dir : 輸出資料夾（預設 ./results）
  - (選）--feat_tag : train_cnn_feature.py 的 tag（預設 baseline）（可選 baseline_hist_eq、baseline_hist_eq_affine 等）
  - (選）--seed : 隨機種子（預設 42）

輸出（方便重現）：
  - out_dir/cnn_svm/<feat_tag>/cnn_svm_feat_scaler.joblib : 最終 feature 標準化器（train-only fit）
  - out_dir/cnn_svm/<feat_tag>/cnn_svm_svm.joblib         : 最終 SVM
  - out_dir/cnn_svm/<feat_tag>/summary.json               : train/test accuracy
  - out_dir/cnn_svm/<feat_tag>/meta.json                  : 參數、CV 最佳超參數、features 來源、可重現資訊

執行範例：
  python run_paper_cnn_svm.py --data pm25_dataset.npz 
  python run_paper_cnn_svm.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq
  python run_paper_cnn_svm.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq_affine
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =============================================================================
# 基本設定：seed、log
# =============================================================================

def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def set_global_seed(seed: int = 42) -> None:
    """固定隨機種子（可重現）。"""
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
    """讀取附件資料庫 .npz 的 y_train, y_test（月份標籤通常 1~12，這裡轉成 0~11）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到資料檔：{path}\n"
            "請把 pm25_dataset.npz 放在同一資料夾，或用 --data 指定路徑。"
        )

    with np.load(path, allow_pickle=False) as data:
        required = ["y_train", "y_test"]
        for k in required:
            if k not in data.files:
                raise KeyError(f"資料檔缺少 key: {k}，需要：{required}")

        y_train = data["y_train"].astype(np.int64)
        y_test = data["y_test"].astype(np.int64)

    # 常見情況：月份標籤為 1~12，轉成 0~11
    if y_train.min() == 1 and y_train.max() == 12:
        y_train = y_train - 1
    if y_test.min() == 1 and y_test.max() == 12:
        y_test = y_test - 1

    return Dataset(y_train=y_train, y_test=y_test)


# =============================================================================
# 讀取 precomputed features
# =============================================================================

def load_features(out_dir: str, feat_tag: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    從 train_cnn_feature.py 的輸出讀 features：
      out_dir/cnn_feature/<feat_tag>/features_train.npy
      out_dir/cnn_feature/<feat_tag>/features_test.npy
    """
    feat_dir = os.path.join(out_dir, "cnn_feature", feat_tag)
    feat_tr_path = os.path.join(feat_dir, "features_train.npy")
    feat_te_path = os.path.join(feat_dir, "features_test.npy")

    if not os.path.exists(feat_tr_path) or not os.path.exists(feat_te_path):
        raise FileNotFoundError(
            "找不到 features 檔案，請先跑 train_cnn_feature.py 產生特徵。\n"
            f"預期位置：\n"
            f"  - {feat_tr_path}\n"
            f"  - {feat_te_path}\n\n"
            "例如：\n"
            f"  python train_cnn_feature.py --data pm25_dataset.npz --out_dir {out_dir} --tag {feat_tag} "
            "--preprocess hist_eq --train_affine 1\n"
        )

    f_train = np.load(feat_tr_path).astype(np.float32)
    f_test = np.load(feat_te_path).astype(np.float32)

    logging.info("[FEAT] loaded: train=%s test=%s (dir=%s)", str(f_train.shape), str(f_test.shape), feat_dir)
    return f_train, f_test, feat_dir


def try_load_feature_meta(feat_dir: str) -> Optional[Dict[str, Any]]:
    """可選：把 train_cnn_feature.py 的 meta.json 一起帶進來，方便可重現報告。"""
    meta_path = os.path.join(feat_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================================
# SVM：CV 選參數 → 全 train fit → test 只評一次
# =============================================================================

def run_svm_with_cv(
    f_train: np.ndarray,
    y_train: np.ndarray,
    f_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    回傳：
      - result dict（含 cv best、train/test acc）
      - pred_train
      - pred_test
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 你原本的搜尋空間（保留）
    cw_list = [None, "balanced"]
    C_list = [0.1, 1, 3, 10, 30, 100, 300, 1000]
    gamma_list = ["scale", 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    best_mean = -1.0
    best_std = 1e9
    best_params: Dict[str, Any] = {"C": 1.0, "gamma": "scale", "class_weight": None}

    # 只用 train 做 CV（每 fold 的 scaler 也只能用該 fold 的 train fit）
    for C in C_list:
        for gamma in gamma_list:
            for cw in cw_list:
                fold_scores = []
                for tr_idx, va_idx in cv.split(f_train, y_train):
                    sc = StandardScaler()
                    X_tr_fold = sc.fit_transform(f_train[tr_idx])   # train-only fit
                    X_va_fold = sc.transform(f_train[va_idx])

                    clf = SVC(kernel="rbf", C=float(C), gamma=gamma, class_weight=cw)
                    clf.fit(X_tr_fold, y_train[tr_idx])

                    pred_va = clf.predict(X_va_fold)
                    fold_scores.append(float(accuracy_score(y_train[va_idx], pred_va)))

                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))

                improved = (mean_score > best_mean + 1e-12) or (
                    abs(mean_score - best_mean) <= 1e-12 and std_score < best_std - 1e-12
                )
                if improved:
                    best_mean = mean_score
                    best_std = std_score
                    best_params = {"C": float(C), "gamma": gamma, "class_weight": cw}
                    logging.info(
                        "[SVM-CV] new best -> C=%s | gamma=%s | class_weight=%s | cv_mean=%.4f | cv_std=%.4f",
                        str(best_params["C"]),
                        str(best_params["gamma"]),
                        str(best_params["class_weight"]),
                        best_mean,
                        best_std,
                    )

    logging.info(
        "[SVM-CV] BEST -> C=%s | gamma=%s | class_weight=%s | cv_mean=%.4f | cv_std=%.4f",
        str(best_params["C"]),
        str(best_params["gamma"]),
        str(best_params["class_weight"]),
        best_mean,
        best_std,
    )

    # 最終：用全 train fit scaler + SVM，test 只做一次評估
    scaler_final = StandardScaler()
    f_train_s = scaler_final.fit_transform(f_train)  # train-only fit
    f_test_s = scaler_final.transform(f_test)

    clf_final = SVC(
        kernel="rbf",
        C=float(best_params["C"]),
        gamma=best_params["gamma"],
        class_weight=best_params["class_weight"],
    )
    clf_final.fit(f_train_s, y_train)

    pred_train = clf_final.predict(f_train_s)
    pred_test = clf_final.predict(f_test_s)

    acc_train = float(accuracy_score(y_train, pred_train))
    acc_test = float(accuracy_score(y_test, pred_test))

    result = {
        "svm_cv": {
            "best_params": best_params,
            "cv_mean_acc": best_mean,
            "cv_std_acc": best_std,
            "n_splits": 5,
        },
        "final": {
            "train_acc": acc_train,
            "test_acc": acc_test,
        },
        "artifacts": {
            "scaler": scaler_final,
            "svm": clf_final,
        },
    }
    return result, pred_train, pred_test


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

    # 2) precomputed features
    f_train, f_test, feat_dir = load_features(args.out_dir, args.feat_tag)

    # 3) sanity check：筆數必須對得上
    if f_train.shape[0] != ds.y_train.shape[0]:
        raise ValueError(
            f"features_train 筆數不一致：features_train={f_train.shape[0]} 但 y_train={ds.y_train.shape[0]}\n"
            f"feat_dir={feat_dir}"
        )
    if f_test.shape[0] != ds.y_test.shape[0]:
        raise ValueError(
            f"features_test 筆數不一致：features_test={f_test.shape[0]} 但 y_test={ds.y_test.shape[0]}\n"
            f"feat_dir={feat_dir}"
        )

    # 4) run SVM
    start = time.time()
    result, _, _ = run_svm_with_cv(
        f_train,
        ds.y_train,
        f_test,
        ds.y_test,
        seed=args.seed,
    )
    elapsed = time.time() - start

    train_acc = float(result["final"]["train_acc"])
    test_acc = float(result["final"]["test_acc"])
    logging.info("[SVM] train_acc=%.4f | test_acc=%.4f | time=%.1fs", train_acc, test_acc, elapsed)

    # 5) save artifacts（輸出獨立資料夾，避免跟 features 混在一起）
    outdir = os.path.join(args.out_dir, "cnn_svm", args.feat_tag)
    os.makedirs(outdir, exist_ok=True)

    scaler_path = os.path.join(outdir, "cnn_svm_feat_scaler.joblib")
    svm_path = os.path.join(outdir, "cnn_svm_svm.joblib")

    joblib.dump(result["artifacts"]["scaler"], scaler_path)
    joblib.dump(result["artifacts"]["svm"], svm_path)

    # summary.json（簡潔：成績）
    summary = {
        "feat_tag": args.feat_tag,
        "train_acc": train_acc,
        "test_acc": test_acc,
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
        "svm_cv": result["svm_cv"],
        "final": result["final"],
        "saved": {
            "scaler": os.path.abspath(scaler_path),
            "svm": os.path.abspath(svm_path),
            "summary": os.path.abspath(summary_path),
        },
        "feature_stage_meta_json": feat_meta,  # 可能為 None（若 features 資料夾沒 meta.json）
        "note": "SVM 超參數以 train features 的 5-fold CV 選出；test 僅最後評估一次。",
    }
    meta_path = os.path.join(outdir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("[DONE] saved: %s", outdir)
    logging.info("  - %s", scaler_path)
    logging.info("  - %s", svm_path)
    logging.info("  - %s", summary_path)
    logging.info("  - %s", meta_path)


if __name__ == "__main__":
    main()
