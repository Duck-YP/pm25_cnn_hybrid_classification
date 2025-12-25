from __future__ import annotations

"""run_baseline_pca_knn.py（原始 X flatten → StandardScaler → PCA → k-NN）

核心流程（PCA + k-NN baseline）：
  1) 讀 pm25_dataset.npz 的 X_train / y_train / X_test / y_test
  2) 將 X 直接 flatten 成 (N, D)（不經 CNN）
  3) 只用 train 做 5-fold Stratified CV，挑：
       - PCA 維度 n_components
       - k-NN 的 k
     ※ 每個 fold 都是「fold-train fit scaler & PCA」，fold-val 只 transform（避免洩漏）
  4) 用全 train fit 最終 StandardScaler + PCA + KNN
  5) test 只評一次（不參與任何調參）

規範：
  - StandardScaler / PCA 都只能在 train（或每個 fold 的 train split）fit
  - test 只能在最後一次評估，不能用來挑 n_components / k

輸入：
  - (必）--data : pm25_dataset.npz（包含 X_train, y_train, X_test, y_test）
  - (選）--out_dir : 輸出資料夾（預設 ./results）
  - (選）--seed : 隨機種子（預設 42）

輸出（方便重現）：
  - out_dir/baseline_pca_knn/pca_knn_feat_scaler.joblib : 最終標準化器（train-only fit）
  - out_dir/baseline_pca_knn/pca_knn_pca.joblib         : 最終 PCA（train-only fit）
  - out_dir/baseline_pca_knn/pca_knn_knn.joblib         : 最終 KNN
  - out_dir/baseline_pca_knn/summary.json               : train/test accuracy
  - out_dir/baseline_pca_knn/meta.json                  : 參數、CV 最佳超參數、資料形狀、可重現資訊

執行範例：
  python run_baseline_pca_knn.py --data pm25_dataset.npz
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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
# 資料讀取：X + y（baseline 直接用原始輸入，不用 CNN features）
# =============================================================================

@dataclass
class Dataset:
    X_train: np.ndarray  # (N_train, H, W) 或 (N_train, ...)
    y_train: np.ndarray  # (N_train,)
    X_test: np.ndarray   # (N_test, H, W) 或 (N_test, ...)
    y_test: np.ndarray   # (N_test,)


def load_pm25_npz(path: str) -> Dataset:
    """讀取附件資料庫 .npz 的 X_train, y_train, X_test, y_test（月份標籤常見 1~12，這裡轉 0~11）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到資料檔：{path}\n"
            "請把 pm25_dataset.npz 放在同一資料夾，或用 --data 指定路徑。"
        )

    with np.load(path, allow_pickle=False) as data:
        required = ["X_train", "y_train", "X_test", "y_test"]
        for k in required:
            if k not in data.files:
                raise KeyError(f"資料檔缺少 key: {k}，需要：{required}")

        X_train = data["X_train"]
        y_train = data["y_train"].astype(np.int64)
        X_test = data["X_test"]
        y_test = data["y_test"].astype(np.int64)

    # 常見情況：月份標籤為 1~12，轉成 0~11（月份-1）
    if y_train.min() == 1 and y_train.max() == 12:
        y_train = y_train - 1
    if y_test.min() == 1 and y_test.max() == 12:
        y_test = y_test - 1

    return Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def flatten_X(X: np.ndarray) -> np.ndarray:
    """把任意形狀的 X flatten 成 (N, D)。"""
    if X.ndim < 2:
        raise ValueError(f"X 維度不合理：X.ndim={X.ndim}，預期至少 2 維（N, ...）")
    N = X.shape[0]
    return X.reshape(N, -1).astype(np.float32)


# =============================================================================
# PCA + k-NN：CV 選參數 → 全 train fit → test 只評一次
# =============================================================================

def build_search_space(*, n_train: int, d_in: int) -> Tuple[list[int], list[int]]:
    """
    依資料集報告（report.txt）做「安全」的搜尋空間：
      - PCA n_components 不能超過 N_train-1（PCA 的可用 rank 上限）
      - 原始 flatten D 很大（157*103=16171），所以 PCA 幾乎一定 <= 364
    """
    pca_max = min(d_in, n_train - 1)  # 365 -> 364
    # 這些維度是常見「由小到大」的候選；會自動裁切到 <= pca_max
    cand_dims = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 364]
    n_components_list = [x for x in cand_dims if x <= pca_max]
    if len(n_components_list) == 0:
        # 理論上不太會發生，除非 n_train 很小
        n_components_list = [min(8, pca_max)]

    # k-NN：小資料集（365）下，k 不宜太大；且通常用奇數避免平手
    k_list = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    # k 不能大於 fold-train 的樣本數；我們後面 fold 內會再保險裁切
    return n_components_list, k_list


def run_pca_knn_with_cv(
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

    n_train, d_in = f_train.shape
    n_components_list, k_list = build_search_space(n_train=n_train, d_in=d_in)

    logging.info("[SEARCH] PCA n_components candidates: %s", n_components_list)
    logging.info("[SEARCH] KNN k candidates: %s", k_list)

    best_mean = -1.0
    best_std = 1e9
    best_params: Dict[str, Any] = {"n_components": n_components_list[0], "k": k_list[0]}

    # 只用 train 做 CV（每 fold scaler/PCA 都只能用該 fold train fit）
    for n_comp in n_components_list:
        for k in k_list:
            fold_scores = []
            for tr_idx, va_idx in cv.split(f_train, y_train):
                # --- fold 資料 ---
                X_tr = f_train[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = f_train[va_idx]
                y_va = y_train[va_idx]

                # --- 防呆：k 不能超過 fold-train 的樣本數 ---
                k_eff = int(min(k, max(1, len(tr_idx))))
                if k_eff != k:
                    # 不用一直刷屏，只在遇到時提示一次即可（這裡保守：仍記錄）
                    logging.debug("[KNN] k=%d too large for fold train=%d -> use k_eff=%d", k, len(tr_idx), k_eff)

                # 1) StandardScaler：fold-train fit
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)   # train-only fit
                X_va_s = sc.transform(X_va)

                # 2) PCA：fold-train fit
                # PCA 維度也不能超過 fold-train 的 rank 上限
                pca_max_fold = min(X_tr_s.shape[1], X_tr_s.shape[0] - 1)
                n_comp_eff = int(min(n_comp, max(1, pca_max_fold)))

                pca = PCA(n_components=n_comp_eff, random_state=seed, svd_solver="randomized")
                X_tr_p = pca.fit_transform(X_tr_s)  # train-only fit
                X_va_p = pca.transform(X_va_s)

                # 3) k-NN：在 PCA 空間分類
                # 直覺：用 Euclidean + uniform 權重（最標準 baseline）
                knn = KNeighborsClassifier(
                    n_neighbors=k_eff,
                    weights="uniform",
                    metric="minkowski",
                    p=2,  # p=2 -> Euclidean distance
                )
                knn.fit(X_tr_p, y_tr)

                pred_va = knn.predict(X_va_p)
                fold_scores.append(float(accuracy_score(y_va, pred_va)))

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            improved = (mean_score > best_mean + 1e-12) or (
                abs(mean_score - best_mean) <= 1e-12 and std_score < best_std - 1e-12
            )
            if improved:
                best_mean = mean_score
                best_std = std_score
                best_params = {"n_components": int(n_comp), "k": int(k)}
                logging.info(
                    "[PCA+KNN-CV] new best -> n_components=%d | k=%d | cv_mean=%.4f | cv_std=%.4f",
                    best_params["n_components"],
                    best_params["k"],
                    best_mean,
                    best_std,
                )

    logging.info(
        "[PCA+KNN-CV] BEST -> n_components=%d | k=%d | cv_mean=%.4f | cv_std=%.4f",
        best_params["n_components"],
        best_params["k"],
        best_mean,
        best_std,
    )

    # 最終：用全 train fit scaler + PCA + KNN，test 只做一次評估
    scaler_final = StandardScaler()
    f_train_s = scaler_final.fit_transform(f_train)  # train-only fit
    f_test_s = scaler_final.transform(f_test)

    # PCA 維度上限：min(D, N_train-1)=364（依 report.txt 的 N_train=365）
    pca_max_final = min(f_train_s.shape[1], f_train_s.shape[0] - 1)
    n_comp_final = int(min(best_params["n_components"], max(1, pca_max_final)))

    pca_final = PCA(n_components=n_comp_final, random_state=seed)
    f_train_p = pca_final.fit_transform(f_train_s)  # train-only fit
    f_test_p = pca_final.transform(f_test_s)

    # k 也不能超過 N_train（理論上不會，但保險）
    k_final = int(min(best_params["k"], max(1, f_train_p.shape[0])))

    knn_final = KNeighborsClassifier(
        n_neighbors=k_final,
        weights="uniform",
        metric="minkowski",
        p=2,
    )
    knn_final.fit(f_train_p, y_train)

    pred_train = knn_final.predict(f_train_p)
    pred_test = knn_final.predict(f_test_p)

    acc_train = float(accuracy_score(y_train, pred_train))
    acc_test = float(accuracy_score(y_test, pred_test))

    result = {
        "pca_knn_cv": {
            "best_params": {
                "n_components": int(best_params["n_components"]),
                "k": int(best_params["k"]),
            },
            "cv_mean_acc": best_mean,
            "cv_std_acc": best_std,
            "n_splits": 5,
            "search_space": {
                "n_components_list": n_components_list,
                "k_list": k_list,
                "pca_max_by_rule": int(min(d_in, n_train - 1)),
            },
        },
        "final": {
            "train_acc": acc_train,
            "test_acc": acc_test,
            "final_n_components_used": int(n_comp_final),
            "final_k_used": int(k_final),
        },
        "artifacts": {
            "scaler": scaler_final,
            "pca": pca_final,
            "knn": knn_final,
        },
    }
    return result, pred_train, pred_test


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="資料檔 .npz 路徑（讀 X_train/y_train/X_test/y_test）")
    p.add_argument("--out_dir", type=str, default="./results", help="輸出資料夾（預設 ./results）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    set_global_seed(args.seed)

    # 1) load dataset
    ds = load_pm25_npz(args.data)
    logging.info("Loaded: X_train=%s y_train=%s | X_test=%s y_test=%s",
                 str(ds.X_train.shape), str(ds.y_train.shape), str(ds.X_test.shape), str(ds.y_test.shape))
    logging.info(
        "Label range: train [%d, %d], test [%d, %d]",
        int(ds.y_train.min()),
        int(ds.y_train.max()),
        int(ds.y_test.min()),
        int(ds.y_test.max()),
    )

    # 2) flatten raw X
    f_train = flatten_X(ds.X_train)
    f_test = flatten_X(ds.X_test)
    logging.info("[FLAT] X -> f_train=%s f_test=%s (dtype=%s)",
                 str(f_train.shape), str(f_test.shape), str(f_train.dtype))

    # 3) sanity check：筆數必須對得上
    if f_train.shape[0] != ds.y_train.shape[0]:
        raise ValueError(f"flatten 後筆數不一致：f_train={f_train.shape[0]} 但 y_train={ds.y_train.shape[0]}")
    if f_test.shape[0] != ds.y_test.shape[0]:
        raise ValueError(f"flatten 後筆數不一致：f_test={f_test.shape[0]} 但 y_test={ds.y_test.shape[0]}")

    # 4) run PCA + KNN
    start = time.time()
    result, _, _ = run_pca_knn_with_cv(
        f_train,
        ds.y_train,
        f_test,
        ds.y_test,
        seed=args.seed,
    )
    elapsed = time.time() - start

    train_acc = float(result["final"]["train_acc"])
    test_acc = float(result["final"]["test_acc"])
    logging.info(
        "[PCA+KNN] train_acc=%.4f | test_acc=%.4f | n_comp=%d | k=%d | time=%.1fs",
        train_acc,
        test_acc,
        int(result["final"]["final_n_components_used"]),
        int(result["final"]["final_k_used"]),
        elapsed,
    )

    # 5) save artifacts
    outdir = os.path.join(args.out_dir, "baseline_pca_knn")
    os.makedirs(outdir, exist_ok=True)

    scaler_path = os.path.join(outdir, "pca_knn_feat_scaler.joblib")
    pca_path = os.path.join(outdir, "pca_knn_pca.joblib")
    knn_path = os.path.join(outdir, "pca_knn_knn.joblib")

    joblib.dump(result["artifacts"]["scaler"], scaler_path)
    joblib.dump(result["artifacts"]["pca"], pca_path)
    joblib.dump(result["artifacts"]["knn"], knn_path)

    # summary.json（簡潔：成績）
    summary = {
        "method": "baseline_pca_knn",
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_n_components": int(result["final"]["final_n_components_used"]),
        "final_k": int(result["final"]["final_k_used"]),
    }
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # meta.json（詳細：可重現、參數、資料形狀）
    meta = {
        "seed": int(args.seed),
        "data": os.path.abspath(args.data),
        "out_dir": os.path.abspath(args.out_dir),
        "raw_input": {
            "X_train_shape": list(ds.X_train.shape),
            "X_test_shape": list(ds.X_test.shape),
            "y_train_shape": list(ds.y_train.shape),
            "y_test_shape": list(ds.y_test.shape),
        },
        "flatten_feature": {
            "train_shape": list(f_train.shape),
            "test_shape": list(f_test.shape),
            "D_in": int(f_train.shape[1]),
            "rule_pca_max": int(min(f_train.shape[1], f_train.shape[0] - 1)),  # 依 report.txt: 365 -> 364
        },
        "pca_knn_cv": result["pca_knn_cv"],
        "final": result["final"],
        "saved": {
            "scaler": os.path.abspath(scaler_path),
            "pca": os.path.abspath(pca_path),
            "knn": os.path.abspath(knn_path),
            "summary": os.path.abspath(summary_path),
        },
        "note": (
            "PCA+KNN baseline：raw flatten features。StandardScaler/PCA 僅在 train（或 fold-train）fit；"
            "test 僅最後評估一次。PCA 維度上限依 N_train-1（report.txt: 365->364）。"
        ),
    }
    meta_path = os.path.join(outdir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("[DONE] saved: %s", outdir)
    logging.info("  - %s", scaler_path)
    logging.info("  - %s", pca_path)
    logging.info("  - %s", knn_path)
    logging.info("  - %s", summary_path)
    logging.info("  - %s", meta_path)


if __name__ == "__main__":
    main()
