from __future__ import annotations

"""run_paper_cnn_adaboost.py（讀取 train_cnn_feature 產生的 features → PCA → AdaBoost）

核心流程（CNN + PCA + AdaBoost）：
  1) 只讀 pm25_dataset.npz 的 y_train / y_test（本檔不再訓練 CNN）
  2) 讀取 out_dir/cnn_feature/<feat_tag>/features_train.npy 與 features_test.npy
  3) 用 train features 做 5-fold Stratified CV 選 (PCA 維度 + AdaBoost 超參數)
      - 每個 fold 各自 fit StandardScaler（train-only）
      - PCA 也只在該 fold 的 train fit（train-only）
      - AdaBoost 也只在該 fold 的 train fit（train-only）
  4) 用全 train features fit 最終 StandardScaler + PCA + AdaBoost
  5) test 只評一次（不參與調參）

規範：
  - features 必須先由 train_cnn_feature.py 產生；本檔不會、也不應另外再訓練 CNN
  - 所有 scaler/PCA/調參只用 train features；test 僅最後評估一次（避免資料洩漏）

輸入：
  - (必）--data : pm25_dataset.npz（包含 X_train, y_train, X_test, y_test；本檔只用 y）
  - (選）--out_dir : 輸出資料夾（預設 ./results）
  - (選）--feat_tag : train_cnn_feature.py 的 tag（預設 baseline）（可選 baseline_hist_eq、baseline_hist_eq_affine 等）
  - (選）--seed : 隨機種子（預設 42）
  - (選）--cv_splits : CV folds（預設 5）

  - (選）--pca_list : PCA 維度候選（逗號分隔；預設 32,64,128,256，會自動過濾掉 > min(F, N-1)）
  - (選）--n_estimators_list : AdaBoost 弱分類器數量候選（逗號分隔；預設依 N_train 自動產生三點）
  - (選）--learning_rate_list : AdaBoost learning_rate 候選（逗號分隔；預設 0.25,0.5,1.0）
  - (選）--stump_depth : Decision stump 深度（預設 1）
  - (選）--min_leaf : 決策樹葉節點最少樣本（預設 2；小樣本較穩）
  - (選）--use_samme : 若設為 1，使用 SAMME（較保守）；否則用 SAMME.R（預設）

輸出（方便重現）：
  - out_dir/cnn_adaboost/<feat_tag>/cnn_adaboost_feat_scaler.joblib : 最終 feature 標準化器（train-only fit）
  - out_dir/cnn_adaboost/<feat_tag>/cnn_adaboost_pca.joblib         : 最終 PCA
  - out_dir/cnn_adaboost/<feat_tag>/cnn_adaboost.joblib             : 最終 AdaBoost
  - out_dir/cnn_adaboost/<feat_tag>/summary.json                        : train/test accuracy
  - out_dir/cnn_adaboost/<feat_tag>/meta.json                           : 參數、CV 最佳超參數、features 來源、可重現資訊

執行範例：
  python run_paper_cnn_adaboost.py --data pm25_dataset.npz
  python run_paper_cnn_adaboost.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq
  python run_paper_cnn_adaboost.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq_affine
"""

import argparse
import json
import logging
import os
import random
import time
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


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
# 小工具：parse list 參數
# =============================================================================

def _parse_int_list(s: str) -> List[int]:
    items = []
    for x in s.split(","):
        x = x.strip()
        if x:
            items.append(int(x))
    return items


def _parse_float_list(s: str) -> List[float]:
    items = []
    for x in s.split(","):
        x = x.strip()
        if x:
            items.append(float(x))
    return items


# =============================================================================
# PCA + AdaBoost：CV 選參數 → 全 train fit → test 只評一次
# =============================================================================

def _default_n_estimators_grid(n_train: int) -> List[int]:
    """
    依資料集分析（N_train=365，12 類近均衡）給「保守且合理」的弱分類器數量起點：
      - 太小：表現不足
      - 太大：小樣本下容易 overfit、也更耗時
    這裡用 n0 ≈ 0.5*N_train（上限 400，下限 100）當中心點，再做三點 grid。
    """
    n0 = int(round(n_train * 0.5))
    n0 = max(100, min(400, n0))
    cand = sorted(set([max(50, n0 // 2), n0, min(600, n0 * 2)]))
    # sklearn 通常用到幾百就夠了；超過 600 多半不划算
    return [int(x) for x in cand]


def build_adaboost_classifier(
    *,
    base_estimator: DecisionTreeClassifier,
    n_estimators: int,
    learning_rate: float,
    seed: int,
    use_samme: bool,
) -> AdaBoostClassifier:
    """
    針對不同 sklearn 版本做相容：
      - 有些版本用 estimator=，有些用 base_estimator=
      - 有些版本有 algorithm=，有些版本已移除
    """
    sig = inspect.signature(AdaBoostClassifier)

    kwargs: Dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "random_state": int(seed),
    }

    # estimator / base_estimator 相容
    if "estimator" in sig.parameters:
        kwargs["estimator"] = base_estimator
    elif "base_estimator" in sig.parameters:
        kwargs["base_estimator"] = base_estimator
    else:
        raise RuntimeError("你的 sklearn 版 AdaBoostClassifier 沒有 estimator/base_estimator 參數，無法指定弱分類器。")

    # algorithm 參數：有才傳（不然就別傳）
    if "algorithm" in sig.parameters:
        kwargs["algorithm"] = ("SAMME" if use_samme else "SAMME.R")

    return AdaBoostClassifier(**kwargs)


def run_pca_adaboost_with_cv(
    f_train: np.ndarray,
    y_train: np.ndarray,
    f_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    cv_splits: int,
    pca_list: List[int],
    n_estimators_list: List[int],
    learning_rate_list: List[float],
    stump_depth: int,
    min_leaf: int,
    use_samme: bool,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    回傳：
      - result dict（含 cv best、train/test acc）
      - pred_train
      - pred_test
    """
    n_train = int(f_train.shape[0])
    feat_dim = int(f_train.shape[1])

    # PCA 維度不能超過 min(F, N-1)
    max_pca = min(feat_dim, max(1, n_train - 1))
    pca_list_eff = [d for d in pca_list if 1 <= d <= max_pca]
    if len(pca_list_eff) == 0:
        raise ValueError(
            f"PCA 維度候選全部無效：pca_list={pca_list}，但允許範圍是 1..{max_pca}"
        )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    best_mean = -1.0
    best_std = 1e9
    best_params: Dict[str, Any] = {
        "pca_dim": int(pca_list_eff[0]),
        "n_estimators": int(n_estimators_list[0]),
        "learning_rate": float(learning_rate_list[0]),
        "stump_depth": int(stump_depth),
        "min_leaf": int(min_leaf),
        "algorithm": "SAMME" if use_samme else "SAMME.R",
    }

    for pca_dim in pca_list_eff:
        for n_est in n_estimators_list:
            for lr in learning_rate_list:
                fold_scores = []
                for tr_idx, va_idx in cv.split(f_train, y_train):
                    # 1) scaler：train-only fit
                    sc = StandardScaler()
                    X_tr_fold = sc.fit_transform(f_train[tr_idx])
                    X_va_fold = sc.transform(f_train[va_idx])

                    # 2) PCA：train-only fit
                    pca = PCA(n_components=int(pca_dim), random_state=seed)
                    Z_tr = pca.fit_transform(X_tr_fold)
                    Z_va = pca.transform(X_va_fold)

                    # 3) AdaBoost（decision stump）
                    base = DecisionTreeClassifier(
                        max_depth=int(stump_depth),
                        min_samples_leaf=int(min_leaf),
                        random_state=seed,
                    )
                    clf = build_adaboost_classifier(
                        base_estimator=base,
                        n_estimators=int(n_est),
                        learning_rate=float(lr),
                        seed=seed,
                        use_samme=use_samme,
                    )
                    clf.fit(Z_tr, y_train[tr_idx])

                    pred_va = clf.predict(Z_va)
                    fold_scores.append(float(accuracy_score(y_train[va_idx], pred_va)))

                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))

                improved = (mean_score > best_mean + 1e-12) or (
                    abs(mean_score - best_mean) <= 1e-12 and std_score < best_std - 1e-12
                )
                if improved:
                    best_mean = mean_score
                    best_std = std_score
                    best_params = {
                        "pca_dim": int(pca_dim),
                        "n_estimators": int(n_est),
                        "learning_rate": float(lr),
                        "stump_depth": int(stump_depth),
                        "min_leaf": int(min_leaf),
                        "algorithm": "SAMME" if use_samme else "SAMME.R",
                    }
                    logging.info(
                        "[PCA+AB-CV] new best -> pca=%d | n_est=%d | lr=%.4f | alg=%s | cv_mean=%.4f | cv_std=%.4f",
                        best_params["pca_dim"],
                        best_params["n_estimators"],
                        best_params["learning_rate"],
                        best_params["algorithm"],
                        best_mean,
                        best_std,
                    )

    logging.info(
        "[PCA+AB-CV] BEST -> pca=%d | n_est=%d | lr=%.4f | alg=%s | cv_mean=%.4f | cv_std=%.4f",
        best_params["pca_dim"],
        best_params["n_estimators"],
        best_params["learning_rate"],
        best_params["algorithm"],
        best_mean,
        best_std,
    )

    # 最終：用全 train fit scaler + PCA + AdaBoost，test 只做一次評估
    scaler_final = StandardScaler()
    Xtr_s = scaler_final.fit_transform(f_train)  # train-only fit
    Xte_s = scaler_final.transform(f_test)

    pca_final = PCA(n_components=int(best_params["pca_dim"]), random_state=seed)
    Ztr = pca_final.fit_transform(Xtr_s)  # train-only fit
    Zte = pca_final.transform(Xte_s)

    base_final = DecisionTreeClassifier(
        max_depth=int(best_params["stump_depth"]),
        min_samples_leaf=int(best_params["min_leaf"]),
        random_state=seed,
    )
    clf_final = build_adaboost_classifier(
        base_estimator=base_final,
        n_estimators=int(best_params["n_estimators"]),
        learning_rate=float(best_params["learning_rate"]),
        seed=seed,
        use_samme=(best_params["algorithm"] == "SAMME"),
    )
    clf_final.fit(Ztr, y_train)

    pred_train = clf_final.predict(Ztr)
    pred_test = clf_final.predict(Zte)

    acc_train = float(accuracy_score(y_train, pred_train))
    acc_test = float(accuracy_score(y_test, pred_test))

    result = {
        "pca_adaboost_cv": {
            "best_params": best_params,
            "cv_mean_acc": best_mean,
            "cv_std_acc": best_std,
            "n_splits": int(cv_splits),
            "pca_list_effective": [int(x) for x in pca_list_eff],
            "n_estimators_list": [int(x) for x in n_estimators_list],
            "learning_rate_list": [float(x) for x in learning_rate_list],
        },
        "final": {
            "train_acc": acc_train,
            "test_acc": acc_test,
        },
        "artifacts": {
            "scaler": scaler_final,
            "pca": pca_final,
            "adaboost": clf_final,
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
    p.add_argument("--cv_splits", type=int, default=5)

    p.add_argument("--pca_list", type=str, default="32,64,128,256", help="PCA 維度候選（逗號分隔）")
    p.add_argument("--n_estimators_list", type=str, default="", help="AdaBoost n_estimators 候選（逗號分隔；留空=依 N_train 自動）")
    p.add_argument("--learning_rate_list", type=str, default="0.25,0.5,1.0", help="AdaBoost learning_rate 候選（逗號分隔）")

    p.add_argument("--stump_depth", type=int, default=1, help="弱分類器（決策樹）深度；1=decision stump")
    p.add_argument("--min_leaf", type=int, default=2, help="決策樹葉節點最少樣本（小樣本較穩）")
    p.add_argument("--use_samme", type=int, default=0, help="1=使用 SAMME（較保守）；0=使用 SAMME.R（預設）")
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

    # 4) grid：依資料集分析（N_train=365）做合理預設，再用 CV 在 train features 內挑最佳
    pca_list = _parse_int_list(args.pca_list)

    if args.n_estimators_list.strip() == "":
        n_estimators_list = _default_n_estimators_grid(int(ds.y_train.shape[0]))
        logging.info("[GRID] n_estimators_list auto (by N_train=%d) -> %s", int(ds.y_train.shape[0]), str(n_estimators_list))
    else:
        n_estimators_list = _parse_int_list(args.n_estimators_list)

    learning_rate_list = _parse_float_list(args.learning_rate_list)

    # 5) run PCA + AdaBoost
    start = time.time()
    result, _, _ = run_pca_adaboost_with_cv(
        f_train,
        ds.y_train,
        f_test,
        ds.y_test,
        seed=args.seed,
        cv_splits=int(args.cv_splits),
        pca_list=pca_list,
        n_estimators_list=n_estimators_list,
        learning_rate_list=learning_rate_list,
        stump_depth=int(args.stump_depth),
        min_leaf=int(args.min_leaf),
        use_samme=(int(args.use_samme) == 1),
    )
    elapsed = time.time() - start

    train_acc = float(result["final"]["train_acc"])
    test_acc = float(result["final"]["test_acc"])
    logging.info("[PCA+AdaBoost] train_acc=%.4f | test_acc=%.4f | time=%.1fs", train_acc, test_acc, elapsed)

    # 6) save artifacts（輸出獨立資料夾，避免跟 features 混在一起）
    outdir = os.path.join(args.out_dir, "cnn_adaboost", args.feat_tag)
    os.makedirs(outdir, exist_ok=True)

    scaler_path = os.path.join(outdir, "cnn_adaboost_feat_scaler.joblib")
    pca_path = os.path.join(outdir, "cnn_adaboost_pca.joblib")
    ab_path = os.path.join(outdir, "cnn_adaboost.joblib")

    joblib.dump(result["artifacts"]["scaler"], scaler_path)
    joblib.dump(result["artifacts"]["pca"], pca_path)
    joblib.dump(result["artifacts"]["adaboost"], ab_path)

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
        "pca_adaboost_cv": result["pca_adaboost_cv"],
        "final": result["final"],
        "saved": {
            "scaler": os.path.abspath(scaler_path),
            "pca": os.path.abspath(pca_path),
            "adaboost": os.path.abspath(ab_path),
            "summary": os.path.abspath(summary_path),
        },
        "feature_stage_meta_json": feat_meta,  # 可能為 None（若 features 資料夾沒 meta.json）
        "note": (
            "PCA 維度與 AdaBoost 超參數以 train features 的 Stratified CV 選出；"
            "StandardScaler/PCA 皆為 train-only fit；test 僅最後評估一次。"
        ),
    }
    meta_path = os.path.join(outdir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("[DONE] saved: %s", outdir)
    logging.info("  - %s", scaler_path)
    logging.info("  - %s", pca_path)
    logging.info("  - %s", ab_path)
    logging.info("  - %s", summary_path)
    logging.info("  - %s", meta_path)


if __name__ == "__main__":
    main()
