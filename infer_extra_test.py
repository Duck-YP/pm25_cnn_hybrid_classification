# -*- coding: utf-8 -*-
from __future__ import annotations

"""infer_extra_test.py

【單一入口：兩種模式】

(1) Extra test（老師額外測試資料，通常只有 X）：
    python infer_extra_test.py -data extra_test.npz

    - 只做「載入已訓練 artifacts → 預測 → 輸出」，完全不會對 extra data 做 fit / 調參。
    - 會一次跑完：
        * Paper-1: CNN + SVM
        * Paper-2: CNN + PCA（PCA 空間 Nearest Centroid）
        * Paper-3: CNN + AdaBoost（含 PCA）
        * Baseline: Raw SVM
        * Baseline: PCA + KNN
    - CNN 系列會自動跑 3 個 tag（ablation）：
        baseline / baseline_hist_eq / baseline_hist_eq_affine

(2) pm25_dataset.npz（有 y_test）：
    python infer_extra_test.py -data pm25_dataset.npz

    - 會整理報告/投影片需要的重點檔案到 ./results/report/ ：
        * Results 表（所有方法 Accuracy + mean±std）
        * Ablation（有/無 Hist+Affine）
        * Error analysis（混淆矩陣、最容易混淆的月份）
        * 結論（誰最好、為什麼；以及 ablation 觀察）

設計原則：
  - 此檔案 *不會* 進行任何訓練。
  - 此檔案 *不會* 對輸入資料做任何 fit（例如 scaler/PCA 的 fit）。
  - 所有 scaler/PCA/model 都必須先由你既有的訓練腳本產生並存到 ./results/...。

備註：
  - CNN 的 feature 抽取會重用 train_cnn_feature.py 的模型定義（CNNFeatureNet）與 preprocess 函式。
  - 如果 extra_test 很大，CNN 抽 feature 在 CPU 會慢一點，但流程仍正確。
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# sklearn / joblib
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# matplotlib：只在 pm25_dataset mode 需要畫 confusion matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 預設設定
# =============================================================================
DEFAULT_OUT_DIR = "./results"
CNN_TAGS = ["baseline", "baseline_hist_eq", "baseline_hist_eq_affine"]
ALL_METHODS = [
    "cnn_svm",
    "cnn_pca",
    "cnn_adaboost",
    "raw_svm",
    "pca_knn",
]

# extra_test npz 可能的 key 名稱（只要找到第一個就用）
X_KEYS_CANDIDATES = [
    "X_extra",
    "X",
    "data",
    "images",
    "x",
    "X_test",
]


# =============================================================================
# 小工具：I/O
# =============================================================================

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def read_np_like(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """讀 npy/npz。

    Returns:
        (array, info)
        info 會包含：
          - kind: "npy" 或 "npz"
          - keys: npz 的 keys（npy 則為空 list）
    """
    path = str(path)
    if path.lower().endswith(".npy"):
        x = np.load(path)
        return x, {"kind": "npy", "keys": []}

    if not path.lower().endswith(".npz"):
        raise ValueError(f"只支援 .npy / .npz，收到：{path}")

    d = np.load(path, allow_pickle=False)
    keys = list(d.files)

    # full dataset
    if set(["X_train", "y_train", "X_test", "y_test"]).issubset(set(keys)):
        # 這裡回傳 None，讓上層用 load_full_dataset 讀
        return np.empty((0,), dtype=np.float32), {"kind": "npz_full", "keys": keys}

    # extra test：找 X key
    x_key = None
    for k in X_KEYS_CANDIDATES:
        if k in keys:
            x_key = k
            break
    if x_key is None:
        raise KeyError(
            "extra_test.npz 找不到可用的 X key。"
            f"目前 keys={keys}，你可以把 X 改名為其中之一：{X_KEYS_CANDIDATES}"
        )

    x = d[x_key]
    return x, {"kind": "npz_extra", "keys": keys, "x_key": x_key}


def load_full_dataset(npz_path: str | Path) -> Dict[str, np.ndarray]:
    d = np.load(str(npz_path), allow_pickle=False)
    return {
        "X_train": d["X_train"],
        "y_train": d["y_train"],
        "X_test": d["X_test"],
        "y_test": d["y_test"],
    }


def write_csv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# 讀取 results 內既有 summary/meta（用於 mean±std；不依賴它也可跑）
# =============================================================================

def safe_read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pick_metric_from_summary(summary: Dict[str, Any], *, kind: str) -> Tuple[Optional[float], Optional[float]]:
    """嘗試從 summary.json 找 cv mean/std。

    kind:
      - "cv"：回傳 (mean, std)
      - "test"：回傳 (acc, None)
    """
    if not isinstance(summary, dict):
        return None, None

    # 常見 key 候選（你不同腳本可能稍有差異，這裡故意寫得寬鬆）
    if kind == "cv":
        mean_keys = ["cv_acc_mean", "cv_mean", "cv_accuracy_mean", "cv_acc", "cv_acc_avg", "cv_val_acc_mean"]
        std_keys = ["cv_acc_std", "cv_std", "cv_accuracy_std", "cv_acc_sd", "cv_val_acc_std"]
        mean = None
        std = None
        for k in mean_keys:
            if k in summary and isinstance(summary[k], (int, float)):
                mean = float(summary[k])
                break
        for k in std_keys:
            if k in summary and isinstance(summary[k], (int, float)):
                std = float(summary[k])
                break
        return mean, std

    if kind == "test":
        test_keys = ["test_acc", "test_accuracy", "acc_test", "test_accuracy_final"]
        for k in test_keys:
            if k in summary and isinstance(summary[k], (int, float)):
                return float(summary[k]), None

    return None, None


def read_cv_table_mean_std(cv_csv_path: str | Path) -> Tuple[Optional[float], Optional[float]]:
    """讀 cv_table.csv，推估 mean/std。

    你的 cv_table.csv 目前確定存在於 cnn_pca/<tag>/cv_table.csv。
    其他方法若沒有，也不影響主要流程。
    """
    try:
        import pandas as pd  # 若沒裝 pandas，會走 except

        df = pd.read_csv(cv_csv_path)
        # 優先找含 acc 的欄位
        cand_cols = [c for c in df.columns if "acc" in c.lower()]
        if len(cand_cols) == 0:
            # 找最後一個 numeric 欄
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(num_cols) == 0:
                return None, None
            col = num_cols[-1]
        else:
            col = cand_cols[0]
        arr = df[col].to_numpy(dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    except Exception:
        # 不依賴 pandas 的簡易讀法
        try:
            with open(cv_csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if len(rows) <= 1:
                return None, None
            header = rows[0]
            # 找含 acc 的欄
            idx = None
            for i, c in enumerate(header):
                if "acc" in c.lower():
                    idx = i
                    break
            if idx is None:
                # 找最後一欄
                idx = len(header) - 1
            vals = []
            for r in rows[1:]:
                if idx < len(r):
                    try:
                        vals.append(float(r[idx]))
                    except Exception:
                        pass
            if len(vals) == 0:
                return None, None
            vals = np.asarray(vals, dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            return mean, std
        except Exception:
            return None, None


# =============================================================================
# CNN feature extraction（extra_test 才需要）
# =============================================================================

@dataclass
class Normalizer:
    mean: float
    std: float
    preprocess: str
    clip_p: float
    clip_hi: Optional[float]


def load_normalizer_npz(path: str | Path) -> Normalizer:
    d = np.load(str(path), allow_pickle=False)
    mean = float(d["mean"].item())
    std = float(d["std"].item())
    preprocess = str(d["preprocess"].item())
    clip_p = float(d["clip_p"].item())
    clip_hi_raw = float(d["clip_hi"].item())
    clip_hi = None if np.isnan(clip_hi_raw) else float(clip_hi_raw)
    return Normalizer(mean=mean, std=std, preprocess=preprocess, clip_p=clip_p, clip_hi=clip_hi)


def pick_torch_device(device: str) -> str:
    """回傳 train_cnn_feature.pick_device 可接受的字串。"""
    device = (device or "auto").lower()
    if device in ("auto", "cpu", "cuda", "mps"):
        return device
    return "auto"


def extract_cnn_features_from_raw(
    X: np.ndarray,
    *,
    out_dir: str | Path,
    tag: str,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """用已訓練好的 cnn_refit.pt + normalizer.npz 抽取 features。

    注意：這裡只做 forward，不訓練。
    """
    # 直接重用你現有 train_cnn_feature.py 的模型/前處理（避免你維護兩份 CNN）
    try:
        from train_cnn_feature import (
            CNNFeatureNet,
            PM25GridDataset,
            apply_preprocess,
            normalize,
            pick_device,
        )
        import torch
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError(
            "無法 import train_cnn_feature.py 的 CNN 定義。\n"
            "請確認 infer_extra_test.py 與 train_cnn_feature.py 在同一資料夾，且已安裝 torch。\n"
            f"原始錯誤：{e}"
        )

    out_dir = Path(out_dir)
    feat_dir = out_dir / "cnn_feature" / tag
    refit_path = feat_dir / "cnn_refit.pt"
    norm_path = feat_dir / "normalizer.npz"
    if not refit_path.exists():
        raise FileNotFoundError(f"找不到 CNN refit 權重：{refit_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"找不到 normalizer：{norm_path}")

    norm = load_normalizer_npz(norm_path)

    # preprocess
    if norm.preprocess.lower() == "log1p":
        Xp = apply_preprocess(X, norm.preprocess, clip_hi=norm.clip_hi)
    else:
        Xp = apply_preprocess(X, norm.preprocess)
    Xn = normalize(Xp, norm.mean, norm.std).astype(np.float32, copy=False)

    dev = pick_device(pick_torch_device(device))

    # model
    ckpt = torch.load(refit_path, map_location="cpu")
    feature_dim = int(ckpt.get("feature_dim", 128))
    model = CNNFeatureNet(num_classes=12, feature_dim=feature_dim).to(dev)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # dataset/dataloader（不做 augmentation）
    y_dummy = np.zeros((Xn.shape[0],), dtype=np.int64)
    ds = PM25GridDataset(Xn, y_dummy, train_affine=False, device=dev)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(dev)
            # 取 feature 向量：CNNFeatureNet 在 train_cnn_feature.py 定義的是 encode()
            if hasattr(model, "encode"):
                fb = model.encode(xb)  # (B, feature_dim=128)
            elif hasattr(model, "extract_feature"):
                fb = model.extract_feature(xb)
            else:
                raise RuntimeError("找不到抽 feature 的方法（需要 model.encode 或 model.extract_feature）")

            feats.append(fb.detach().cpu().numpy())

    return np.concatenate(feats, axis=0)


# =============================================================================
# 各方法的推論（全部是 pure inference）
# =============================================================================


def flatten_X(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"預期 X shape=(N,H,W)，收到：{X.shape}")
    return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)


def predict_raw_svm(X_raw: np.ndarray, *, out_dir: str | Path) -> np.ndarray:
    base = Path(out_dir) / "baseline_raw_svm"
    scaler_path = base / "raw_svm_feat_scaler.joblib"
    svm_path = base / "raw_svm_svm.joblib"
    if not scaler_path.exists() or not svm_path.exists():
        raise FileNotFoundError(f"raw_svm artifacts 不齊：{scaler_path} / {svm_path}")

    Xf = flatten_X(X_raw)
    scaler = joblib.load(scaler_path)
    svm = joblib.load(svm_path)
    Xs = scaler.transform(Xf)
    y_pred = svm.predict(Xs)
    return y_pred.astype(np.int64)


def predict_pca_knn(X_raw: np.ndarray, *, out_dir: str | Path) -> np.ndarray:
    base = Path(out_dir) / "baseline_pca_knn"
    scaler_path = base / "pca_knn_feat_scaler.joblib"
    pca_path = base / "pca_knn_pca.joblib"
    knn_path = base / "pca_knn_knn.joblib"
    if not scaler_path.exists() or not pca_path.exists() or not knn_path.exists():
        raise FileNotFoundError(f"pca_knn artifacts 不齊：{scaler_path} / {pca_path} / {knn_path}")

    Xf = flatten_X(X_raw)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    knn = joblib.load(knn_path)
    Xs = scaler.transform(Xf)
    Xp = pca.transform(Xs)
    y_pred = knn.predict(Xp)
    return y_pred.astype(np.int64)


def predict_cnn_svm(
    features: np.ndarray,
    *,
    out_dir: str | Path,
    tag: str,
) -> np.ndarray:
    base = Path(out_dir) / "cnn_svm" / tag
    scaler_path = base / "cnn_svm_feat_scaler.joblib"
    svm_path = base / "cnn_svm_svm.joblib"
    if not scaler_path.exists() or not svm_path.exists():
        raise FileNotFoundError(f"cnn_svm artifacts 不齊：{scaler_path} / {svm_path}")

    scaler = joblib.load(scaler_path)
    svm = joblib.load(svm_path)
    Xs = scaler.transform(features)
    y_pred = svm.predict(Xs)
    return y_pred.astype(np.int64)


def predict_cnn_pca(features, *, out_dir, tag):
    base = Path(out_dir) / "cnn_pca" / tag
    scaler = joblib.load(base / "cnn_pca_feat_scaler.joblib")
    pca = joblib.load(base / "cnn_pca_pca.joblib")
    centroid = joblib.load(base / "cnn_pca_centroid.joblib")

    Xp = pca.transform(scaler.transform(features))

    # A) 若剛好是 estimator（有 .predict）就直接用
    if hasattr(centroid, "predict"):
        return np.asarray(centroid.predict(Xp), dtype=np.int64)

    # B) 你的真實情況：dict(classes, centroids)
    if isinstance(centroid, dict) and "classes" in centroid and "centroids" in centroid:
        classes = np.asarray(centroid["classes"], dtype=np.int64)      # (C,)
        centroids = np.asarray(centroid["centroids"], dtype=np.float32) # (C,D)
        d2 = ((Xp[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # (N,C)
        idx = d2.argmin(axis=1)
        return classes[idx].astype(np.int64)

    raise TypeError(f"cnn_pca_centroid.joblib 型別不支援：{type(centroid)}")


def predict_cnn_adaboost(
    features: np.ndarray,
    *,
    out_dir: str | Path,
    tag: str,
) -> np.ndarray:
    base = Path(out_dir) / "cnn_adaboost" / tag
    scaler_path = base / "cnn_adaboost_feat_scaler.joblib"
    pca_path = base / "cnn_adaboost_pca.joblib"
    clf_path = base / "cnn_adaboost.joblib"
    if not scaler_path.exists() or not pca_path.exists() or not clf_path.exists():
        raise FileNotFoundError(
            f"cnn_adaboost artifacts 不齊：{scaler_path} / {pca_path} / {clf_path}"
        )

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    clf = joblib.load(clf_path)
    Xs = scaler.transform(features)
    Xp = pca.transform(Xs)
    y_pred = clf.predict(Xp)
    return y_pred.astype(np.int64)


# =============================================================================
# 統一流程：跑 extra_test 或跑 report
# =============================================================================

def save_predictions_bundle(
    *,
    out_dir: str | Path,
    method: str,
    tag: str,
    y_pred_0_11: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """輸出預測檔（0~11 與 1~12）"""
    out_dir = Path(out_dir)
    outp = out_dir / "infer_extra" / method / tag
    ensure_dir(outp)

    y_pred_0_11 = y_pred_0_11.astype(np.int64)
    y_month_1_12 = (y_pred_0_11 + 1).astype(np.int64)

    np.save(outp / "pred_label0_11.npy", y_pred_0_11)
    np.save(outp / "pred_month1_12.npy", y_month_1_12)

    # CSV：對老師最友善
    with open(outp / "pred_month1_12.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "month"])
        for i, m in enumerate(y_month_1_12.tolist()):
            w.writerow([i, m])

    write_json(outp / "meta.json", meta)


def try_load_precomputed_features(
    *,
    out_dir: str | Path,
    tag: str,
    split: str,
) -> Optional[np.ndarray]:
    """優先讀 results/cnn_feature/<tag>/features_{train|test}.npy。"""
    p = Path(out_dir) / "cnn_feature" / tag / f"features_{split}.npy"
    if p.exists():
        return np.load(p)
    return None


def run_extra_mode(
    *,
    X_extra: np.ndarray,
    data_info: Dict[str, Any],
    out_dir: str | Path,
    device: str,
    methods: Sequence[str],
    cnn_tags: Sequence[str],
) -> None:
    print("========== [MODE] extra_test：純推論（不 fit） ==========")
    print(f"[INFO] X_extra shape={X_extra.shape} | dtype={X_extra.dtype}")
    print(f"[INFO] input_info: {data_info}")
    print(f"[INFO] out_dir: {out_dir}")

    # Baselines（不需要 tag）
    if "raw_svm" in methods:
        try:
            y = predict_raw_svm(X_extra, out_dir=out_dir)
            save_predictions_bundle(
                out_dir=out_dir,
                method="raw_svm",
                tag="default",
                y_pred_0_11=y,
                meta={"mode": "extra", "method": "raw_svm", "tag": "default", "note": "pure inference"},
            )
            print("[DONE] raw_svm")
        except Exception as e:
            print(f"[WARN] raw_svm 失敗：{e}")

    if "pca_knn" in methods:
        try:
            y = predict_pca_knn(X_extra, out_dir=out_dir)
            save_predictions_bundle(
                out_dir=out_dir,
                method="pca_knn",
                tag="default",
                y_pred_0_11=y,
                meta={"mode": "extra", "method": "pca_knn", "tag": "default", "note": "pure inference"},
            )
            print("[DONE] pca_knn")
        except Exception as e:
            print(f"[WARN] pca_knn 失敗：{e}")

    # CNN methods（每個 tag 都跑）
    for tag in cnn_tags:
        # 先抽 feature（一次抽好，三個 head 共用）
        try:
            feats = extract_cnn_features_from_raw(X_extra, out_dir=out_dir, tag=tag, device=device)
            print(f"[INFO] CNN features extracted | tag={tag} | shape={feats.shape}")
        except Exception as e:
            print(f"[WARN] CNN feature 抽取失敗（tag={tag}）：{e}")
            continue

        if "cnn_svm" in methods:
            try:
                y = predict_cnn_svm(feats, out_dir=out_dir, tag=tag)
                save_predictions_bundle(
                    out_dir=out_dir,
                    method="cnn_svm",
                    tag=tag,
                    y_pred_0_11=y,
                    meta={"mode": "extra", "method": "cnn_svm", "tag": tag, "note": "pure inference"},
                )
                print(f"[DONE] cnn_svm | tag={tag}")
            except Exception as e:
                print(f"[WARN] cnn_svm 失敗（tag={tag}）：{e}")

        if "cnn_pca" in methods:
            try:
                y = predict_cnn_pca(feats, out_dir=out_dir, tag=tag)
                save_predictions_bundle(
                    out_dir=out_dir,
                    method="cnn_pca",
                    tag=tag,
                    y_pred_0_11=y,
                    meta={"mode": "extra", "method": "cnn_pca", "tag": tag, "note": "pure inference"},
                )
                print(f"[DONE] cnn_pca | tag={tag}")
            except Exception as e:
                print(f"[WARN] cnn_pca 失敗（tag={tag}）：{e}")

        if "cnn_adaboost" in methods:
            try:
                y = predict_cnn_adaboost(feats, out_dir=out_dir, tag=tag)
                save_predictions_bundle(
                    out_dir=out_dir,
                    method="cnn_adaboost",
                    tag=tag,
                    y_pred_0_11=y,
                    meta={"mode": "extra", "method": "cnn_adaboost", "tag": tag, "note": "pure inference"},
                )
                print(f"[DONE] cnn_adaboost | tag={tag}")
            except Exception as e:
                print(f"[WARN] cnn_adaboost 失敗（tag={tag}）：{e}")

    print("\n[OK] extra_test 推論完成。輸出位置：./results/infer_extra/")


@dataclass
class ResultRow:
    method: str
    tag: str
    test_acc: float
    macro_f1: float
    artifact_dir: str


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, mf1


def run_report_mode(
    *,
    dataset_npz: str | Path,
    out_dir: str | Path,
    device: str,
    methods: Sequence[str],
    cnn_tags: Sequence[str],
) -> None:
    print("========== [MODE] pm25_dataset：整理報告檔 ==========")
    data = load_full_dataset(dataset_npz)
    X_test = data["X_test"].astype(np.float32, copy=False)
    y_test = data["y_test"].astype(np.int64, copy=False)

    # [FIX] 資料集標籤可能是 1~12（月），模型與所有 head 都是 0~11
    y_min, y_max = int(y_test.min()), int(y_test.max())
    if (y_min, y_max) == (1, 12):
        y_test = y_test - 1
    elif (y_min, y_max) == (0, 11):
        pass
    else:
        raise ValueError(f"y_test label range 異常：min={y_min}, max={y_max}（預期 0~11 或 1~12）")

    print(f"[INFO] y_test range(after fix) = [{int(y_test.min())}, {int(y_test.max())}]")
    print(f"[INFO] X_test={X_test.shape} | y_test={y_test.shape}")

    rows: List[ResultRow] = []

    # Baselines（不需要 tag）
    if "raw_svm" in methods:
        base = Path(out_dir) / "baseline_raw_svm"
        y_pred = predict_raw_svm(X_test, out_dir=out_dir)
        test_acc, mf1 = compute_metrics(y_test, y_pred)
        rows.append(
            ResultRow(
                method="raw_svm",
                tag="default",
                test_acc=test_acc,
                macro_f1=mf1,
                artifact_dir=str(base),
            )
        )

        print(f"[OK] raw_svm | test_acc={test_acc:.4f}")

    if "pca_knn" in methods:
        base = Path(out_dir) / "baseline_pca_knn"
        y_pred = predict_pca_knn(X_test, out_dir=out_dir)
        test_acc, mf1 = compute_metrics(y_test, y_pred)
        rows.append(
            ResultRow(
                method="pca_knn",
                tag="default",
                test_acc=test_acc,
                macro_f1=mf1,
                artifact_dir=str(base),
            )
        )
        print(f"[OK] pca_knn | test_acc={test_acc:.4f}")

    # CNN methods（每個 tag 都跑）
    for tag in cnn_tags:
        # 優先讀 precomputed features_test；沒有才用 cnn_refit 抽一次
        feats_test = try_load_precomputed_features(out_dir=out_dir, tag=tag, split="test")
        if feats_test is None:
            print(f"[WARN] 找不到 precomputed features_test.npy（tag={tag}），改用 CNN forward 抽取")
            feats_test = extract_cnn_features_from_raw(X_test, out_dir=out_dir, tag=tag, device=device)

        # cnn_svm
        if "cnn_svm" in methods:
            base = Path(out_dir) / "cnn_svm" / tag
            y_pred = predict_cnn_svm(feats_test, out_dir=out_dir, tag=tag)
            test_acc, mf1 = compute_metrics(y_test, y_pred)
            rows.append(
                ResultRow(
                    method="cnn_svm",
                    tag=tag,
                    test_acc=test_acc,
                    macro_f1=mf1,
                    artifact_dir=str(base),
                )
            )
            print(f"[OK] cnn_svm | tag={tag} | test_acc={test_acc:.4f}")

        # cnn_pca
        if "cnn_pca" in methods:
            base = Path(out_dir) / "cnn_pca" / tag
            y_pred = predict_cnn_pca(feats_test, out_dir=out_dir, tag=tag)
            test_acc, mf1 = compute_metrics(y_test, y_pred)
            rows.append(
                ResultRow(
                    method="cnn_pca",
                    tag=tag,
                    test_acc=test_acc,
                    macro_f1=mf1,
                    artifact_dir=str(base),
                )
            )
            print(f"[OK] cnn_pca | tag={tag} | test_acc={test_acc:.4f}")

        # cnn_adaboost
        if "cnn_adaboost" in methods:
            base = Path(out_dir) / "cnn_adaboost" / tag
            y_pred = predict_cnn_adaboost(feats_test, out_dir=out_dir, tag=tag)
            test_acc, mf1 = compute_metrics(y_test, y_pred)
            rows.append(
                ResultRow(
                    method="cnn_adaboost",
                    tag=tag,
                    test_acc=test_acc,
                    macro_f1=mf1,
                    artifact_dir=str(base),
                )
            )
            print(f"[OK] cnn_adaboost | tag={tag} | test_acc={test_acc:.4f}")

    # -------------------------
    # 輸出報告檔
    # -------------------------
    report_dir = Path(out_dir) / "report" / f"{Path(dataset_npz).stem}"
    ensure_dir(report_dir)

    # Results table
    rows_sorted = sorted(rows, key=lambda r: r.test_acc, reverse=True)
    table_csv = report_dir / "results_table.csv"
    write_csv(
        table_csv,
        header=["method", "tag", "test_acc", "macro_f1", "artifact_dir"],
        rows=[
            [r.method, r.tag, f"{r.test_acc:.6f}", f"{r.macro_f1:.6f}", r.artifact_dir]
            for r in rows_sorted
        ],
    )

    # Ablation：baseline vs baseline_hist_eq vs baseline_hist_eq_affine（同一 method）
    ablation_csv = report_dir / "ablation_hist_eq_affine.csv"
    ab_rows = []
    for m in ["cnn_svm", "cnn_pca", "cnn_adaboost"]:
        base_row = next((r for r in rows_sorted if r.method == m and r.tag == "baseline"), None)
        he_row   = next((r for r in rows_sorted if r.method == m and r.tag == "baseline_hist_eq"), None)
        aff_row  = next((r for r in rows_sorted if r.method == m and r.tag == "baseline_hist_eq_affine"), None)

        # 至少要有 baseline 才有意義
        if base_row is None:
            continue

        b = base_row.test_acc
        he = he_row.test_acc if he_row is not None else None
        af = aff_row.test_acc if aff_row is not None else None

        ab_rows.append(
            [
                m,
                f"{b:.6f}",
                f"{he:.6f}" if he is not None else "",
                f"{af:.6f}" if af is not None else "",
                f"{(he - b):+.6f}" if he is not None else "",
                f"{(af - b):+.6f}" if af is not None else "",
            ]
        )

    write_csv(
        ablation_csv,
        header=[
            "method",
            "test_acc_baseline",
            "test_acc_hist_eq",
            "test_acc_hist_eq_affine",
            "delta_hist_eq_minus_baseline",
            "delta_hist_eq_affine_minus_baseline",
        ],
        rows=ab_rows,
    )

    # Best method：做 confusion matrix + error analysis
    best = rows_sorted[0]

    # 重新算 best 的 y_pred（確保與 rows 一致）
    if best.method == "raw_svm":
        y_best = predict_raw_svm(X_test, out_dir=out_dir)
    elif best.method == "pca_knn":
        y_best = predict_pca_knn(X_test, out_dir=out_dir)
    else:
        feats_test_best = try_load_precomputed_features(out_dir=out_dir, tag=best.tag, split="test")
        if feats_test_best is None:
            feats_test_best = extract_cnn_features_from_raw(X_test, out_dir=out_dir, tag=best.tag, device=device)
        if best.method == "cnn_svm":
            y_best = predict_cnn_svm(feats_test_best, out_dir=out_dir, tag=best.tag)
        elif best.method == "cnn_pca":
            y_best = predict_cnn_pca(feats_test_best, out_dir=out_dir, tag=best.tag)
        else:
            y_best = predict_cnn_adaboost(feats_test_best, out_dir=out_dir, tag=best.tag)

    cm = confusion_matrix(y_test, y_best, labels=list(range(12)))
    np.savetxt(report_dir / "confusion_best.csv", cm.astype(int), fmt="%d", delimiter=",")

    # confusion matrix png（不指定顏色；用預設 colormap）
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (best={best.method} | tag={best.tag})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_best.png", dpi=200)
    plt.close()

    # error analysis：列出最常見的 off-diagonal 混淆
    pairs = []
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    topk = pairs[:20]
    write_csv(
        report_dir / "error_top_confusions.csv",
        header=["count", "true_label0_11", "pred_label0_11", "true_month1_12", "pred_month1_12"],
        rows=[[c, i, j, i + 1, j + 1] for (c, i, j) in topk],
    )

    # 結論文字
    conclusion = {
        "best": {
            "method": best.method,
            "tag": best.tag,
            "test_acc": best.test_acc,
            "macro_f1": best.macro_f1,
        },
        "top3": [
            {
                "method": r.method,
                "tag": r.tag,
                "test_acc": r.test_acc,
                "macro_f1": r.macro_f1,
            }
            for r in rows_sorted[:3]
        ],
        "ablation_hist_eq_affine": ab_rows,
        "notes": [
            "Results 表請看 results_table.csv（含 test_acc 與 cv mean±std）。",
            "Ablation 表請看 ablation_hist_eq_affine.csv（同方法 baseline vs hist_eq_affine）。",
            "Error analysis：confusion_best.png / error_top_confusions.csv",
        ],
    }
    write_json(report_dir / "conclusion.json", conclusion)

    # 也輸出一份 markdown（方便直接貼到投影片/報告）
    md_lines = []
    md_lines.append("# Results Summary\n")
    md_lines.append(f"Best: **{best.method}** (tag={best.tag}) | test_acc={best.test_acc:.4f} | macro_f1={best.macro_f1:.4f}\n")
    md_lines.append("## Top-3 by test accuracy\n")
    for i, r in enumerate(rows_sorted[:3], 1):
        md_lines.append(
            f"{i}. {r.method} (tag={r.tag}) | test_acc={r.test_acc:.4f}"
        )
    md_lines.append("\n## Ablation: baseline vs hist_eq vs hist_eq_affine\n")
    if len(ab_rows) == 0:
        md_lines.append("(NA) 缺 baseline 的結果，無法做 ablation。")
    else:
        # ab_rows 欄位順序：
        # [method, baseline, hist_eq, hist_eq_affine, delta_hist_eq, delta_hist_eq_affine]
        for m, bacc, heacc, afacc, d_he, d_af in ab_rows:
            parts = [f"- {m}: baseline={bacc}"]
            if heacc != "":
                parts.append(f"→ hist_eq={heacc} (Δ {d_he})")
            if afacc != "":
                parts.append(f"→ hist_eq_affine={afacc} (Δ {d_af})")
            md_lines.append(" ".join(parts))

    md_lines.append("\n## Error analysis\n")
    md_lines.append("- confusion_best.png：最佳方法混淆矩陣")
    md_lines.append("- error_top_confusions.csv：最常見的月份混淆對")

    with open(report_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n[OK] report 產出完成：")
    print(f"  - {table_csv}")
    print(f"  - {ablation_csv}")
    print(f"  - {report_dir / 'confusion_best.png'}")
    print(f"  - {report_dir / 'REPORT.md'}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra-test inference + report generator (pure inference, no training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 你希望老師用：-data extra_test.npz
    p.add_argument("-data", "--data", required=True, help="輸入 .npz 或 .npy（extra_test 或 pm25_dataset）")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="results 根目錄（需包含已訓練 artifacts）")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="CNN 抽 feature 用（extra_test 或缺 features_test.npy 時才用到）",
    )
    p.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help=f"要執行的方法列表（逗號分隔）可選：{ALL_METHODS}",
    )
    p.add_argument(
        "--cnn_tags",
        default=",".join(CNN_TAGS),
        help=f"CNN ablation tags（逗號分隔）預設：{CNN_TAGS}",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"未知 methods: {m}；可選：{ALL_METHODS}")

    cnn_tags = [t.strip() for t in args.cnn_tags.split(",") if t.strip()]

    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到輸入檔：{data_path}")

    # 先判斷是 full dataset 還是 extra
    if data_path.lower().endswith(".npz"):
        # 先偷看 keys
        d = np.load(data_path, allow_pickle=False)
        keys = set(d.files)
        if set(["X_train", "y_train", "X_test", "y_test"]).issubset(keys):
            run_report_mode(
                dataset_npz=data_path,
                out_dir=args.out_dir,
                device=args.device,
                methods=methods,
                cnn_tags=cnn_tags,
            )
            return

    # extra mode
    X_extra, info = read_np_like(data_path)
    if X_extra.ndim == 2:
        # 有些人會存成 (N, D) flatten，這裡嘗試猜回 (N,H,W) 會很危險
        # 直接拒絕，讓使用者修正檔案格式
        raise ValueError(
            f"extra_test 的 X 形狀不對（預期 3D: (N,H,W)，收到 {X_extra.shape}）。\n"
            "請確認 extra_test.npz / .npy 內的 X 是原始 157×103 的網格。"
        )

    run_extra_mode(
        X_extra=X_extra.astype(np.float32, copy=False),
        data_info=info,
        out_dir=args.out_dir,
        device=args.device,
        methods=methods,
        cnn_tags=cnn_tags,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
