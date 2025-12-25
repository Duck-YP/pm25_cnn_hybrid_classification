# -*- coding: utf-8 -*-
from __future__ import annotations

"""
inspect_pm25_npz_professor.py

用途：
  針對 pm25_dataset.npz 做「完整資料資訊調查」，輸出一份詳細的報告檔（預設檔名 report.txt）。
  報告內容包含：
    1) Executive Summary（重點摘要：資料規格、是否完整、是否有 NaN/Inf、train/test 分佈差異）
    2) Dataset Schema（keys、shape、dtype、大小）
    3) X 統計（min/max/mean/std/percentiles） 
    4) y 統計（label 範圍、每月分佈）
    5) Leakage Check（train/test 是否有完全相同樣本：全量 SHA1 指紋比對）
    6) Sample-level Diagnostics（每張圖的均值/標準差分佈、極端樣本 Top-K）

重要聲明：
  - 本程式只做「讀取 + 統計 + 檢查」，不會對 X_test 做任何 fit/訓練/調參行為。

使用方式：
  python inspect_pm25_data.py --data pm25_dataset.npz  (可選後綴： --out report.txt)
"""

import argparse
import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# 小工具：格式化
# =============================================================================
def bytes_human(n: int) -> str:
    """把 bytes 轉成可讀（KiB/MiB/GiB）。
    
    Args:
        n (int): bytes 數量
    
    Returns:
        str: 可讀字串
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """計算檔案 SHA256（讓報告可追溯）。
    
    Args:
        path (str): 檔案路徑
        chunk_size (int): 每次讀取的區塊大小（預設 1MB）

    Returns:
        str: SHA256 哈希值（十六進位字串）
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def array_sha1(a: np.ndarray) -> str:
    """
    對單一樣本做 SHA1 指紋。
    完全相同（dtype/shape/內容一致）才會得到相同 hash。

    Args:
        a (np.ndarray): 輸入陣列
    
    Returns:
        str: SHA1 哈希值（十六進位字串）
    """
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def now_utc_iso() -> str:
    """回傳 UTC 現在時間（ISO 格式）。
    
    Returns:
        str: ISO 格式時間字串
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_percentiles(x: np.ndarray, pcts: List[int], *, sample_limit: int, seed: int) -> Tuple[np.ndarray, bool]:
    """
    取得分位數；若資料太大則抽樣避免爆記憶體。
    
    Args:
        x (np.ndarray): 輸入一維陣列
        pcts (List[int]): 要計算的分位數列表（0~100）
        sample_limit (int): 抽樣上限（元素數）
        seed (int): 抽樣用 random seed

    Returns:
        Tuple[np.ndarray, bool]: 分位數值陣列、是否有用抽樣
    """
    if x.size > sample_limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=sample_limit, replace=False)
        xs = x[idx]
        return np.percentile(xs, pcts), True
    return np.percentile(x, pcts), False


# =============================================================================
# 統計整理：X / y
# =============================================================================
@dataclass
class XStats:
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    nan_count: int
    inf_count: int
    finite_ratio: float
    x_min: float
    x_max: float
    mean: float
    std: float
    percentiles: Dict[int, float]
    used_sampling: bool
    sample_limit: int


def summarize_x(
    X: np.ndarray,
    *,
    pct_list: List[int] = [1, 10, 50, 90, 99],
    pct_sample_limit: int = 5_000_000,
    seed: int = 42,
) -> XStats:
    """輸出 X 的統計資訊（含 NaN/Inf、min/max、均值/標準差、分位數）。
    
    Args:
        X (np.ndarray): 輸入陣列
        pct_list (List[int]): 要計算的分位數列表（0~100）
        pct_sample_limit (int): 分位數估計抽樣上限（元素數）
        seed (int): 抽樣用 random seed

    Returns:
        XStats: 統計結果
    """
    Xf = X.astype(np.float32, copy=False)

    nan_count = int(np.isnan(Xf).sum())
    inf_count = int(np.isinf(Xf).sum())

    finite_mask = np.isfinite(Xf)
    finite_cnt = int(finite_mask.sum())
    total_cnt = int(Xf.size)
    finite_ratio = float(finite_cnt / total_cnt) if total_cnt > 0 else 0.0

    if finite_cnt == 0:
        # 極端異常：全部非 finite
        return XStats(
            shape=tuple(Xf.shape),
            dtype=str(Xf.dtype),
            nbytes=int(Xf.nbytes),
            nan_count=nan_count,
            inf_count=inf_count,
            finite_ratio=finite_ratio,
            x_min=float("nan"),
            x_max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            percentiles={p: float("nan") for p in pct_list},
            used_sampling=False,
            sample_limit=pct_sample_limit,
        )

    Xv = Xf[finite_mask]
    x_min = float(np.min(Xv))
    x_max = float(np.max(Xv))
    mean = float(np.mean(Xv))
    std = float(np.std(Xv))

    pct_vals, used_sampling = safe_percentiles(
        Xv, pct_list, sample_limit=pct_sample_limit, seed=seed
    )
    percentiles = {int(p): float(v) for p, v in zip(pct_list, pct_vals.tolist())}

    return XStats(
        shape=tuple(Xf.shape),
        dtype=str(Xf.dtype),
        nbytes=int(Xf.nbytes),
        nan_count=nan_count,
        inf_count=inf_count,
        finite_ratio=finite_ratio,
        x_min=x_min,
        x_max=x_max,
        mean=mean,
        std=std,
        percentiles=percentiles,
        used_sampling=used_sampling,
        sample_limit=pct_sample_limit,
    )


@dataclass
class YStats:
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    y_min: int
    y_max: int
    unique_count: int
    distribution: List[Tuple[int, int, float]]  # (label, count, ratio)
    label_style_note: str


def summarize_y(y: np.ndarray) -> YStats:
    """輸出 y 的統計資訊（類別分佈、範圍）。
    
    Args:
        y (np.ndarray): 輸入陣列

    Returns:
        YStats: 統計結果
    """
    yy = y.astype(np.int64, copy=False)
    y_min = int(np.min(yy))
    y_max = int(np.max(yy))

    uniq, cnt = np.unique(yy, return_counts=True)
    total = int(yy.size)
    dist = [(int(u), int(c), float(c / total)) for u, c in zip(uniq, cnt)]

    # 純提示：不做自動轉換，避免誤導
    if y_min >= 1 and y_max <= 12:
        note = "label 看起來是 1~12（月份）。實作分類時常轉成 0~11（月份-1）。"
    elif y_min >= 0 and y_max <= 11:
        note = "label 看起來已是 0~11（月份）。"
    else:
        note = "WARNING：label 範圍不像月份（0~11 或 1~12），請確認資料是否正確。"

    return YStats(
        shape=tuple(yy.shape),
        dtype=str(yy.dtype),
        nbytes=int(yy.nbytes),
        y_min=y_min,
        y_max=y_max,
        unique_count=int(len(uniq)),
        distribution=dist,
        label_style_note=note,
    )


# =============================================================================
# Leakage Check：全量 SHA1（你這份資料很小，直接全量做最乾淨）
# =============================================================================
@dataclass
class LeakageResult:
    train_n: int
    test_n: int
    exact_dup_count: int
    exact_dup_train_indices: List[int]
    exact_dup_test_indices: List[int]


def leakage_check_full(X_train: np.ndarray, X_test: np.ndarray) -> LeakageResult:
    """
    全量檢查 train/test 是否有「完全相同樣本」。

    Args:
        X_train (np.ndarray): 訓練特徵陣列
        X_test (np.ndarray): 測試特徵陣列

    Returns:
        LeakageResult: 重複數量與對應索引（若有）。
    """
    train_hash_to_idx: Dict[str, int] = {}
    for i in range(X_train.shape[0]):
        h = array_sha1(X_train[i])
        # 若 train 自己內部有重複，我們保留第一個索引即可（這裡是 train vs test 檢查）
        if h not in train_hash_to_idx:
            train_hash_to_idx[h] = i

    dup_train_idx: List[int] = []
    dup_test_idx: List[int] = []
    for j in range(X_test.shape[0]):
        h = array_sha1(X_test[j])
        if h in train_hash_to_idx:
            dup_train_idx.append(train_hash_to_idx[h])
            dup_test_idx.append(j)

    return LeakageResult(
        train_n=int(X_train.shape[0]),
        test_n=int(X_test.shape[0]),
        exact_dup_count=int(len(dup_train_idx)),
        exact_dup_train_indices=dup_train_idx,
        exact_dup_test_indices=dup_test_idx,
    )


# =============================================================================
# Sample-level diagnostics（每張圖的 mean/std，列出極端樣本）
# =============================================================================
@dataclass
class SampleDiag:
    mean_min: float
    mean_max: float
    std_min: float
    std_max: float
    topk_high_mean: List[Tuple[int, float]]
    topk_low_mean: List[Tuple[int, float]]
    topk_high_std: List[Tuple[int, float]]
    topk_low_std: List[Tuple[int, float]]


def sample_level_diagnostics(X: np.ndarray, *, topk: int = 5) -> SampleDiag:
    """
    對每個樣本計算 mean/std，找出極端樣本 Top-K。
    只做描述統計，不做任何訓練或 fitting。

    Args:
        X (np.ndarray): 輸入陣列
        topk (int): 要列出的極端樣本數量

    Returns:
        SampleDiag: 統計結果
    """
    Xf = X.astype(np.float32, copy=False)
    # 以 (H,W) 為例：對每張圖取平均/標準差
    means = Xf.reshape(Xf.shape[0], -1).mean(axis=1)
    stds = Xf.reshape(Xf.shape[0], -1).std(axis=1)

    # 取排序索引
    high_mean_idx = np.argsort(-means)[:topk]
    low_mean_idx = np.argsort(means)[:topk]
    high_std_idx = np.argsort(-stds)[:topk]
    low_std_idx = np.argsort(stds)[:topk]

    return SampleDiag(
        mean_min=float(means.min()),
        mean_max=float(means.max()),
        std_min=float(stds.min()),
        std_max=float(stds.max()),
        topk_high_mean=[(int(i), float(means[i])) for i in high_mean_idx],
        topk_low_mean=[(int(i), float(means[i])) for i in low_mean_idx],
        topk_high_std=[(int(i), float(stds[i])) for i in high_std_idx],
        topk_low_std=[(int(i), float(stds[i])) for i in low_std_idx],
    )


# =============================================================================
# 報告組裝
# =============================================================================
def format_distribution_table(dist: List[Tuple[int, int, float]]) -> List[str]:
    """把 label distribution 排成好讀的表。
    
    Args:
        dist (List[Tuple[int, int, float]]): 分佈列表（label, count, ratio）

    Returns:
        List[str]: 表格列字串列表
    """
    lines = []
    lines.append("label | count | ratio")
    lines.append("----- | ----- | -----")
    for label, c, r in dist:
        lines.append(f"{label:>5d} | {c:>5d} | {r:>5.4f}")
    return lines


def main() -> int:
    """
    主程式入口。

    Returns:
        int: 回傳碼（0=成功，非0=失敗）
    """
    parser = argparse.ArgumentParser(description="PM2.5 NPZ Dataset Investigation (Professor-ready report)")
    parser.add_argument("--data", type=str, required=True, help="pm25_dataset.npz 路徑")
    parser.add_argument("--out", type=str, default="report.txt", help="輸出報告檔名（txt）")
    parser.add_argument("--pct_sample_limit", type=int, default=5_000_000, help="分位數估計抽樣上限（元素數）")
    parser.add_argument("--seed", type=int, default=42, help="抽樣用 random seed（只影響分位數抽樣，不影響資料）")
    parser.add_argument("--topk", type=int, default=5, help="極端樣本 Top-K 顯示數量")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    if not data_path.exists():
        print(f"[ERROR] 找不到檔案：{data_path}")
        return 1

    # 讀取 npz（安全起見不允許 pickle）
    try:
        npz = np.load(str(data_path), allow_pickle=False)
    except Exception as e:
        print(f"[ERROR] 無法讀取 npz：{e}")
        return 1

    required = ["X_train", "y_train", "X_test", "y_test"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        print(f"[ERROR] 缺少 keys：{missing}（需要 {required}）")
        # 還是輸出一份最基本的報告，方便交代
        with out_path.open("w", encoding="utf-8") as f:
            f.write("PM2.5 Dataset NPZ Investigation Report\n")
            f.write(f"generated_at_utc: {now_utc_iso()}\n")
            f.write(f"data_path: {data_path}\n")
            f.write(f"file_size: {bytes_human(data_path.stat().st_size)}\n")
            f.write(f"keys: {', '.join(npz.files)}\n")
            f.write(f"\n[ERROR] missing keys: {missing}\n")
        print(f"[DONE] 已輸出：{out_path}（但 keys 不完整）")
        return 2

    # 取出陣列
    X_train = npz["X_train"]
    y_train = npz["y_train"]
    X_test = npz["X_test"]
    y_test = npz["y_test"]

    # 基本一致性檢查
    basic_warnings: List[str] = []
    if X_train.shape[0] != y_train.shape[0]:
        basic_warnings.append(f"WARNING: X_train 筆數({X_train.shape[0]}) != y_train 筆數({y_train.shape[0]})")
    if X_test.shape[0] != y_test.shape[0]:
        basic_warnings.append(f"WARNING: X_test 筆數({X_test.shape[0]}) != y_test 筆數({y_test.shape[0]})")
    if X_train.shape[1:] != X_test.shape[1:]:
        basic_warnings.append(f"WARNING: train/test 特徵形狀不同：train={X_train.shape[1:]}, test={X_test.shape[1:]}")

    # 統計
    xtr = summarize_x(X_train, pct_sample_limit=args.pct_sample_limit, seed=args.seed)
    ytr = summarize_y(y_train)
    xte = summarize_x(X_test, pct_sample_limit=args.pct_sample_limit, seed=args.seed)
    yte = summarize_y(y_test)

    # 全量 leakage 檢查
    leak = leakage_check_full(X_train, X_test)

    # sample-level diagnostics
    diag_tr = sample_level_diagnostics(X_train, topk=args.topk)
    diag_te = sample_level_diagnostics(X_test, topk=args.topk)

    # 檔案可追溯資訊
    fsize = int(data_path.stat().st_size)
    fsha = file_sha256(data_path)

    # train/test shift 摘要（用最直觀的數字，不做花俏推論）
    shift_lines = [
        f"Train mean/std: {xtr.mean:.6f} / {xtr.std:.6f}",
        f"Test  mean/std: {xte.mean:.6f} / {xte.std:.6f}",
        f"Train max: {xtr.x_max:.6f} | Test max: {xte.x_max:.6f}",
        f"Train P90/P99: {xtr.percentiles[90]:.6f} / {xtr.percentiles[99]:.6f}",
        f"Test  P90/P99: {xte.percentiles[90]:.6f} / {xte.percentiles[99]:.6f}",
    ]

    # =============================================================================
    # 組報告（教授可讀版）
    # =============================================================================
    lines: List[str] = []
    lines.append("PM2.5 Dataset NPZ Investigation Report (Professor-ready)")
    lines.append(f"generated_at_utc: {now_utc_iso()}")
    lines.append("")

    lines.append("=== Executive Summary ===")
    lines.append(f"- data_path: {data_path}")
    lines.append(f"- file_size: {bytes_human(fsize)}")
    lines.append(f"- file_sha256: {fsha}")
    lines.append(f"- keys: {', '.join(npz.files)}")
    lines.append(f"- X_train: shape={X_train.shape}, dtype={X_train.dtype} | y_train: shape={y_train.shape}, dtype={y_train.dtype}")
    lines.append(f"- X_test : shape={X_test.shape}, dtype={X_test.dtype}  | y_test : shape={y_test.shape}, dtype={y_test.dtype}")
    lines.append(f"- NaN/Inf (train): {xtr.nan_count}/{xtr.inf_count} | NaN/Inf (test): {xte.nan_count}/{xte.inf_count}")
    lines.append(f"- Label range (train/test): {ytr.y_min}..{ytr.y_max} / {yte.y_min}..{yte.y_max} (unique={ytr.unique_count}/{yte.unique_count})")
    lines.append(f"- Leakage check (exact duplicates, full SHA1): {leak.exact_dup_count} matches (train_n={leak.train_n}, test_n={leak.test_n})")
    lines.append("- Train vs Test distribution shift (quick view):")
    for s in shift_lines:
        lines.append(f"  - {s}")
    if basic_warnings:
        lines.append("- Basic warnings:")
        for w in basic_warnings:
            lines.append(f"  - {w}")
    lines.append("")

    lines.append("=== Environment (for reproducibility) ===")
    lines.append(f"- python: {sys.version.split()[0]}")
    lines.append(f"- numpy: {np.__version__}")
    lines.append(f"- platform: {platform.platform()}")
    lines.append("")

    lines.append("=== Dataset Schema ===")
    lines.append(f"- X_train: shape={xtr.shape}, dtype={xtr.dtype}, nbytes={bytes_human(xtr.nbytes)}")
    lines.append(f"- y_train: shape={ytr.shape}, dtype={ytr.dtype}, nbytes={bytes_human(ytr.nbytes)}")
    lines.append(f"- X_test : shape={xte.shape}, dtype={xte.dtype}, nbytes={bytes_human(xte.nbytes)}")
    lines.append(f"- y_test : shape={yte.shape}, dtype={yte.dtype}, nbytes={bytes_human(yte.nbytes)}")
    lines.append("")

    lines.append("=== X Statistics (Train) ===")
    lines.append(f"- finite_ratio: {xtr.finite_ratio:.6f} | NaN={xtr.nan_count}, Inf={xtr.inf_count}")
    lines.append(f"- min/max: {xtr.x_min:.6f} / {xtr.x_max:.6f}")
    lines.append(f"- mean/std: {xtr.mean:.6f} / {xtr.std:.6f}")
    if xtr.used_sampling:
        lines.append(f"- percentiles: (sampled {xtr.sample_limit} elements, seed={args.seed})")
    else:
        lines.append("- percentiles: (full data)")
    lines.append("  " + " | ".join([f"P{p}={xtr.percentiles[p]:.6f}" for p in [1, 10, 50, 90, 99]]))
    lines.append("")

    lines.append("=== X Statistics (Test) ===")
    lines.append(f"- finite_ratio: {xte.finite_ratio:.6f} | NaN={xte.nan_count}, Inf={xte.inf_count}")
    lines.append(f"- min/max: {xte.x_min:.6f} / {xte.x_max:.6f}")
    lines.append(f"- mean/std: {xte.mean:.6f} / {xte.std:.6f}")
    if xte.used_sampling:
        lines.append(f"- percentiles: (sampled {xte.sample_limit} elements, seed={args.seed})")
    else:
        lines.append("- percentiles: (full data)")
    lines.append("  " + " | ".join([f"P{p}={xte.percentiles[p]:.6f}" for p in [1, 10, 50, 90, 99]]))
    lines.append("")

    lines.append("=== y Statistics (Train) ===")
    lines.append(f"- label min/max: {ytr.y_min} / {ytr.y_max} | unique={ytr.unique_count}")
    lines.append(f"- note: {ytr.label_style_note}")
    lines.extend(format_distribution_table(ytr.distribution))
    lines.append("")

    lines.append("=== y Statistics (Test) ===")
    lines.append(f"- label min/max: {yte.y_min} / {yte.y_max} | unique={yte.unique_count}")
    lines.append(f"- note: {yte.label_style_note}")
    lines.extend(format_distribution_table(yte.distribution))
    lines.append("")

    lines.append("=== Leakage Check (Train vs Test, FULL SHA1) ===")
    lines.append(f"- exact duplicate matches: {leak.exact_dup_count}")
    if leak.exact_dup_count > 0:
        lines.append("- matched indices (train_idx -> test_idx):")
        for tr_i, te_i in zip(leak.exact_dup_train_indices, leak.exact_dup_test_indices):
            lines.append(f"  - {tr_i} -> {te_i}")
        lines.append("WARNING: 發現完全相同樣本，可能存在資料洩漏風險。")
    else:
        lines.append("OK: 未發現 train/test 完全相同樣本（以全量 SHA1 指紋比對）。")
    lines.append("")

    lines.append("=== Sample-level Diagnostics (Train) ===")
    lines.append(f"- per-sample mean range: {diag_tr.mean_min:.6f} .. {diag_tr.mean_max:.6f}")
    lines.append(f"- per-sample std  range: {diag_tr.std_min:.6f} .. {diag_tr.std_max:.6f}")
    lines.append(f"- Top-{args.topk} highest mean samples (idx, mean): {diag_tr.topk_high_mean}")
    lines.append(f"- Top-{args.topk} lowest  mean samples (idx, mean): {diag_tr.topk_low_mean}")
    lines.append(f"- Top-{args.topk} highest std  samples (idx, std):  {diag_tr.topk_high_std}")
    lines.append(f"- Top-{args.topk} lowest  std  samples (idx, std):  {diag_tr.topk_low_std}")
    lines.append("")

    lines.append("=== Sample-level Diagnostics (Test) ===")
    lines.append(f"- per-sample mean range: {diag_te.mean_min:.6f} .. {diag_te.mean_max:.6f}")
    lines.append(f"- per-sample std  range: {diag_te.std_min:.6f} .. {diag_te.std_max:.6f}")
    lines.append(f"- Top-{args.topk} highest mean samples (idx, mean): {diag_te.topk_high_mean}")
    lines.append(f"- Top-{args.topk} lowest  mean samples (idx, mean): {diag_te.topk_low_mean}")
    lines.append(f"- Top-{args.topk} highest std  samples (idx, std):  {diag_te.topk_high_std}")
    lines.append(f"- Top-{args.topk} lowest  std  samples (idx, std):  {diag_te.topk_low_std}")
    lines.append("")

    # 寫檔
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 終端機輸出摘要
    print("[DONE] 完整資料集診斷報告：", str(out_path))
    print("---- Quick Summary (for screenshot) ----")
    print(f"train: N={X_train.shape[0]}, shape={X_train.shape[1:]} | test: N={X_test.shape[0]}, shape={X_test.shape[1:]}")
    print(f"NaN/Inf train={xtr.nan_count}/{xtr.inf_count} | test={xte.nan_count}/{xte.inf_count}")
    print(f"label range train={ytr.y_min}..{ytr.y_max} | test={yte.y_min}..{yte.y_max}")
    print(f"leak(exact dup, full SHA1)={leak.exact_dup_count}")
    print("---------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
