# -*- coding: utf-8 -*-
from __future__ import annotations

"""train_cnn_feature.py

核心流程（CNN Feature Extractor）：
  [Stage 1] 只用 train_split fit 正規化（mean/std + 可選 clip_hi），並在 train/val 上訓練 CNN
            - 用 val_acc 做 early-stopping 挑 best_epoch 與 best 權重（cnn_best.pt）
  [Stage 2] 拿 best_epoch 後，用「整個 X_train」重新 fit 正規化，並從頭重訓 CNN 恰好 best_epoch 回合
            - 重訓過程不看 val、不做 early-stopping（避免 train 內資訊交叉疑慮）
            - 輸出 refit 權重（cnn_refit.pt）與 normalizer.npz
  [Stage 3] 用 cnn_refit + normalizer 匯出 features（train/test）
            - features_train.npy / features_test.npy（維度 feature_dim=128）
            - meta.json（可重現的設定與 best_epoch）

規範（你這份作業常見的「扣分點」這裡直接鎖死）：
  1) 不得用 test 資料做任何「fit」：包含 mean/std、clip 門檻、early-stopping、調參。
     - 本程式：Stage1 的 normalizer 只用 train_split fit；Stage2 只用 X_train fit。
  2) 「權重固定為預設」：不使用 class weight / sample weight（CrossEntropyLoss 的 weight=None）。
     - 目的：避免你不小心把權重當成可調參數，造成不公平比較或流程混亂。
  3) train_affine（資料增強）只作用於 train（train_split 或 full train），val/test 不做 augmentation。

輸入：
  - (必) --data        : pm25_dataset.npz（需包含 X_train, y_train, X_test, y_test）
  - (必) --out_dir     : 輸出根目錄（例如 ./results）
  - (必) --tag         : 本次實驗標籤（會建立 out_dir/cnn_feature/<tag>/）
  - (選) --preprocess  : none / hist_eq / log1p
  - (選) --clip_p      : log1p 的截尾百分位（例如 99.9）。0 表示不截尾
  - (選) --train_affine: 0/1，是否在 train 進行 affine augmentation（RandomAffine）
  - (選) --seed        : 固定隨機種子（預設 42）
  - (選) --device      : auto / cpu / mps / cuda（預設 auto）

輸出（方便重現）：
  - out_dir/cnn_feature/<tag>/cnn_best.pt        : Stage1（train/val）val_acc 最佳權重
  - out_dir/cnn_feature/<tag>/cnn_refit.pt       : Stage2（全 train 重訓）權重（最終拿來抽特徵）
  - out_dir/cnn_feature/<tag>/normalizer.npz     : mean/std + preprocess/clip 設定（只用 train fit）
  - out_dir/cnn_feature/<tag>/features_train.npy : (N_train, feature_dim)
  - out_dir/cnn_feature/<tag>/features_test.npy  : (N_test,  feature_dim)
  - out_dir/cnn_feature/<tag>/meta.json          : 參數、best_epoch、best_val_acc、環境摘要

執行範例：
# 1) 不用 hist_eq、不用 affine（baseline 特徵）
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline

# 2) ablation：加上 histogram equalization（對 train/test 都做 preprocessing；不需 fit）
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline_hist_eq --preprocess hist_eq

# 3) ablation：hist_eq + affine（affine 只在 train 做 augmentation）
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline_hist_eq_affine --preprocess hist_eq --train_affine 1
"""

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---- PyTorch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---- sklearn（切 train/val 用）----
from sklearn.model_selection import train_test_split


# =============================================================================
# 小工具：可重現
# =============================================================================
def set_global_seed(seed: int) -> None:
    """固定隨機種子，讓切分與訓練更可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 盡量 deterministic（可能稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(device: str) -> torch.device:
    """選擇運算裝置。"""
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        # Apple Silicon
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Preprocess：hist_eq / log1p + clip
# =============================================================================
def hist_equalize_per_sample(x: np.ndarray, bins: int = 256) -> np.ndarray:
    """簡單 histogram equalization（每張圖各自做；不需要 fit）。

    注意：這裡把每張圖的值線性映射到 [0, 1] 做等化，再映射回 [0, 1]。
    對於 PM2.5 這種非影像資料，這是「強行拉直分布」的手段，可能有利也可能傷害。
    """
    x = x.astype(np.float32, copy=False)
    flat = x.reshape(-1)
    vmin = float(flat.min())
    vmax = float(flat.max())
    if vmax <= vmin + 1e-12:
        # 幾乎常數圖，直接回傳 0
        return np.zeros_like(x, dtype=np.float32)

    # 映射到 [0, 1]
    z = (flat - vmin) / (vmax - vmin)

    # 直方圖 + CDF
    hist, bin_edges = np.histogram(z, bins=bins, range=(0.0, 1.0), density=False)
    cdf = hist.cumsum().astype(np.float64)
    cdf = cdf / (cdf[-1] + 1e-12)

    # 依照 bin 對每個值做映射（用 searchsorted 找 bin）
    idx = np.searchsorted(bin_edges[1:], z, side="right")
    idx = np.clip(idx, 0, bins - 1)
    z_eq = cdf[idx].astype(np.float32)

    return z_eq.reshape(x.shape)


def apply_preprocess(
    X: np.ndarray,
    preprocess: str,
    *,
    clip_hi: Optional[float] = None,
) -> np.ndarray:
    """對整個 X 套用 preprocess（不做 fit）。"""
    preprocess = preprocess.lower()
    if preprocess in ("none", ""):
        out = X.astype(np.float32, copy=False)
    elif preprocess == "hist_eq":
        out = np.empty_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            out[i] = hist_equalize_per_sample(X[i])
    elif preprocess == "log1p":
        out = np.log1p(X.astype(np.float32, copy=False))
        if clip_hi is not None:
            out = np.minimum(out, np.float32(clip_hi))
    else:
        raise ValueError(f"Unknown preprocess: {preprocess}")
    return out


def fit_log1p_clip_hi(X_train_like: np.ndarray, clip_p: float) -> Optional[float]:
    """只用 train 來估 clip_hi（嚴禁用 test）。clip_p<=0 代表不截尾。"""
    if clip_p is None or clip_p <= 0:
        return None
    flat = X_train_like.reshape(-1).astype(np.float32, copy=False)
    # 這裡直接 exact percentile（你資料量 ~ 590 萬，OK）
    return float(np.percentile(flat, clip_p))


def fit_normalizer(X: np.ndarray) -> Tuple[float, float]:
    """只用 train fit mean/std（嚴禁用 test）。"""
    mean = float(X.mean())
    std = float(X.std())
    std = max(std, 1e-6)  # 防呆
    return mean, std


def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (X - np.float32(mean)) / np.float32(std)


# =============================================================================
# Dataset：train 可選 affine augmentation（只在 train）
# =============================================================================
class PM25GridDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        train_affine: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.train_affine = bool(train_affine)
        self.device = device

        # affine 參數（對齊你 log 顯示的預設）
        self.affine_deg = 0.0
        self.affine_translate = 0.02  # 以影像寬高比例為單位
        self.affine_scale = (1.0, 1.0)

        # torchvision 有就用；沒有就自動關掉 augmentation（避免你環境炸掉）
        self._tv_ok = False
        try:
            import torchvision.transforms as T  # noqa: F401
            self._tv_ok = True
        except Exception:
            self._tv_ok = False

        if self.train_affine and (not self._tv_ok):
            print("[WARN] torchvision 不可用：train_affine 會被忽略（避免程式中斷）。")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _maybe_affine(self, x_chw: torch.Tensor) -> torch.Tensor:
        """只在 train 使用的 affine augmentation。"""
        if (not self.train_affine) or (not self._tv_ok):
            return x_chw
        # torchvision RandomAffine 期望 (C,H,W) 的 tensor
        import torchvision.transforms as T

        tfm = T.RandomAffine(
            degrees=self.affine_deg,
            translate=(self.affine_translate, self.affine_translate),
            scale=self.affine_scale,
        )
        return tfm(x_chw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (H,W)
        y = self.y[idx]
        # 轉成 torch： (1,H,W)
        x_t = torch.from_numpy(x).unsqueeze(0)

        # augmentation（只在 train）
        x_t = self._maybe_affine(x_t)

        return x_t, torch.tensor(y, dtype=torch.long)


# =============================================================================
# CNN：簡單、穩定、可抽 feature（128-d）
# =============================================================================
class CNNFeatureNet(nn.Module):
    def __init__(self, num_classes: int = 12, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat = nn.Linear(128, feature_dim)
        self.cls = nn.Linear(feature_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        z = self.feat(h)
        z = F.relu(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        logits = self.cls(z)
        return logits


# =============================================================================
# 訓練 / 評估
# =============================================================================
@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(total, 1))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)  # ✅ 權重固定預設：loss_fn 內 weight=None
        loss.backward()
        optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

    return float(total_loss / max(total_n, 1))


# =============================================================================
# 設定
# =============================================================================
@dataclass
class TrainConfig:
    # 這些超參數維持你 log 的預設（不要亂動）
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    epochs_max: int = 120
    patience: int = 25
    feature_dim: int = 128
    val_ratio: float = 0.2


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="pm25_dataset.npz")
    parser.add_argument("--out_dir", type=str, required=True, help="output root dir")
    parser.add_argument("--tag", type=str, required=True, help="experiment tag")
    parser.add_argument("--preprocess", type=str, default="none", choices=["none", "hist_eq", "log1p"])
    parser.add_argument("--clip_p", type=float, default=0.0, help="log1p clip percentile; 0 means no clip")
    parser.add_argument("--train_affine", type=int, default=0, help="0/1; affine augmentation on train only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    cfg = TrainConfig()
    set_global_seed(args.seed)
    device = pick_device(args.device)

    out_dir = os.path.join(args.out_dir, "cnn_feature", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Load NPZ
    # -------------------------
    npz = np.load(args.data)
    X_train = npz["X_train"].astype(np.float32, copy=False)
    y_train = npz["y_train"].astype(np.int64, copy=False)
    X_test = npz["X_test"].astype(np.float32, copy=False)
    y_test = npz["y_test"].astype(np.int64, copy=False)

    # 你的資料標籤是 1..12，這裡統一轉 0..11（跟你 log 對齊）
    if y_train.min() == 1:
        y_train = y_train - 1
    if y_test.min() == 1:
        y_test = y_test - 1

    print("========== [DATA] ==========")
    print(f"[INFO] X_train={X_train.shape} | y_train={y_train.shape} | y_range=[{y_train.min()},{y_train.max()}]")
    print(f"[INFO] X_test ={X_test.shape} | y_test ={y_test.shape} | y_range=[{y_test.min()},{y_test.max()}]")
    print("============================")

    # -------------------------
    # Stage 1: train/val split
    # -------------------------
    print("========== [Stage 1] train/val 早停挑 best_epoch ==========")
    idx_all = np.arange(len(X_train))
    idx_tr, idx_va = train_test_split(
        idx_all,
        test_size=cfg.val_ratio,
        random_state=args.seed,
        stratify=y_train,
    )
    X_tr_raw, y_tr = X_train[idx_tr], y_train[idx_tr]
    X_va_raw, y_va = X_train[idx_va], y_train[idx_va]

    # 1) preprocess：注意 clip_hi 只能用 train_split 估（嚴禁看 val/test）
    preprocess = args.preprocess.lower()
    clip_hi_stage1: Optional[float] = None
    if preprocess == "log1p":
        X_tr_log = np.log1p(X_tr_raw)
        clip_hi_stage1 = fit_log1p_clip_hi(X_tr_log, args.clip_p)
        X_tr_p = apply_preprocess(X_tr_raw, preprocess, clip_hi=clip_hi_stage1)
        X_va_p = apply_preprocess(X_va_raw, preprocess, clip_hi=clip_hi_stage1)
    else:
        X_tr_p = apply_preprocess(X_tr_raw, preprocess)
        X_va_p = apply_preprocess(X_va_raw, preprocess)

    # 2) fit normalizer：只用 train_split
    mean1, std1 = fit_normalizer(X_tr_p)
    X_tr_n = normalize(X_tr_p, mean1, std1)
    X_va_n = normalize(X_va_p, mean1, std1)

    train_affine = bool(int(args.train_affine) == 1)
    print(f"[INFO] train_split={len(idx_tr)} | val_split={len(idx_va)} | device={device.type}")
    if preprocess == "log1p":
        print(f"[INFO] preprocess_done=True | clip_p={float(args.clip_p):.1f} | clip_hi={clip_hi_stage1}")
    else:
        print(f"[INFO] preprocess_done={(preprocess != 'none')} | clip_p={float(args.clip_p):.1f} | clip_hi=None")
    print(f"[INFO] affine(train_only)={train_affine} | deg=0.0 | trans=0.02 | scale=[1.0,1.0]")
    print(
        f"[INFO] lr={cfg.lr} | weight_decay={cfg.weight_decay} | batch_size={cfg.batch_size} | "
        f"epochs_max={cfg.epochs_max} | patience={cfg.patience} | feature_dim={cfg.feature_dim}"
    )
    print("===========================================================")

    ds_tr = PM25GridDataset(X_tr_n, y_tr, train_affine=train_affine, device=device)
    ds_va = PM25GridDataset(X_va_n, y_va, train_affine=False, device=device)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = CNNFeatureNet(num_classes=12, feature_dim=cfg.feature_dim).to(device)

    # ✅「權重固定預設」：CrossEntropyLoss(weight=None)
    loss_fn = nn.CrossEntropyLoss(weight=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    best_epoch = 0
    best_state: Dict[str, Any] = {}
    bad = 0

    for epoch in range(1, cfg.epochs_max + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, dl_tr, optimizer, device, loss_fn=loss_fn)
        va_acc = eval_acc(model, dl_va, device)

        # val_loss（為了 log 一致性）
        model.eval()
        with torch.no_grad():
            va_loss_sum = 0.0
            va_n = 0
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                bs = int(y.size(0))
                va_loss_sum += float(loss.item()) * bs
                va_n += bs
            va_loss = float(va_loss_sum / max(va_n, 1))

        improved = va_acc > best_val_acc + 1e-12
        mark = "★" if improved else " "
        dt = time.time() - t0

        if improved:
            best_val_acc = va_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} {mark} | {dt:.1f}s")

        if bad >= cfg.patience:
            print(f"[EARLY STOP] 超過 patience={cfg.patience} 未提升，停止於 epoch={epoch}，best_epoch={best_epoch} (val_acc={best_val_acc:.4f})")
            break

    # save cnn_best.pt
    best_path = os.path.join(out_dir, "cnn_best.pt")
    torch.save(
        {
            "state_dict": best_state,
            "feature_dim": cfg.feature_dim,
            "num_classes": 12,
        },
        best_path,
    )
    print(f"[DONE] best_epoch={best_epoch} | best_val_acc={best_val_acc:.4f} | saved: {best_path}")

    # -------------------------
    # Stage 2: refit on full train for best_epoch
    # -------------------------
    print("\n========== [Stage 2] 用整個 X_train 重訓 (refit) ==========")

    # 1) preprocess：clip_hi 只能用「全 train」估（仍然嚴禁看 test）
    clip_hi_stage2: Optional[float] = None
    if preprocess == "log1p":
        X_full_log = np.log1p(X_train)
        clip_hi_stage2 = fit_log1p_clip_hi(X_full_log, args.clip_p)
        X_full_p = apply_preprocess(X_train, preprocess, clip_hi=clip_hi_stage2)
    else:
        X_full_p = apply_preprocess(X_train, preprocess)

    # 2) fit normalizer：只用 full train
    mean2, std2 = fit_normalizer(X_full_p)
    X_full_n = normalize(X_full_p, mean2, std2)

    print(f"[INFO] full_train={len(X_train)} | best_epoch={best_epoch} | device={device.type}")
    print("===========================================================")

    ds_full = PM25GridDataset(X_full_n, y_train, train_affine=train_affine, device=device)
    dl_full = DataLoader(ds_full, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model2 = CNNFeatureNet(num_classes=12, feature_dim=cfg.feature_dim).to(device)
    loss_fn2 = nn.CrossEntropyLoss(weight=None)  # ✅ 固定預設
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for e in range(1, best_epoch + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model2, dl_full, optimizer2, device, loss_fn=loss_fn2)
        dt = time.time() - t0
        print(f"[Refit {e:03d}/{best_epoch:03d}] train_loss={tr_loss:.4f} | {dt:.1f}s")

    refit_path = os.path.join(out_dir, "cnn_refit.pt")
    torch.save(
        {
            "state_dict": model2.state_dict(),
            "feature_dim": cfg.feature_dim,
            "num_classes": 12,
        },
        refit_path,
    )

    norm_path = os.path.join(out_dir, "normalizer.npz")
    np.savez(
        norm_path,
        mean=np.float32(mean2),
        std=np.float32(std2),
        preprocess=preprocess,
        clip_p=np.float32(args.clip_p),
        clip_hi=np.float32(clip_hi_stage2 if clip_hi_stage2 is not None else np.nan),
        # 明確記錄：本程式固定「不使用 class weight」
        class_weight_used=np.int32(0),
    )

    print(f"[DONE] saved: {refit_path}")
    print(f"[DONE] saved: {norm_path}")

    # -------------------------
    # Stage 3: export features for train/test (use refit + full-train normalizer)
    # -------------------------
    print("\n========== [Stage 3] 匯出 features（train/test） ==========")

    # preprocess test：嚴禁用 test 重新估 clip_hi，所以沿用 stage2 的 clip_hi_stage2
    if preprocess == "log1p":
        X_test_p = apply_preprocess(X_test, preprocess, clip_hi=clip_hi_stage2)
    else:
        X_test_p = apply_preprocess(X_test, preprocess)

    X_test_n = normalize(X_test_p, mean2, std2)

    # 特徵抽取（不做 augmentation）
    ds_train_feat = PM25GridDataset(X_full_n, y_train, train_affine=False, device=device)
    ds_test_feat = PM25GridDataset(X_test_n, y_test, train_affine=False, device=device)

    dl_train_feat = DataLoader(ds_train_feat, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_test_feat = DataLoader(ds_test_feat, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model2.eval()
    feats_train = []
    feats_test = []

    with torch.no_grad():
        for x, _ in dl_train_feat:
            x = x.to(device)
            z = model2.encode(x)
            feats_train.append(z.detach().cpu().numpy())
        for x, _ in dl_test_feat:
            x = x.to(device)
            z = model2.encode(x)
            feats_test.append(z.detach().cpu().numpy())

    feats_train = np.concatenate(feats_train, axis=0).astype(np.float32, copy=False)
    feats_test = np.concatenate(feats_test, axis=0).astype(np.float32, copy=False)

    feat_tr_path = os.path.join(out_dir, "features_train.npy")
    feat_te_path = os.path.join(out_dir, "features_test.npy")
    np.save(feat_tr_path, feats_train)
    np.save(feat_te_path, feats_test)

    meta = {
        "data": os.path.abspath(args.data),
        "out_dir": os.path.abspath(out_dir),
        "tag": args.tag,
        "seed": int(args.seed),
        "device": str(device),
        "cfg": asdict(cfg),
        "preprocess": preprocess,
        "clip_p": float(args.clip_p),
        "clip_hi": (float(clip_hi_stage2) if clip_hi_stage2 is not None else None),
        "train_affine": bool(train_affine),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "feature_dim": int(cfg.feature_dim),
        "class_weight_used": False,  # ✅ 固定預設
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "features_train": list(feats_train.shape),
            "features_test": list(feats_test.shape),
        },
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] features_train: {feats_train.shape} -> {feat_tr_path}")
    print(f"[DONE] features_test : {feats_test.shape} -> {feat_te_path}")
    print(f"[DONE] meta.json -> {meta_path}")
    print("\n[OK] train_cnn_feature.py 完成：你接下來的 (CNN+SVM/PCA/AdaBoost) 只要讀這裡的 features 即可。")


if __name__ == "__main__":
    main()
