## 0. 專案結構與主要檔案

你會用到的檔案（依「建議執行順序」排列）：

1. `inspect_pm25_data.py`：快速檢查資料 shape/label 分佈，產生 `report.txt`
2. `train_cnn_feature.py`：訓練 CNN feature extractor 並輸出 features（供 3 個 paper head 共用） :contentReference[oaicite:0]{index=0}
3. `run_paper_cnn_svm.py`：Paper-1 CNN + SVM（讀 features）  
4. `run_paper_cnn_pca.py`：Paper-2 CNN + PCA（讀 features）  
5. `run_paper_cnn_adaboost.py`：Paper-3 CNN + PCA + AdaBoost（讀 features） :contentReference[oaicite:1]{index=1}
6. `run_baseline_raw_svm.py`：Baseline Raw flatten + SVM  
7. `run_baseline_pca_knn.py`：Baseline Raw flatten + PCA + KNN :contentReference[oaicite:2]{index=2}
8. `infer_extra_test.py`：**單一入口**（extra_test 純推論 / pm25_dataset 報告整理） :contentReference[oaicite:3]{index=3}

資料與輸出：
- `pm25_dataset.npz`：作業資料（包含 X_train/y_train/X_test/y_test）
- `results/`：所有訓練 artifacts、summary、與整理後報告都在這

---

## 1. 環境安裝

### 1.1 建議 Python 版本
建議用 Python 3.10+（你的 requirements 版本較新）。

### 1.2 安裝套件
```bash
python -m venv venv
source venv/bin/activate           # macOS/Linux
# venv\Scripts\activate            # Windows PowerShell

pip install -r requirements.txt
````

requirements 內容請以專案內為準。 

---

## 2. 資料檢查（建議先做）

目的：確認資料 shape、label 範圍、訓練/測試筆數，並保存檢查結果（`report.txt`）。

```bash
python inspect_pm25_data.py
```

輸出：

* `report.txt`（資料分佈摘要、基本檢查資訊）

> 註：專案內部會把月份標籤常見的 **1~12 轉成 0~11** 來訓練/推論（輸出給老師時再轉回 1~12）。

---

## 3. 訓練與測試（建議執行順序）

### 3.1 Step A：先產生 CNN features（所有 paper head 共用）

你有 3 組 tag（也就是 ablation 設定）：

* `baseline`
* `baseline_hist_eq`
* `baseline_hist_eq_affine`

#### A-1) baseline

```bash
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline
```

#### A-2) ablation：histogram equalization

```bash
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline_hist_eq --preprocess hist_eq
```

#### A-3) ablation：hist_eq + affine（affine **只作用於 train**）

```bash
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline_hist_eq_affine --preprocess hist_eq --train_affine 1
```

輸出位置（每個 tag 一套）：

* `results/cnn_feature/<tag>/cnn_best.pt`：Stage1（train/val）val 最佳權重
* `results/cnn_feature/<tag>/cnn_refit.pt`：Stage2（全 train 重訓 best_epoch）最終權重
* `results/cnn_feature/<tag>/normalizer.npz`：train-only mean/std 與 preprocess 設定
* `results/cnn_feature/<tag>/features_train.npy` / `features_test.npy`：抽出的特徵（N×128）
* `results/cnn_feature/<tag>/meta.json`：可重現資訊（seed、best_epoch、環境等）

（以上流程與輸出定義請以 `train_cnn_feature.py` 為準） 

---

### 3.2 Step B：跑 3 個 paper 方法（讀取 features）

> 下列三個方法都**不會再訓練 CNN**，只會讀 `results/cnn_feature/<tag>/features_*.npy`。

#### B-1) Paper-1：CNN + SVM

（對每個 tag 跑一次）

```bash
python run_paper_cnn_svm.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline
python run_paper_cnn_svm.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq
python run_paper_cnn_svm.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq_affine
```

輸出（每個 tag 一套）：

* `results/cnn_svm/<tag>/cnn_svm_feat_scaler.joblib`
* `results/cnn_svm/<tag>/cnn_svm_svm.joblib`
* `results/cnn_svm/<tag>/summary.json`
* `results/cnn_svm/<tag>/meta.json`

#### B-2) Paper-2：CNN + PCA（PCA 空間分類器）

```bash
python run_paper_cnn_pca.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline
python run_paper_cnn_pca.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq
python run_paper_cnn_pca.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq_affine
```

輸出（每個 tag 一套）：

* `results/cnn_pca/<tag>/cnn_pca_feat_scaler.joblib`
* `results/cnn_pca/<tag>/cnn_pca_pca.joblib`
* `results/cnn_pca/<tag>/cnn_pca_centroid.joblib`
* `results/cnn_pca/<tag>/cv_table.csv`
* `results/cnn_pca/<tag>/summary.json`
* `results/cnn_pca/<tag>/meta.json`

#### B-3) Paper-3：CNN + PCA + AdaBoost

```bash
python run_paper_cnn_adaboost.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline
python run_paper_cnn_adaboost.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq
python run_paper_cnn_adaboost.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline_hist_eq_affine
```

輸出（每個 tag 一套）：

* `results/cnn_adaboost/<tag>/cnn_adaboost_feat_scaler.joblib`
* `results/cnn_adaboost/<tag>/cnn_adaboost_pca.joblib`
* `results/cnn_adaboost/<tag>/cnn_adaboost.joblib`
* `results/cnn_adaboost/<tag>/summary.json`
* `results/cnn_adaboost/<tag>/meta.json`

（paper-3 腳本的參數與防洩漏規則請以原始檔頭註解為準） 

---

### 3.3 Step C：跑 2 個 baseline（不用 CNN）

#### C-1) Baseline：Raw flatten + SVM

```bash
python run_baseline_raw_svm.py --data pm25_dataset.npz --out_dir ./results
```

輸出：

* `results/baseline_raw_svm/raw_svm_feat_scaler.joblib`
* `results/baseline_raw_svm/raw_svm_svm.joblib`
* `results/baseline_raw_svm/summary.json`
* `results/baseline_raw_svm/meta.json`

#### C-2) Baseline：Raw flatten + PCA + KNN

```bash
python run_baseline_pca_knn.py --data pm25_dataset.npz --out_dir ./results
```

輸出：

* `results/baseline_pca_knn/pca_knn_feat_scaler.joblib`
* `results/baseline_pca_knn/pca_knn_pca.joblib`
* `results/baseline_pca_knn/pca_knn_knn.joblib`
* `results/baseline_pca_knn/summary.json`
* `results/baseline_pca_knn/meta.json`

（baseline PCA+KNN 的 CV 邏輯與輸出定義請以腳本註解為準） 

---

## 4. 一鍵整理報告（pm25_dataset 模式）

當你已經把上面訓練都跑完後，用下面指令整理投影片需要的結果（Results table / Ablation / Confusion matrix / Error analysis）：

```bash
python infer_extra_test.py -data pm25_dataset.npz --out_dir ./results
```

這個模式會 **只讀你已經訓練好的 artifacts**，不會做任何訓練或 fit。 

輸出位置：

* `results/report/pm25_dataset/`

  * `results_table.csv`：所有方法 test_acc（含 tag）與 macro_f1
  * `ablation_hist_eq_affine.csv`：baseline vs hist_eq vs hist_eq_affine
  * `confusion_best.csv` / `confusion_best.png`：最佳方法混淆矩陣
  * `error_top_confusions.csv`：最常見的月份混淆 pair
  * `conclusion.json`：最佳方法與 top3、ablation 摘要 
  * `REPORT.md`：可直接貼投影片/報告的文字版摘要 

## 5. 額外測試資料推論（extra_test 模式：純推論、零訓練）
### 5.1 前置條件（一定要先有已訓練 artifacts）

### 5.2 執行方式
`extra_test` 檔案不一定要放在專案根目錄，`-data` 可以給相對/絕對路徑：

```bash
python infer_extra_test.py -data extra_test.npz --out_dir ./results
````

可選參數：

* `--device {auto,cpu,cuda,mps}`：只有在 **CNN 抽 feature** 時會用到（extra_test 一定會抽）
* `--methods`：逗號分隔，預設全跑：`cnn_svm,cnn_pca,cnn_adaboost,raw_svm,pca_knn`
* `--cnn_tags`：逗號分隔，預設：`baseline,baseline_hist_eq,baseline_hist_eq_affine`

### 5.3 會做什麼（實際行為）

* **Baseline 只跑一次**（它們沒有 tag 概念，輸出會用 `tag=default`）：

  * `raw_svm`
  * `pca_knn`
* **CNN 系列會對每個 tag 都跑一次**（三個 head 共用同一份「當下 tag」抽出來的 CNN features）：

  * `cnn_svm`（CNN feature → SVM）
  * `cnn_pca`（CNN feature → PCA → Nearest Centroid / estimator）
  * `cnn_adaboost`（CNN feature → PCA → AdaBoost）

> 注意：每個方法都有 try/except；若某方法缺 artifacts 會印 `[WARN]` 並跳過，但其他方法仍會繼續跑，所以「輸出可能不完整」是正常的。

### 5.4 輸出位置與檔案格式（重點：baseline 的 tag 會是 default）

輸出根目錄固定是：

* `results/infer_extra/`

各方法輸出資料夾：

* Baselines（固定 `default`）

  * `results/infer_extra/raw_svm/default/`
  * `results/infer_extra/pca_knn/default/`

* CNN methods（對每個 tag 各一份）

  * `results/infer_extra/cnn_svm/<tag>/`
  * `results/infer_extra/cnn_pca/<tag>/`
  * `results/infer_extra/cnn_adaboost/<tag>/`

每個資料夾都會包含：

* `pred_label0_11.npy`：模型輸出的類別（0~11）
* `pred_month1_12.npy`：轉回月份（1~12）
* `pred_month1_12.csv`：兩欄 `id,month`（最適合交給老師）
* `meta.json`：推論資訊摘要（mode/method/tag/note）

### 5.5 輸入資料格式限制（程式真的會拒絕）

* 程式會**直接拒絕** `X` 是 2D（例如 `(N,D)` flatten）的 extra_test：
  會報錯並要求你提供原始網格 `(N,H,W)`（作業通常是 `157×103`）。
* extra_test.npz 內的 X key 會自動嘗試從以下名稱抓第一個找到的：

  * `X_extra`, `X`, `data`, `images`, `x`, `X_test`

## 6. 常見問題（快速排雷）

### Q1：為什麼要先跑 `train_cnn_feature.py`？

因為 3 個 paper 方法（CNN+SVM / CNN+PCA / CNN+AdaBoost）共用同一個 CNN feature extractor。
後面三個 paper 腳本都只讀 features，不再訓練 CNN。 

### Q2：我只想重跑某個方法？

可以。只要它依賴的 artifacts 已存在即可：

* CNN 系列（cnn_svm/cnn_pca/cnn_adaboost）必須先有 `results/cnn_feature/<tag>/features_*.npy`
* Baseline 直接跑即可

### Q3：結果要去哪裡找？

* 單一方法的訓練 artifacts / summary：都在 `results/<method>/...`
* 整理後可用的報告：`results/report/pm25_dataset/`（建議直接用這個資料夾的輸出）

---

## 7. 建議的「最少可交付」訓練跑法

最簡單完整且可解釋的一套（不跑 ablation）：

```bash
# 1) CNN features（baseline）
python train_cnn_feature.py --data pm25_dataset.npz --out_dir ./results --tag baseline

# 2) 3 個 paper head（baseline）
python run_paper_cnn_svm.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline
python run_paper_cnn_pca.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline
python run_paper_cnn_adaboost.py --data pm25_dataset.npz --out_dir ./results --feat_tag baseline

# 3) baselines
python run_baseline_raw_svm.py --data pm25_dataset.npz --out_dir ./results
python run_baseline_pca_knn.py --data pm25_dataset.npz --out_dir ./results

# 4) 產出報告檔
python infer_extra_test.py -data pm25_dataset.npz --out_dir ./results
```

---

## 8. 參考：目前專案已產出的最佳結果（以 results/report 為準）

目前 `results/report/pm25_dataset/` 的摘要顯示：

* Best：`cnn_svm (tag=baseline)`，test_acc 約 0.309、macro_f1 約 0.311
  （詳細以 `conclusion.json` / `REPORT.md` / `results_table.csv` 為準）  

