# Results Summary

Best: **cnn_svm** (tag=baseline) | test_acc=0.3090 | macro_f1=0.3110

## Top-3 by test accuracy

1. cnn_svm (tag=baseline) | test_acc=0.3090
2. cnn_adaboost (tag=baseline) | test_acc=0.2303
3. cnn_pca (tag=baseline) | test_acc=0.2275

## Ablation: baseline vs hist_eq vs hist_eq_affine

- cnn_svm: baseline=0.308989 → hist_eq=0.202247 (Δ -0.106742) → hist_eq_affine=0.188202 (Δ -0.120787)
- cnn_pca: baseline=0.227528 → hist_eq=0.157303 (Δ -0.070225) → hist_eq_affine=0.171348 (Δ -0.056180)
- cnn_adaboost: baseline=0.230337 → hist_eq=0.109551 (Δ -0.120787) → hist_eq_affine=0.202247 (Δ -0.028090)

## Error analysis

- confusion_best.png：最佳方法混淆矩陣
- error_top_confusions.csv：最常見的月份混淆對
