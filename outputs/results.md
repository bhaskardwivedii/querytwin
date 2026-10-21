| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8367 | 0.7791 | 0.7782 | 0.7786 | 0.9148 | 782.5s |
| Random Forest | 0.7979 | 0.7312 | 0.7159 | 0.7235 | 0.8825 | 347.6s |
| XGBoost | 0.8317 | 0.7726 | 0.7710 | 0.7718 | 0.9133 | 268.1s |
| Ensemble (LR + XGBoost) * | 0.8510 | 0.8021 | 0.7918 | 0.7969 | 0.9295 | 1050.6s |
\* best on validation F1. Metrics are on the held-out test split at threshold 0.5.
