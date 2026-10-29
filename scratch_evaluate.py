import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
import joblib

# Ensure src on path
sys.path.insert(0, 'src')
from models_multimodal import SklearnHbRegressor

# 1. Evaluate Nail & Skin & Fusion from patient_test_predictions.csv
df = pd.read_csv('results/patient_test_predictions.csv')
y_test = df['Actual_Hb_g_dL'].values

# 2. Evaluate Conjunctiva (Model 3)
conj_path = os.path.join("data", "cached_features", "features_conjunctiva.npz")
if os.path.isfile(conj_path) and os.path.isfile("checkpoints/model3_conjunctiva.joblib"):
    data_conj = np.load(conj_path, allow_pickle=True)
    X_conj = data_conj["features"]
    y_conj = data_conj["hemoglobin"]
    splits_conj = data_conj["splits"]
    test_mask_conj = (splits_conj == "test")
    X_conj_test = X_conj[test_mask_conj]
    y_conj_test = y_conj[test_mask_conj]
    reg_m3 = joblib.load("checkpoints/model3_conjunctiva.joblib")
    preds_m3 = reg_m3.predict(X_conj_test)
else:
    preds_m3 = None
    y_conj_test = None

models_dict = {
    'Model 1 (Fingernail ROI)': (y_test, df['Pred_M1_Nail_g_dL'].values),
    'Model 2 (Skin ROI)': (y_test, df['Pred_M2_Skin_g_dL'].values),
    'Model 4 (Nail+Skin Fusion)': (y_test, df['Pred_M4_Fusion_g_dL'].values),
}
if preds_m3 is not None:
    models_dict['Model 3 (Conjunctiva ROI)'] = (y_conj_test, preds_m3)

print("=" * 80)
print("EVALUATION RESULTS FOR PROJECT MODELS")
print("=" * 80)

summary_records = []

for name, (y_true, y_pred) in models_dict.items():
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    corr, pval = stats.pearsonr(y_true, y_pred)
    mean_err = np.mean(y_pred - y_true)  # bias
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Diagnostic classification: Anemia cutoff (< 12.0 g/dL as per WHO women / standard threshold)
    y_bin = (y_true < 12.0).astype(int)
    pred_bin = (y_pred < 12.0).astype(int)
    acc = accuracy_score(y_bin, pred_bin)
    sens = recall_score(y_bin, pred_bin, zero_division=0)
    spec = recall_score(1 - y_bin, 1 - pred_bin, zero_division=0)
    ppv = precision_score(y_bin, pred_bin, zero_division=0)
    npv = precision_score(1 - y_bin, 1 - pred_bin, zero_division=0)
    
    # Negative prediction for AUC since lower Hb means higher chance of anemia
    try:
        auc = roc_auc_score(y_bin, -y_pred)
    except:
        auc = np.nan

    cm = confusion_matrix(y_bin, pred_bin)
    
    # 95% Bland-Altman limits of agreement
    diff = y_pred - y_true
    loa_upper = mean_err + 1.96 * np.std(diff)
    loa_lower = mean_err - 1.96 * np.std(diff)

    print(f"\n--- {name} (N={len(y_true)}) ---")
    print(f"  MAE: {mae:.3f} g/dL | RMSE: {rmse:.3f} g/dL | R²: {r2:.4f} | Pearson r: {corr:.4f} (p={pval:.4e})")
    print(f"  Mean Error (Bias): {mean_err:.3f} g/dL | MAPE: {mape:.2f}%")
    print(f"  Bland-Altman 95% LoA: [{loa_lower:.2f}, {loa_upper:.2f}] g/dL")
    print(f"  Classification (<12.0 g/dL): Acc={acc*100:.1f}%, Sens={sens*100:.1f}%, Spec={spec*100:.1f}%, PPV={ppv*100:.1f}%, NPV={npv*100:.1f}%, AUC={auc:.3f}")
    print(f"  Confusion Matrix (TN, FP / FN, TP): TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    summary_records.append({
        'Model': name,
        'N': len(y_true),
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'R2': round(r2, 4),
        'Pearson_r': round(corr, 4),
        'Mean_Bias': round(mean_err, 3),
        'MAPE': round(mape, 2),
        'LoA_Lower': round(loa_lower, 2),
        'LoA_Upper': round(loa_upper, 2),
        'Accuracy': round(acc, 3),
        'Sensitivity': round(sens, 3),
        'Specificity': round(spec, 3),
        'PPV': round(ppv, 3),
        'NPV': round(npv, 3),
        'AUC': round(auc, 3)
    })

res_df = pd.DataFrame(summary_records)
res_df.to_csv('results/evaluation_with_diagnostic_metrics.csv', index=False)
print("\nSaved results/evaluation_with_diagnostic_metrics.csv")
