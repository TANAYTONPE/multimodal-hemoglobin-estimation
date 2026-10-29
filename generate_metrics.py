import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, 'src')

# ── Load test predictions ──────────────────────────────────────────────────────

df = pd.read_csv('results/patient_test_predictions.csv')
y = df['Actual_Hb_g_dL'].values
p1 = df['Pred_M1_Nail_g_dL'].values
p2 = df['Pred_M2_Skin_g_dL'].values
p4 = df['Pred_M4_Fusion_g_dL'].values

def compute_metrics(y_true, y_pred, label):
    n = len(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res/ss_tot
    corr, pval = stats.pearsonr(y_true, y_pred)
    print(f"{label}: n={n} | MAE={mae:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | r={corr:.4f} (p={pval:.4f})")
    return {'label': label, 'n': n, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Corr': corr}

print("=== COMPUTED METRICS FROM TEST SET ===")
m1 = compute_metrics(y, p1, 'Model 1 (Fingernail)')
m2 = compute_metrics(y, p2, 'Model 2 (Skin)')
m4 = compute_metrics(y, p4, 'Model 4 (Nail+Skin Fusion)')

# Model 3 from model_comparison.csv

m3 = {'label': 'Model 3 (Conjunctiva)', 'n': 31, 'MAE': 1.392, 'MSE': 2.873, 'RMSE': 1.695, 'R2': 0.4877, 'Corr': None}
m3_corr = np.sqrt(max(0, m3['R2']))
m3['Corr'] = m3_corr
print(f"Model 3 (Conjunctiva) [from CSV]: MAE={m3['MAE']:.3f} | MSE={m3['MSE']:.3f} | RMSE={m3['RMSE']:.3f} | R2={m3['R2']:.4f} | r~{m3_corr:.4f}")

# Color Baselines from CSV

b_nail = {'label': 'Baseline-Nail (Color)', 'n': 38, 'MAE': 1.511, 'MSE': 4.141, 'RMSE': 2.035, 'R2': 0.4353, 'Corr': np.sqrt(0.4353)}
b_skin = {'label': 'Baseline-Skin (Color)', 'n': 38, 'MAE': 1.667, 'MSE': 5.578, 'RMSE': 2.362, 'R2': 0.2395, 'Corr': np.sqrt(0.2395)}

print(f"\nBaseline Nail (Color-only): MAE={b_nail['MAE']:.3f} | MSE={b_nail['MSE']:.3f} | R²={b_nail['R2']:.4f} | r≈{b_nail['Corr']:.4f}")
print(f"Baseline Skin (Color-only): MAE={b_skin['MAE']:.3f} | MSE={b_skin['MSE']:.3f} | R²={b_skin['R2']:.4f} | r≈{b_skin['Corr']:.4f}")

all_models = [b_nail, b_skin, m1, m2, m3, m4]
labels = [m['label'] for m in all_models]
maes   = [m['MAE'] for m in all_models]
mses   = [m['MSE'] for m in all_models]
rmses  = [m['RMSE'] for m in all_models]
r2s    = [m['R2'] for m in all_models]
corrs  = [m['Corr'] for m in all_models]

# Colors

colors = ['#94a3b8', '#64748b', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

# ── Figure 1: MAE Bar Chart ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
bars = ax.barh(labels, maes, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('MAE (g/dL)', fontsize=12)
ax.set_title('Model Comparison — Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
ax.axvline(1.0, color='green', linestyle='--', alpha=0.6, label='Clinically useful (<1.0 g/dL)')
ax.axvline(1.5, color='orange', linestyle='--', alpha=0.6, label='Acceptable (<1.5 g/dL)')
for bar, val in zip(bars, maes):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax.set_xlim(0, max(maes)*1.18)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/comparison_mae.png', dpi=150)
plt.close()

# ── Figure 2: MSE Bar Chart ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
bars = ax.barh(labels, mses, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('MSE (g/dL)²', fontsize=12)
ax.set_title('Model Comparison — Mean Squared Error (MSE)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, mses):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax.set_xlim(0, max(mses)*1.18)
plt.tight_layout()
plt.savefig('results/comparison_mse.png', dpi=150)
plt.close()

# ── Figure 3: R² Bar Chart ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
bars = ax.barh(labels, r2s, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('R² Score', fontsize=12)
ax.set_title('Model Comparison — R² Score (Coefficient of Determination)', fontsize=13, fontweight='bold')
ax.axvline(0.5, color='green', linestyle='--', alpha=0.6, label='Good (R²≥0.5)')
ax.axvline(0.3, color='orange', linestyle='--', alpha=0.6, label='Moderate (R²≥0.3)')
for bar, val in zip(bars, r2s):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.set_xlim(0, max(r2s)*1.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/comparison_r2.png', dpi=150)
plt.close()

# ── Figure 4: Pearson Correlation Bar Chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
bars = ax.barh(labels, corrs, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel("Pearson Correlation (r)", fontsize=12)
ax.set_title('Model Comparison — Pearson Correlation (Predicted vs. Actual Hb)', fontsize=13, fontweight='bold')
ax.axvline(0.7, color='green', linestyle='--', alpha=0.6, label='Strong (r≥0.7)')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.6, label='Moderate (r≥0.5)')
for bar, val in zip(bars, corrs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.set_xlim(0, 1.0)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/comparison_correlation.png', dpi=150)
plt.close()

# ── Figure 5: Grouped bar - all metrics side by side ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
fig.suptitle('HemoLens Model Performance Dashboard — All Metrics', fontsize=14, fontweight='bold', y=1.01)

x = np.arange(len(labels))
metric_data = [
    (axes[0,0], maes, 'MAE (g/dL)', 'Mean Absolute Error'),
    (axes[0,1], mses, 'MSE (g/dL²)', 'Mean Squared Error'),
    (axes[1,0], r2s, 'R² Score', 'Coefficient of Determination'),
    (axes[1,1], corrs, 'Pearson r', 'Pearson Correlation'),
]

short_labels = ['Base-Nail', 'Base-Skin', 'M1-Nail', 'M2-Skin', 'M3-Conj', 'M4-Fusion']

for ax, vals, ylabel, title in metric_data:
    bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/comparison_all_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 6: Dataset Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
fig.suptitle('HemoLens Dataset — Hemoglobin Distribution & Splits', fontsize=13, fontweight='bold')

df_ns = pd.read_csv('data/metadata.csv')
df_conj = pd.read_csv('data/conjunctiva_manifest.csv')
split_colors = {'train': '#3b82f6', 'val': '#f59e0b', 'test': '#10b981'}

# Pie chart: Nail/Skin splits
splits_ns = df_ns['split'].value_counts()
axes[0].pie(splits_ns.values, labels=splits_ns.index, colors=[split_colors.get(k, 'gray') for k in splits_ns.index],
            autopct='%1.0f%%', startangle=90)
axes[0].set_title('Nail/Skin Dataset Splits\n(N=250 patients)', fontsize=11)

# Pie chart: Conjunctiva splits
splits_c = df_conj['split'].value_counts()
axes[1].pie(splits_c.values, labels=splits_c.index, colors=[split_colors.get(k, 'gray') for k in splits_c.index],
            autopct='%1.0f%%', startangle=90)
axes[1].set_title('Conjunctiva Dataset Splits\n(N=201 patients)', fontsize=11)

# Hb distribution histogram
axes[2].hist(df_ns['hemoglobin'], bins=20, alpha=0.7, color='#3b82f6', label='Nail/Skin (N=250)', edgecolor='white')
axes[2].hist(df_conj['hemoglobin'], bins=15, alpha=0.7, color='#f59e0b', label='Conjunctiva (N=201)', edgecolor='white')
axes[2].axvline(12.0, color='red', linestyle='--', alpha=0.7, label='Anemia threshold (12.0 g/dL)')
axes[2].set_xlabel('Hemoglobin (g/dL)', fontsize=11)
axes[2].set_ylabel('Count', fontsize=11)
axes[2].set_title('Hemoglobin Distribution\n(Both Datasets)', fontsize=11)
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('results/dataset_distribution.png', dpi=150)
plt.close()

print("\n=== ALL FIGURES SAVED ===")
print("results/comparison_mae.png")
print("results/comparison_mse.png")
print("results/comparison_r2.png")
print("results/comparison_correlation.png")
print("results/comparison_all_metrics.png")
print("results/dataset_distribution.png")

# Print final summary table

print("\n=== FINAL METRICS SUMMARY TABLE ===")
print(f"{'Model':<30} {'n':>5} {'MAE':>7} {'MSE':>7} {'RMSE':>7} {'R²':>7} {'Corr':>7}")
print("-"*70)
for m in all_models:
    print(f"{m['label']:<30} {m['n']:>5} {m['MAE']:>7.3f} {m['MSE']:>7.3f} {m['RMSE']:>7.3f} {m['R2']:>7.4f} {m['Corr']:>7.4f}")
