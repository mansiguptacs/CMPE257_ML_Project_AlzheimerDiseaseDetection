import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Ablation summary chart ─────────────────────────────────────────
ablation = pd.read_csv("results/ablation/ablation_results.csv")
scenario_avg = ablation.groupby('scenario')['accuracy'].mean().reset_index()
scenario_avg = scenario_avg.sort_values('accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(11, 6))
y_pos = range(len(scenario_avg))
baseline_acc = scenario_avg[scenario_avg['scenario'] == 'Baseline (All Features)']['accuracy'].values[0]
colors_abl = ['#e74c3c' if a < baseline_acc - 0.05 else '#f39c12' if a < baseline_acc - 0.01
              else '#2ecc71' for a in scenario_avg['accuracy']]
bars = ax.barh(y_pos, scenario_avg['accuracy'], color=colors_abl, edgecolor='white', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(scenario_avg['scenario'])
ax.set_xlabel("Average Accuracy (across RF, XGB, LogReg)")
ax.set_title("Ablation Study: Impact of Feature Removal on Accuracy", fontweight='bold', fontsize=13)
ax.set_xlim(0.68, 0.90)
for bar, val in zip(bars, scenario_avg['accuracy']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va='center', fontweight='bold', fontsize=10)
ax.axvline(baseline_acc, color='#2ecc71', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_acc:.1%}')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(OUT / "ablation_summary.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ ablation_summary.png")

# ── Feature importance comparison (top models) ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fi_files = {
    'Random Forest': 'models/phase1_oasis1/random_forest_feature_importance.csv',
    'XGBoost': 'models/phase1_oasis1/xgboost_feature_importance.csv',
    'Gradient Boosting': 'models/phase1_oasis1/gradient_boosting_feature_importance.csv',
}
for ax, (name, path) in zip(axes, fi_files.items()):
    try:
        fi = pd.read_csv(path)
        fi = fi.sort_values('importance', ascending=True)
        colors_fi = ['#e74c3c' if f == 'MMSE' else '#3498db' for f in fi['feature']]
        ax.barh(fi['feature'], fi['importance'], color=colors_fi, edgecolor='white')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel("Feature Importance")
        # Highlight MMSE %
        mmse_imp = fi[fi['feature'] == 'MMSE']['importance'].values
        if len(mmse_imp) > 0:
            ax.text(0.95, 0.05, f"MMSE: {mmse_imp[0]*100:.1f}%",
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    color='#e74c3c', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha='center')

plt.suptitle("Feature Importance: MMSE Dominance Across Models", fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "feature_importance_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ feature_importance_comparison.png")

print(f"\n✅ All plots saved to {OUT}/")
