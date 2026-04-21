#!/usr/bin/env python3
"""Generate EDA and preprocessing visualizations for the progress report."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load raw data ──────────────────────────────────────────────────────
df = pd.read_excel("data/oasis_cross-sectional-5708aa0a98d82080.xlsx")
print(f"Loaded {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ── 1. CDR class distribution (raw) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw CDR
cdr_counts = df['CDR'].value_counts(dropna=False).sort_index()
colors_raw = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad', '#95a5a6']
labels = [f"CDR {v}" if not pd.isna(v) else "CDR NaN" for v in cdr_counts.index]
axes[0].bar(labels, cdr_counts.values, color=colors_raw[:len(cdr_counts)])
axes[0].set_title("Raw CDR Distribution", fontweight='bold')
axes[0].set_ylabel("Count")
for i, v in enumerate(cdr_counts.values):
    axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')

# Binary target
df_binary = df.copy()
df_binary['target'] = df_binary['CDR'].fillna(0).apply(lambda x: 1 if x > 0 else 0)
target_counts = df_binary['target'].value_counts().sort_index()
colors_bin = ['#2ecc71', '#e74c3c']
bar_labels = ['Healthy (0)', 'Dementia (1)']
axes[1].bar(bar_labels, target_counts.values, color=colors_bin)
axes[1].set_title("Binary Target Distribution", fontweight='bold')
axes[1].set_ylabel("Count")
for i, v in enumerate(target_counts.values):
    axes[1].text(i, v + 3, f"{v} ({v/len(df_binary)*100:.1f}%)", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ class_distribution.png")

# ── 2. Missing values heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
colors_miss = ['#e74c3c' if v > 100 else '#f39c12' if v > 10 else '#3498db' for v in missing.values]
bars = ax.barh(missing.index, missing.values, color=colors_miss)
ax.set_xlabel("Number of Missing Values")
ax.set_title("Missing Values by Feature", fontweight='bold')
for bar, val in zip(bars, missing.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f"{val} ({val/len(df)*100:.1f}%)", va='center', fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "missing_values.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ missing_values.png")

# ── 3. Feature distributions by CDR status ────────────────────────────
numeric_features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
df_plot = df_binary.copy()
df_plot['Status'] = df_plot['target'].map({0: 'Healthy', 1: 'Dementia'})

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, feat in enumerate(numeric_features):
    ax = axes[i]
    for status, color in [('Healthy', '#2ecc71'), ('Dementia', '#e74c3c')]:
        data = df_plot[df_plot['Status'] == status][feat].dropna()
        ax.hist(data, bins=20, alpha=0.6, color=color, label=status, edgecolor='white')
    ax.set_title(feat, fontweight='bold')
    ax.legend(fontsize=8)

axes[7].axis('off')
plt.suptitle("Feature Distributions: Healthy vs Dementia", fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "feature_distributions.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ feature_distributions.png")

# ── 4. Correlation heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = numeric_features + ['target']
corr = df_plot[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title("Feature Correlation Matrix", fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ correlation_heatmap.png")

# ── 5. MMSE vs CDR boxplot (the key finding) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# MMSE by CDR group
cdr_groups = df.dropna(subset=['CDR', 'MMSE'])
ax = axes[0]
cdr_vals = sorted(cdr_groups['CDR'].unique())
bp_data = [cdr_groups[cdr_groups['CDR'] == c]['MMSE'].values for c in cdr_vals]
bp = ax.boxplot(bp_data, labels=[f"CDR={c}" for c in cdr_vals], patch_artist=True)
colors_box = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_box[:len(cdr_vals)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("MMSE Score")
ax.set_title("MMSE Score by CDR Group", fontweight='bold')
ax.set_xlabel("Clinical Dementia Rating")

# nWBV by CDR group
ax = axes[1]
bp_data2 = [cdr_groups[cdr_groups['CDR'] == c]['nWBV'].dropna().values for c in cdr_vals]
bp2 = ax.boxplot(bp_data2, labels=[f"CDR={c}" for c in cdr_vals], patch_artist=True)
for patch, color in zip(bp2['boxes'], colors_box[:len(cdr_vals)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("nWBV (Normalized Whole Brain Volume)")
ax.set_title("Brain Volume by CDR Group", fontweight='bold')
ax.set_xlabel("Clinical Dementia Rating")

plt.tight_layout()
plt.savefig(OUT / "mmse_nwbv_vs_cdr.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ mmse_nwbv_vs_cdr.png")

# ── 6. Age distribution by dementia status ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for status, color in [('Healthy', '#2ecc71'), ('Dementia', '#e74c3c')]:
    data = df_plot[df_plot['Status'] == status]['Age'].dropna()
    ax.hist(data, bins=25, alpha=0.6, color=color, label=f"{status} (n={len(data)})", edgecolor='white')
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution by Dementia Status", fontweight='bold')
ax.legend()
ax.axvline(df_plot[df_plot['Status'] == 'Healthy']['Age'].median(), color='#2ecc71',
           linestyle='--', alpha=0.8, label='Healthy median')
ax.axvline(df_plot[df_plot['Status'] == 'Dementia']['Age'].median(), color='#e74c3c',
           linestyle='--', alpha=0.8, label='Dementia median')
plt.tight_layout()
plt.savefig(OUT / "age_distribution.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ age_distribution.png")

# ── 7. Model comparison bar chart ─────────────────────────────────────
results = pd.read_csv("models/phase1_oasis1/model_comparison.csv")
results = results.sort_values('accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(results))
colors_model = ['#e74c3c' if a < 0.83 else '#f39c12' if a < 0.87 else '#2ecc71'
                for a in results['accuracy']]
bars = ax.barh(y_pos, results['accuracy'], color=colors_model, edgecolor='white', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(results['model'].str.replace('_', ' ').str.title())
ax.set_xlabel("Accuracy")
ax.set_title("Phase 1 Baseline: Model Accuracy Comparison", fontweight='bold', fontsize=13)
ax.set_xlim(0.75, 0.92)
for bar, val in zip(bars, results['accuracy']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va='center', fontweight='bold', fontsize=10)
ax.axvline(0.881, color='#2ecc71', linestyle='--', alpha=0.5, label='Best: 88.1%')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(OUT / "model_accuracy_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ model_accuracy_comparison.png")

