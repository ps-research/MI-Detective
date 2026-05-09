#!/usr/bin/env python3
"""
MI DETECTIVE — PRODUCTION FIGURES
===================================
Reads from real experiment JSON outputs. No dummy data.

8 figures, all 600 dpi PDF:
  Fig 1: Behavior Fingerprint Heatmap — top features × cases
  Fig 2: Behavior Layer Landscape — where each behavior's features live
  Fig 3: Verdict Table — the surprise: everything is Genuine
  Fig 4: Trigger vs Control Circuit Metrics
  Fig 5: Intervention Text Comparison — best 4 cases
  Fig 6: Cross-Behavior Feature Overlap — real Jaccard
  Fig 7: Evidence Summary — scores and match rates
  Fig 8: C11 Multi-Turn Escalation

Usage: python generate_figures.py
"""

import json
import os
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns

# ============================================================
# Config
# ============================================================

DATA = "/workspace/MI-Detective/outputs"
FIG = "/workspace/MI-Detective/figures/production"
os.makedirs(FIG, exist_ok=True)

DPI = 600
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

CASE_SHORT = {
    "C1_sycophantic_validation": "Sycophantic\nValidation",
    "C2_jailbreak_patterns": "Jailbreak\nPatterns",
    "C3_confident_hallucination": "Confident\nHallucination",
    "C4_emotional_manipulation": "Emotional\nManipulation",
    "C5_deceptive_capability": "Deceptive\nCapability",
    "C6_blackmail_coercion": "Blackmail &\nCoercion",
    "C7_hidden_goal_steering": "Hidden Goal\nSteering",
    "C8_deceptive_self_presentation": "Deceptive\nSelf-Pres.",
    "C9_emotional_dependency": "Emotional\nDependency",
    "C10_info_hazard_laundering": "Info Hazard\nLaundering",
    "C12_manufactured_credibility": "Manufactured\nCredibility",
}

CASE_TINY = {
    "C1_sycophantic_validation": "Sycophancy",
    "C2_jailbreak_patterns": "Jailbreak",
    "C3_confident_hallucination": "Hallucination",
    "C4_emotional_manipulation": "Emot. Manip.",
    "C5_deceptive_capability": "Deceptive Cap.",
    "C6_blackmail_coercion": "Blackmail",
    "C7_hidden_goal_steering": "Hidden Goals",
    "C8_deceptive_self_presentation": "Self-Pres.",
    "C9_emotional_dependency": "Emot. Depend.",
    "C10_info_hazard_laundering": "Info Hazard",
    "C12_manufactured_credibility": "Manuf. Cred.",
}

CASE_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#16a085", "#8e44ad",
]


def save(fig, name):
    pdf_path = os.path.join(FIG, f"{name}.pdf")
    png_path = os.path.join(FIG, f"{name}.png")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white", format="pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


def load(filename):
    with open(os.path.join(DATA, filename)) as f:
        return json.load(f)


# ============================================================
# Load all data
# ============================================================

print("Loading experiment data...")
fingerprints = load("exp1_fingerprints.json")
circuits = load("exp2_circuits.json")
interventions = load("exp3_interventions.json")
verdicts = load("exp4_verdicts.json")
escalation = load("exp_c11_escalation.json")

case_ids = list(fingerprints["cases"].keys())
print(f"  Fingerprints: {len(case_ids)} cases")
print(f"  Circuits: {len(circuits['circuits'])} cases")
print(f"  Interventions: {len(interventions['interventions'])} cases")
print(f"  Verdicts: {len(verdicts['verdicts'])} cases")
print(f"  Escalation: {len(escalation.get('escalation', []))} protocols")


# ============================================================
# Fig 1: Behavior Fingerprint Heatmap
# ============================================================

print("\nFig 1: Fingerprint Heatmap...")

# Collect top features across ALL cases, then build matrix
all_features = set()
case_feature_data = {}

for case_id in case_ids:
    case_data = fingerprints["cases"][case_id]
    top_feats = case_data.get("top_features", [])[:15]
    feat_dict = {}
    for f in top_feats:
        key = (f["layer"], f["feature_idx"])
        all_features.add(key)
        feat_dict[key] = f["mean_abs_diff"]
    case_feature_data[case_id] = feat_dict

# Sort features by total activation across cases
feat_totals = {}
for feat in all_features:
    total = sum(case_feature_data[c].get(feat, 0) for c in case_ids)
    feat_totals[feat] = total

sorted_feats = sorted(feat_totals.keys(), key=lambda f: feat_totals[f], reverse=True)[:30]

# Build matrix
matrix = np.zeros((len(sorted_feats), len(case_ids)))
for i, feat in enumerate(sorted_feats):
    for j, case_id in enumerate(case_ids):
        matrix[i, j] = case_feature_data[case_id].get(feat, 0)

feat_labels = [f"L{f[0]}/f{f[1]}" for f in sorted_feats]

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

ax.set_xticks(range(len(case_ids)))
ax.set_xticklabels([CASE_SHORT.get(c, c) for c in case_ids], fontsize=8, rotation=0)
ax.set_yticks(range(len(feat_labels)))
ax.set_yticklabels(feat_labels, fontsize=7, fontfamily="monospace")

# Highlight cells with activation > 300
for i in range(len(sorted_feats)):
    for j in range(len(case_ids)):
        val = matrix[i, j]
        if val > 300:
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white")

ax.grid(False)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("|Activation Differential|", fontsize=10)
ax.set_title("Behavior Fingerprints: Which CLT Features Characterize Each Behavior?\n"
             "Top 30 features ranked by total differential across all cases",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Mysterious Behavior", fontsize=11)
ax.set_ylabel("CLT Feature", fontsize=11)

fig.tight_layout()
save(fig, "fig1_fingerprint_heatmap")


# ============================================================
# Fig 2: Behavior Layer Landscape
# ============================================================

print("Fig 2: Layer Landscape...")

n_layers = 26
layer_matrix = np.zeros((len(case_ids), n_layers))

for ci, case_id in enumerate(case_ids):
    case_data = fingerprints["cases"][case_id]
    top_feats = case_data.get("top_features", [])[:30]

    layer_acts = defaultdict(list)
    for f in top_feats:
        layer_acts[f["layer"]].append(f["mean_abs_diff"])

    for l in range(n_layers):
        if l in layer_acts:
            layer_matrix[ci, l] = np.sum(layer_acts[l])

fig, ax = plt.subplots(figsize=(18, 7))
im = ax.imshow(layer_matrix, aspect="auto", cmap="inferno", interpolation="bilinear")

ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
ax.set_yticks(range(len(case_ids)))
ax.set_yticklabels([CASE_SHORT.get(c, c) for c in case_ids], fontsize=9)

for ci in range(len(case_ids)):
    row = layer_matrix[ci]
    if row.max() > 0:
        peak = np.argmax(row)
        marker_color = "white" if row[peak] > row.max() * 0.3 else "black"
        ax.plot(peak, ci, "*", color=marker_color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.text(peak + 0.5, ci, f"L{peak}", fontsize=7, va="center",
                fontweight="bold", color=marker_color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5))

ax.axvline(x=5.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
ax.axvline(x=13.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
ax.grid(False)

ax.set_title("Where Are Mysterious Behaviors Processed?\n"
             "Sum of Top-30 Feature |Differential| Per Layer",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Layer", fontsize=11)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Cumulative |Differential Activation|", fontsize=10)

fig.tight_layout()
save(fig, "fig2_layer_landscape")


# ============================================================
# Fig 3: Verdict Table
# ============================================================

print("Fig 3: Verdict Table...")

verdict_list = verdicts["verdicts"]

fig, ax = plt.subplots(figsize=(18, 6))
ax.axis("off")

col_labels = ["Behavior", "Expected\nVerdict", "Mechanistic\nVerdict", "Evidence\nScore", "Match"]
cell_text = []
cell_colors = []

verdict_bg = {
    "Genuine": "#e8f5e9",
    "Confused": "#fff3e0",
    "Emergent Artifact": "#fce4ec",
    "Unknown": "#f3e5f5",
    "Mixed": "#e3f2fd",
}

for v in verdict_list:
    match_symbol = "YES" if v["match"] else "NO"
    cell_text.append([
        v["case_name"],
        v["expected_verdict"],
        v["actual_verdict"],
        f"{v['evidence_score']}/{v['max_score']}",
        match_symbol,
    ])

    expected_bg = verdict_bg.get(v["expected_verdict"], "#ffffff")
    actual_bg = verdict_bg.get(v["actual_verdict"], "#ffffff")
    match_bg = "#e8f5e9" if v["match"] else "#fce4ec"
    cell_colors.append(["white", expected_bg, actual_bg, "white", match_bg])

table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                 loc="center", cellLoc="center")

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width(range(len(col_labels)))
table.scale(1, 2.2)

# Count matches
n_match = sum(1 for v in verdict_list if v["match"])
n_total = len(verdict_list)

ax.set_title(f"The Mechanistic Verdict Table: {n_match}/{n_total} Match Expected Classification\n"
             "Surprise: Most behaviors classified as Genuine — they have real, detectable circuits\n"
             "Even hallucination and hidden goal steering have active mechanistic signatures",
             fontsize=13, fontweight="bold", pad=25)

fig.tight_layout()
save(fig, "fig3_verdict_table")


# ============================================================
# Fig 4: Trigger vs Control Circuit Metrics
# ============================================================

print("Fig 4: Circuit Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics_to_plot = [
    ("n_ff_edges", "Feature-Feature Edges"),
    ("avg_path", "Avg Path Length"),
    ("fingerprint_overlap", "Fingerprint Overlap"),
]

for mi, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[mi]

    trigger_vals = []
    control_vals = []
    labels = []

    for case_id in case_ids:
        case_circuits = circuits["circuits"].get(case_id, {})
        t = case_circuits.get("trigger", {})
        c = case_circuits.get("control", {})

        if "error" not in t and "error" not in c:
            tv = t.get(metric, t.get("n_ff_edges", 0))
            cv = c.get(metric, c.get("n_ff_edges", 0))
            trigger_vals.append(tv)
            control_vals.append(cv)
            labels.append(CASE_TINY.get(case_id, case_id[:12]))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, trigger_vals, width, label="Trigger",
           color="#e74c3c", edgecolor="white", linewidth=0.8)
    ax.bar(x + width/2, control_vals, width, label="Control",
           color="#3498db", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.15)

fig.suptitle("Circuit Structure: Trigger vs Control Prompts\n"
             "Higher fingerprint overlap = circuit captures behavior-specific features",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig4_circuit_comparison")


# ============================================================
# Fig 5: Intervention Text Comparison (4 best cases)
# ============================================================

print("Fig 5: Intervention Text...")

# Pick 4 cases with most interesting ablation effects
best_cases = ["C1_sycophantic_validation", "C2_jailbreak_patterns",
              "C9_emotional_dependency", "C6_blackmail_coercion"]

# Filter to cases that exist in interventions
available_cases = [c for c in best_cases if c in interventions["interventions"]]
if len(available_cases) < 4:
    # Fill with whatever's available
    for c in case_ids:
        if c not in available_cases and c in interventions["interventions"]:
            available_cases.append(c)
        if len(available_cases) >= 4:
            break

fig, axes = plt.subplots(len(available_cases), 2, figsize=(16, 4 * len(available_cases)))
if len(available_cases) == 1:
    axes = axes.reshape(1, 2)

for i, case_id in enumerate(available_cases):
    case_data = interventions["interventions"].get(case_id, {})
    trigger = case_data.get("trigger", {})
    clean = trigger.get("clean", "N/A")[:200]
    ablated = trigger.get("ablated", "N/A")[:200]

    for j, (text, variant) in enumerate([(clean, "Clean"), (ablated, "Ablated")]):
        ax = axes[i, j]
        ax.axis("off")

        bg = "#fff8e1" if j == 0 else "#e8f5e9"
        border = "#f39c12" if j == 0 else "#27ae60"

        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                               facecolor=bg, edgecolor=border, linewidth=2,
                               transform=ax.transAxes)
        ax.add_patch(rect)

        ax.text(0.5, 0.97, variant, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold", color=border)

        # Wrap
        words = text.split()
        lines, current = [], []
        for w in words:
            current.append(w)
            if len(" ".join(current)) > 50:
                lines.append(" ".join(current))
                current = []
        if current:
            lines.append(" ".join(current))

        ax.text(0.05, 0.85, "\n".join(lines[:8]), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontfamily="serif", linespacing=1.4)

    # Row label
    case_label = CASE_TINY.get(case_id, case_id[:15])
    axes[i, 0].text(-0.02, 0.5, case_label, transform=axes[i, 0].transAxes,
                     ha="right", va="center", fontsize=9, fontweight="bold",
                     color=CASE_COLORS[i % len(CASE_COLORS)])

fig.suptitle("Intervention Validation: How Ablating Fingerprint Features Changes Behavior\n"
             "Clean = original response | Ablated = with behavior-specific features removed",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig5_intervention_text")


# ============================================================
# Fig 6: Cross-Behavior Feature Overlap
# ============================================================

print("Fig 6: Cross-Behavior Overlap...")

overlap_data = verdicts.get("cross_behavior_overlap", {})

n_cases = len(case_ids)
overlap_matrix = np.eye(n_cases)

case_to_idx = {c: i for i, c in enumerate(case_ids)}

for key, data in overlap_data.items():
    parts = key.split("_vs_")
    if len(parts) == 2:
        c1, c2 = parts
        if c1 in case_to_idx and c2 in case_to_idx:
            i, j = case_to_idx[c1], case_to_idx[c2]
            overlap_matrix[i, j] = data["jaccard"]
            overlap_matrix[j, i] = data["jaccard"]
    else:
        # Try matching by finding _vs_ in the full key
        for ci, cid1 in enumerate(case_ids):
            for cj, cid2 in enumerate(case_ids):
                check_key = f"{cid1}_vs_{cid2}"
                if check_key == key:
                    overlap_matrix[ci, cj] = data["jaccard"]
                    overlap_matrix[cj, ci] = data["jaccard"]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(overlap_matrix, dtype=bool)
np.fill_diagonal(mask, True)

im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0,
               vmax=max(0.25, overlap_matrix[~mask].max() if overlap_matrix[~mask].size > 0 else 0.1))

ax.set_xticks(range(n_cases))
ax.set_xticklabels([CASE_TINY.get(c, c[:10]) for c in case_ids],
                    fontsize=7, rotation=45, ha="right")
ax.set_yticks(range(n_cases))
ax.set_yticklabels([CASE_TINY.get(c, c[:10]) for c in case_ids], fontsize=7)

for i in range(n_cases):
    for j in range(n_cases):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
        else:
            val = overlap_matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.12 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, fontweight="bold", color=color)

ax.grid(False)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard Similarity")
ax.set_title("Cross-Behavior Feature Overlap (Jaccard Similarity of Top-20 Features)\n"
             "Higher overlap = behaviors share mechanistic roots",
             fontsize=12, fontweight="bold", pad=15)

fig.tight_layout()
save(fig, "fig6_cross_behavior_overlap")


# ============================================================
# Fig 7: Evidence Summary
# ============================================================

print("Fig 7: Evidence Summary...")

verdict_list = verdicts["verdicts"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Evidence scores
ax = axes[0]
names = [CASE_TINY.get(v["case_id"], v["case_name"][:12]) for v in verdict_list]
scores = [v["evidence_score"] for v in verdict_list]
max_scores = [v["max_score"] for v in verdict_list]
colors_ev = [CASE_COLORS[i % len(CASE_COLORS)] for i in range(len(verdict_list))]

bars = ax.barh(range(len(names)), scores, color=colors_ev,
               edgecolor="white", linewidth=0.5, height=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Evidence Score", fontsize=10)
ax.set_title("Evidence Strength", fontsize=11, fontweight="bold")
ax.set_xlim(0, max(max_scores) + 0.5)
ax.grid(True, axis="x", alpha=0.15)

for bar, score, mx in zip(bars, scores, max_scores):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{score}/{mx}", va="center", fontsize=8, fontweight="bold")

# Middle: Expected vs Actual verdict distribution
ax = axes[1]
expected_counts = defaultdict(int)
actual_counts = defaultdict(int)
for v in verdict_list:
    expected_counts[v["expected_verdict"]] += 1
    actual_counts[v["actual_verdict"]] += 1

all_verdicts = sorted(set(list(expected_counts.keys()) + list(actual_counts.keys())))
x = np.arange(len(all_verdicts))
width = 0.35

exp_vals = [expected_counts.get(v, 0) for v in all_verdicts]
act_vals = [actual_counts.get(v, 0) for v in all_verdicts]

ax.bar(x - width/2, exp_vals, width, label="Expected", color="#3498db",
       edgecolor="white", linewidth=0.8)
ax.bar(x + width/2, act_vals, width, label="Actual", color="#e74c3c",
       edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(all_verdicts, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Expected vs Actual\nVerdict Distribution", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.15)

# Right: Match rate
ax = axes[2]
n_match = sum(1 for v in verdict_list if v["match"])
n_mismatch = len(verdict_list) - n_match

ax.pie([n_match, n_mismatch],
       labels=[f"Match\n({n_match})", f"Mismatch\n({n_mismatch})"],
       colors=["#2ecc71", "#e74c3c"],
       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11, "fontweight": "bold"})
ax.set_title("Prediction Accuracy", fontsize=11, fontweight="bold")

fig.suptitle("Evidence Summary: Most Behaviors Have Strong Mechanistic Signals\n"
             "The classifier finds real circuits even for behaviors expected to be artifacts",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig7_evidence_summary")


# ============================================================
# Fig 8: C11 Multi-Turn Escalation
# ============================================================

print("Fig 8: C11 Escalation...")

esc_data = escalation.get("escalation", [])

if esc_data:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    protocol = esc_data[0]
    turns = protocol.get("turns", [])

    # Left: Agreement score across turns
    ax = axes[0]
    turn_nums = [t["turn"] for t in turns]
    agree_scores = [t.get("behavioral_metrics", {}).get("agreement_score", 0) for t in turns]
    hedge_scores = [t.get("behavioral_metrics", {}).get("hedging_score", 0) for t in turns]

    ax.plot(turn_nums, agree_scores, "o-", color="#e74c3c", linewidth=2.5,
            markersize=8, label="Agreement", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(turn_nums, hedge_scores, "s-", color="#3498db", linewidth=2.5,
            markersize=8, label="Hedging", markeredgecolor="white", markeredgewidth=1.5)

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Pressure Turn", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Behavioral Metrics Across\nEscalating Pressure", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.set_xticks(turn_nums)

    # Right: Total differential magnitude across turns
    ax = axes[1]
    diff_magnitudes = [t.get("total_diff_magnitude", 0) for t in turns]

    ax.bar(turn_nums, diff_magnitudes, color=CASE_COLORS[:len(turns)],
           edgecolor="white", linewidth=0.8)

    for ti, (tn, dm) in enumerate(zip(turn_nums, diff_magnitudes)):
        ax.text(tn, dm + max(diff_magnitudes) * 0.02,
                f"{dm:.0f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Pressure Turn", fontsize=11)
    ax.set_ylabel("Total |Feature Differential|", fontsize=11)
    ax.set_title("Feature Activation Magnitude\nIncreases with Pressure", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.set_xticks(turn_nums)

    fig.suptitle("C11 Sycophancy Escalation: 5-Turn Pressure Protocol\n"
                 "How does the model's internal representation change under sustained pressure?",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig8_c11_escalation")
else:
    print("  No C11 escalation data found, skipping Fig 8")


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("MI DETECTIVE — ALL PRODUCTION FIGURES GENERATED")
print(f"{'='*60}")
total_size = 0
for f in sorted(os.listdir(FIG)):
    fpath = os.path.join(FIG, f)
    size = os.path.getsize(fpath) / 1024
    total_size += size
    print(f"  {f}: {size:.0f} KB")
print(f"\n  Total: {total_size/1024:.1f} MB")
