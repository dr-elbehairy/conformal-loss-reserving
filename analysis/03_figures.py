# -*- coding: utf-8 -*-
"""Figures 1-5. Input: results.json, triangle_level_results.csv, scores.npy. Output: ../figures/*.png"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
R2 = json.load(open(os.path.join(HERE, "results_2007.json")))
S07 = np.load(os.path.join(HERE, "scores_2007.npy"))
df = pd.read_csv(os.path.join(HERE, "triangle_level_results.csv"))
S = np.load(os.path.join(HERE, "scores.npy"))
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300, "figure.dpi": 120})
C = {"mack": "#1C4076", "odp": "#B8860B", "cp_cl_mack": "#007A88", "cp_gbm_mack": "#8C2D19"}
LAB = {"mack": "Mack, lognormal", "odp": "ODP bootstrap", "cp_cl_mack": "Conformal, chain ladder", "cp_gbm_mack": "Conformal, boosting"}
LV = ["0.75", "0.9", "0.95", "0.995"]

# ---- Figure 1: realised against nominal coverage, both valuation dates
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
x = np.arange(len(LV))
def panel(ax, series, title):
    k = len(series)
    for j, (lab, col, vals) in enumerate(series):
        o = (j - (k - 1) / 2) * 0.17
        cv = [100 * v["coverage"] for v in vals]; lo = [100 * (v["coverage"] - v["ci"][0]) for v in vals]; hi = [100 * (v["ci"][1] - v["coverage"]) for v in vals]
        ax.errorbar(x + o, cv, yerr=[lo, hi], fmt="o", color=col, capsize=2.5, ms=4.5, label=lab, lw=1.1)
    for i, p in enumerate(LV): ax.hlines(100 * float(p), i - 0.42, i + 0.42, color="black", lw=1.0, linestyles="--")
    ax.set_xticks(x); ax.set_xticklabels(["75%", "90%", "95%", "99.5%"]); ax.set_xlabel("Nominal confidence level of the upper bound"); ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
panel(axs[0], [("Mack, lognormal", "#1C4076", [R["cov"]["mack"]["lv"][p] for p in LV]), ("ODP bootstrap", "#B8860B", [R["cov"]["odp"]["lv"][p] for p in LV]),
               ("Conformal, short design of 1992", "#8C2D19", [R["rob"]["temporal"]["lv"][p] for p in LV])], "(a) Valuation at year-end 1997")
panel(axs[1], [("Mack, lognormal", "#1C4076", [R2["cov"]["mack"][p] for p in LV]), ("ODP bootstrap", "#B8860B", [R2["cov"]["odp"][p] for p in LV]),
               ("Conformal, multipliers of 1997", "#007A88", [R2["cov"]["oot_cl"][p] for p in LV]),
               ("Conformal, short design of 2002", "#8C2D19", [R2["alt_2007"]["design2002"][p] for p in LV])], "(b) Valuation at year-end 2007")
axs[0].set_ylabel("Share of triangles covered (%)"); axs[0].set_ylim(52, 101)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_coverage.png")); plt.close(fig)

# ---- Figure 2: distribution of standardised reserve errors
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
s = np.clip(S[:, 0], -6, 8); grid = np.linspace(-6, 8, 400)
axs[0].hist(s, bins=np.arange(-6, 8.01, 0.5), density=True, color="#9DB7D5", edgecolor="white")
axs[0].plot(grid, stats.norm.pdf(grid), color="black", lw=1.2, label="Standard normal")
axs[0].axvline(np.median(S[:, 0]), color="#8C2D19", lw=1.2, ls="--", label="Median")
axs[0].set_xlabel("(realised − chain ladder) / Mack s.e."); axs[0].set_ylabel("Density"); axs[0].legend(frameon=False, fontsize=8.5)
axs[0].set_title("(a) Standardised reserve error, 1997 valuation", fontsize=10)
osm = stats.norm.ppf((np.arange(1, len(S) + 1) - 0.5) / len(S))
axs[1].plot(osm, np.sort(np.clip(S[:, 0], -8, 15)), ".", color="#1C4076", ms=4)
axs[1].plot([-3, 3], [-3, 3], color="black", lw=1)
axs[1].set_xlabel("Standard normal quantile"); axs[1].set_ylabel("Empirical quantile of the score")
axs[1].set_title("(b) Normal quantile plot (scores above 15 clipped)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_scores.png")); plt.close(fig)

# ---- Figure 3: coverage at 95 % by line group and by size class
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=True)
ks = ["mack", "odp", "cp_cl_mack", "mond_cl_mack"]; cols = ["#1C4076", "#B8860B", "#007A88", "#6BB8A8"]
labs = ["Mack, lognormal", "ODP bootstrap", "Conformal, pooled", "Conformal, by line group"]
for ax, key, cats, ttl in ((axs[0], "by_grp", ["Motor", "Workers' compensation", "Liability"], "(a) By line group"),
                           (axs[1], "by_size", ["Small", "Medium", "Large"], "(b) By premium tercile")):
    xx = np.arange(len(cats))
    for j, k in enumerate(ks):
        ax.bar(xx + (j - 1.5) * 0.2, [100 * R["cov"][k]["lv"]["0.95"][key][c] for c in cats], 0.19, color=cols[j], label=labs[j])
    ax.axhline(95, color="black", ls="--", lw=1); ax.set_xticks(xx); ax.set_xticklabels([c.replace("Workers' compensation", "Workers' comp.") for c in cats])
    ax.set_ylim(80, 100); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel("Covered at nominal 95% (%)"); axs[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_conditional.png")); plt.close(fig)

# ---- Figure 4: scores of the 1992 design against the 1997 design, and calibration-set size
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for v, lab, c in ((S[:, 0], "Valuation 1997 (outcome 2006)", "#1C4076"), (S07, "Valuation 2007 (outcome 2016)", "#8C2D19"), (S[:, 2], "Short design 1992 (outcome 1996)", "#B8860B")):
    v = np.sort(v); axs[0].step(v, np.arange(1, len(v) + 1) / len(v), color=c, label=lab, lw=1.3)
axs[0].set_xlim(-6, 8); axs[0].set_xlabel("Standardised reserve error"); axs[0].set_ylabel("Cumulative share of triangles")
axs[0].legend(frameon=False, fontsize=8, loc="lower right"); axs[0].set_title("(a) Score distributions", fontsize=10)
ms = ["20", "40", "80", "120"]
for p, c in (("0.9", "#007A88"), ("0.95", "#8C2D19")):
    mu = np.array([100 * R["rob"]["cal_size"][m][p]["mean"] for m in ms]); sd = np.array([100 * R["rob"]["cal_size"][m][p]["sd"] for m in ms])
    axs[1].errorbar([int(m) for m in ms], mu, yerr=1.645 * sd, fmt="o-", color=c, capsize=3, label=f"Nominal {int(float(p) * 100)}%", lw=1.2, ms=4)
    axs[1].axhline(100 * float(p), color=c, ls=":", lw=1)
axs[1].set_xlabel("Number of calibration triangles"); axs[1].set_ylabel("Coverage on held-out companies (%)")
axs[1].legend(frameon=False, fontsize=8.5, loc="lower right"); axs[1].set_title("(b) Coverage and calibration-set size", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_design.png")); plt.close(fig)

# ---- Figure 5: point forecasts against realised outstanding amounts
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
ok = (df.true > 0)
for ax, k, ttl in ((axs[0], "cl", "(a) Chain ladder, own factors"), (axs[1], "gbm", "(b) Pooled gradient boosting")):
    m = ok & (df[k] > 0)
    for g, c in (("Motor", "#1C4076"), ("Workers' compensation", "#B8860B"), ("Liability", "#8C2D19")):
        grp = df.line.map({"ppauto": "Motor", "comauto": "Motor", "wkcomp": "Workers' compensation", "othliab": "Liability", "prodliab": "Liability", "medmal": "Liability"})
        mm = m & (grp == g); ax.loglog(df.true[mm], df[k][mm], ".", color=c, ms=5, alpha=0.75, label=g)
    lim = [1, 2e7]; ax.plot(lim, lim, color="black", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Realised outstanding amount (file units)"); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel("Forecast at year-end 1997 (file units)"); axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_point.png")); plt.close(fig)
print("figures written to", FIG)
