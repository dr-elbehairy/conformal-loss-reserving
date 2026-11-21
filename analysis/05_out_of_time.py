# -*- coding: utf-8 -*-
"""Out-of-time test. Conformal multipliers are calibrated on the 1997 valuation (outcomes complete by 2006) and applied,
unchanged, to the 2007 valuation (accident years 1998-2007, outcomes to 2016). Everything used for the 2007 bounds was
observable at year-end 2007. Input: core.pkl, core_2007.pkl. Output: results_2007.json"""
import os, json, pickle
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
A = pickle.load(open(os.path.join(HERE, "core.pkl"), "rb")); B = pickle.load(open(os.path.join(HERE, "core_2007.pkl"), "rb"))
d97 = A["df"].copy(); d07 = B["df"].copy(); LEVELS = A["levels"]; rng = np.random.default_rng(A["seed"] + 7)
GROUP = {"ppauto": "Motor", "comauto": "Motor", "wkcomp": "Workers' compensation", "othliab": "Liability", "prodliab": "Liability", "medmal": "Liability"}
for d in (d97, d07):
    d["grp"] = d.line.map(GROUP); d["s_mack"] = np.maximum(d.mack_se, 1e-4 * d.prem)
d07["size"] = pd.qcut(d07.prem, 3, labels=["Small", "Medium", "Large"]).astype(str)
n = len(d07); comp = d07.grcode.values

def cq(s, p):
    m = len(s); k = int(np.ceil((m + 1) * p)); return np.inf if k > m else np.sort(s)[k - 1]
def cluster_ci(x, g, nb=4000):
    ug = np.unique(g); idx = {c: np.where(g == c)[0] for c in ug}; out = np.empty(nb)
    for b in range(nb):
        out[b] = x[np.concatenate([idx[c] for c in rng.choice(ug, size=len(ug), replace=True)])].mean()
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]
def pinball(y, U, p):
    e = y - U; return np.where(e > 0, p * e, (p - 1) * e)
def ev(U, p):
    y = d07.true.values; c = (y <= U).astype(float); fin = bool(np.isfinite(U).all())
    o = dict(coverage=float(c.mean()), ci=cluster_ci(c, comp), finite=fin,
             by_grp={g: float(c[d07.grp.values == g].mean()) for g in sorted(set(d07.grp))},
             by_size={g: float(c[d07["size"].values == g].mean()) for g in ["Small", "Medium", "Large"]},
             binom_p=float(stats.binomtest(int(c.sum()), n, p).pvalue))
    if fin:
        o["pinball"] = float((pinball(y, U, p) / d07.prem.values).mean()); o["margin_med"] = float(np.median((U - d07.cl.values) / d07.prem.values))
        o["_pb"] = pinball(y, U, p) / d07.prem.values
    return o
def loco(point, p, strata=None):
    pt = d07[point].values; sc = d07.s_mack.values; s = (d07.true.values - pt) / sc; U = np.empty(n)
    st = None if strata is None else d07[strata].values
    for c in np.unique(comp):
        te = comp == c
        if st is None: U[te] = pt[te] + cq(s[~te], p) * sc[te]
        else:
            for v in np.unique(st[te]):
                tt = te & (st == v); U[tt] = pt[tt] + cq(s[(~te) & (st == v)], p) * sc[tt]
    return U

R = dict(n_triangles=int(n), n_companies=int(len(np.unique(comp))), flow=B["flow"],
         overlap_companies=int(len(set(d97.grcode) & set(d07.grcode))),
         overlap_triangles=int(len(set(zip(d97.grcode, d97.line)) & set(zip(d07.grcode, d07.line)))))
# description and point accuracy
R["desc"] = dict(prem_med=float(d07.prem.median()), prem_q1=float(d07.prem.quantile(.25)), prem_q3=float(d07.prem.quantile(.75)),
                 true_sum=float(d07.true.sum()), paid_sum=float(d07.latest_paid.sum()), single=float(d07.single.mean()),
                 by_line={l: int((d07.line == l).sum()) for l in GROUP}, neg_true=int((d07.true <= 0).sum()))
acc = {}
for k in ("cl", "pool", "gbm"):
    e = (d07[k] - d07.true) / d07.prem
    acc[k] = dict(mae=float(e.abs().mean()), medae=float(e.abs().median()), bias=float(e.mean()), agg=float(d07[k].sum() / d07.true.sum()),
                  over_share=float((d07[k] > d07.true).mean()))
dl = ((d07.gbm - d07.true).abs() - (d07.cl - d07.true).abs()).values / d07.prem.values
acc["gbm_vs_cl"] = dict(diff_mae=float(dl.mean()), ci=cluster_ci(dl, comp), wins=float((dl < 0).mean()))
R["acc"] = acc
# scores of the two cohorts
s97 = ((d97.true - d97.cl) / d97.s_mack).values; s07 = ((d07.true - d07.cl) / d07.s_mack).values
g97 = ((d97.true - d97.gbm) / d97.s_mack).values
ks = stats.ks_2samp(s97, s07)
R["scores"] = dict(med97=float(np.median(s97)), med07=float(np.median(s07)), sd97=float(np.std(s97, ddof=1)), sd07=float(np.std(s07, ddof=1)),
                   gt2_97=float((s97 > 2).mean()), gt2_07=float((s07 > 2).mean()), lt2_97=float((s97 < -2).mean()), lt2_07=float((s07 < -2).mean()),
                   ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
cov = {}
for key in ("mack", "odp", "oot_cl", "oot_cl_grp", "oot_gbm", "within_cl", "within_cl_grp"):
    cov[key] = {}
    for p in LEVELS:
        if key == "mack": U = d07[f"mack_q{p}"].values
        elif key == "odp": U = d07[f"odp_q{p}"].values
        elif key == "oot_cl": q = cq(s97, p); U = d07.cl.values + q * d07.s_mack.values
        elif key == "oot_cl_grp":
            qg = {g: cq(s97[d97.grp.values == g], p) for g in sorted(set(d97.grp))}; U = d07.cl.values + d07.grp.map(qg).values * d07.s_mack.values
        elif key == "oot_gbm": q = cq(g97, p); U = d07.gbm.values + q * d07.s_mack.values
        elif key == "within_cl": U = loco("cl", p)
        else: U = loco("cl", p, "grp")
        cov[key][str(p)] = ev(U, p)
        if key.startswith("oot_cl") and key == "oot_cl": cov[key][str(p)]["q"] = float(cq(s97, p))
    
for ref in ("mack", "odp"):
    for p in LEVELS:
        a = cov["oot_cl"][str(p)]; b = cov[ref][str(p)]
        if "_pb" in a and "_pb" in b:
            dlt = a["_pb"] - b["_pb"]; a[f"pb_vs_{ref}"] = dict(diff=float(dlt.mean()), ci=cluster_ci(dlt, comp))
for k in cov:
    for p in cov[k]: cov[k][p].pop("_pb", None)
R["cov"] = cov
R["q07_own"] = {str(p): float(cq(s07, p)) for p in LEVELS}
# deployable alternatives at the 2007 valuation
d07["s5"] = np.maximum(d07.se5, 1e-4 * d07.prem); d97["s5"] = np.maximum(d97.se5, 1e-4 * d97.prem)
s5_07 = ((d07.true5 - d07.cl5) / d07.s5).values            # as-of-2002 sub-triangles, realised by 2006
s5_97 = ((d97.true5 - d97.cl5) / d97.s5).values            # as-of-1992 sub-triangles, realised by 1996
alt = {}
for lab, cal in (("design2002", s5_07), ("pooled_1997_and_2002", np.r_[s97, s5_07]), ("max_1997_2002", None)):
    alt[lab] = {}
    for p in LEVELS:
        q = max(cq(s97, p), cq(s5_07, p)) if cal is None else cq(cal, p)
        e = ev(d07.cl.values + q * d07.s_mack.values, p); e.pop("_pb", None); e["q"] = float(q); alt[lab][str(p)] = e
R["alt_2007"] = alt
R["mack_normal_5x5"] = {"1992": {str(p): float((d97.true5 <= d97.cl5 + stats.norm.ppf(p) * d97.s5).mean()) for p in LEVELS},
                        "2002": {str(p): float((d07.true5 <= d07.cl5 + stats.norm.ppf(p) * d07.s5).mean()) for p in LEVELS}}
R["scores"]["med5_1992"] = float(np.median(s5_97)); R["scores"]["med5_2002"] = float(np.median(s5_07))
print("ALT", {k: {p: (round(v[p]["coverage"], 3), round(v[p]["q"], 2), round(v[p].get("margin_med", float("nan")), 2)) for p in v} for k, v in alt.items()})
print("5x5", R["mack_normal_5x5"], R["scores"]["med5_1992"], R["scores"]["med5_2002"])
json.dump(R, open(os.path.join(HERE, "results_2007.json"), "w"), indent=1)
np.save(os.path.join(HERE, "scores_2007.npy"), s07)
d07.drop(columns=["name"]).to_csv(os.path.join(HERE, "triangle_level_results_2007.csv"), index=False)
print("n", n, R["n_companies"], "overlap", R["overlap_companies"], R["overlap_triangles"], R["desc"]["by_line"])
print("acc", {k: {a: (round(b, 3) if not isinstance(b, list) else [round(x, 3) for x in b]) for a, b in v.items()} for k, v in acc.items()})
print("scores", {k: round(v, 4) for k, v in R["scores"].items()})
for k in cov:
    print(f"{k:14s}", " | ".join(f"{p}: {cov[k][str(p)]['coverage']:.3f} [{cov[k][str(p)]['ci'][0]:.3f},{cov[k][str(p)]['ci'][1]:.3f}] pb {cov[k][str(p)].get('pinball', float('nan')):.4f}" for p in LEVELS))
print("grp95", {k: {g[:5]: round(v, 3) for g, v in cov[k]["0.95"]["by_grp"].items()} for k in cov})
print("q07", R["q07_own"], "pbcmp", {p: cov["oot_cl"][str(p)].get("pb_vs_mack") for p in LEVELS})
