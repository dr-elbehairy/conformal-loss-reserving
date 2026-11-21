# -*- coding: utf-8 -*-
"""Conformal upper bounds for the outstanding amount, their backtest against the realised lower triangle,
and the robustness checks reported in Section 4.2.
Input : core.pkl (from 01_core.py)      Output: results.json"""
import os, json, pickle
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(HERE, "core.pkl"), "rb"))
df = D["df"].copy(); LEVELS = D["levels"]; SEED = D["seed"]
rng = np.random.default_rng(SEED)
GROUP = {"ppauto": "Motor", "comauto": "Motor", "wkcomp": "Workers' compensation",
         "othliab": "Liability", "prodliab": "Liability", "medmal": "Liability"}
LABEL = {"ppauto": "Private passenger auto", "comauto": "Commercial auto", "wkcomp": "Workers' compensation",
         "othliab": "Other liability", "prodliab": "Products liability", "medmal": "Medical malpractice"}
df["grp"] = df.line.map(GROUP)
df["size"] = pd.qcut(df.prem, 3, labels=["Small", "Medium", "Large"]).astype(str)
floor = 1e-4 * df.prem
df["s_mack"] = np.maximum(df.mack_se, floor)
df["s_prem"] = df.prem
df["s5"] = np.maximum(df.se5, floor)
comp = df.grcode.values; ucomp = np.unique(comp)
n = len(df)
R = dict(n_triangles=int(n), n_companies=int(len(ucomp)), flow=D["flow"], levels=LEVELS, B=D["B"], seed=SEED,
         gbm=dict(best=D["best"], n_rows=int(D["n_rows"]), cv_mse=D["tune"][0][0], cv_mse_worst=D["tune"][-1][0]))


def cq(scores, p):
    """Conformal quantile: the ceil((m+1)p)-th smallest calibration score (infinite if m is too small)."""
    m = len(scores); k = int(np.ceil((m + 1) * p))
    return np.inf if k > m else np.sort(scores)[k - 1]


def loco_bound(point, scale, p, strata=None, data=df, true="true"):
    """Leave-one-company-out split conformal upper bound. All triangles of the test company are excluded
    from the calibration set, so that an insurer never calibrates its own bound."""
    pt = data[point].values; sc = data[scale].values; y = data[true].values; g = data.grcode.values
    s = (y - pt) / sc; U = np.empty(len(data))
    st = None if strata is None else data[strata].values
    for c in np.unique(g):
        te = g == c
        if st is None:
            U[te] = pt[te] + cq(s[~te], p) * sc[te]
        else:
            for v in np.unique(st[te]):
                tt = te & (st == v)
                U[tt] = pt[tt] + cq(s[(~te) & (st == v)], p) * sc[tt]
    return U


def cluster_ci(x, g, nb=4000, f=np.mean):
    ug = np.unique(g); idx = {c: np.where(g == c)[0] for c in ug}; out = np.empty(nb)
    for b in range(nb):
        pick = rng.choice(ug, size=len(ug), replace=True)
        out[b] = f(x[np.concatenate([idx[c] for c in pick])])
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def pinball(y, U, p):
    d = y - U
    return np.where(d > 0, p * d, (p - 1) * d)


def evaluate(U, p, data=df, true="true", ci=True):
    y = data[true].values; pr = data.prem.values; g = data.grcode.values
    fin = np.isfinite(U); cov = (y <= U).astype(float)
    out = dict(coverage=float(cov.mean()), n=int(len(y)), finite=float(fin.mean()))
    if ci: out["ci"] = cluster_ci(cov, g)
    if fin.all():
        pb = pinball(y, U, p) / pr
        out.update(pinball=float(pb.mean()), excess=float((np.maximum(U - y, 0) / pr).mean()),
                   shortfall=float((np.maximum(y - U, 0) / pr).mean()),
                   margin_med=float(np.median((U - data["cl"].values) / pr)))
        out["_pb"] = pb
    out["by_grp"] = {k: float(cov[data.grp.values == k].mean()) for k in sorted(set(data.grp))}
    out["by_size"] = {k: float(cov[data["size"].values == k].mean()) for k in ["Small", "Medium", "Large"]}
    out["by_line"] = {k: float(cov[data.line.values == k].mean()) for k in LABEL}
    return out


# ------------------------------------------------------------------ 1. sample description
desc = {}
for l, lab in LABEL.items():
    x = df[df.line == l]
    desc[l] = dict(label=lab, n=int(len(x)), companies=int(x.grcode.nunique()), prem_med=float(x.prem.median()),
                   prem_q1=float(x.prem.quantile(.25)), prem_q3=float(x.prem.quantile(.75)),
                   true_sum=float(x.true.sum()), paid_sum=float(x.latest_paid.sum()),
                   res_to_paid=float(x.true.sum() / x.latest_paid.sum()), single=float(x.single.mean()),
                   cv_mack_med=float(np.median(x.mack_se / np.abs(x.cl).clip(lower=1e-9))))
desc["all"] = dict(label="All lines", n=int(n), companies=int(len(ucomp)), prem_med=float(df.prem.median()),
                   prem_q1=float(df.prem.quantile(.25)), prem_q3=float(df.prem.quantile(.75)),
                   true_sum=float(df.true.sum()), paid_sum=float(df.latest_paid.sum()),
                   res_to_paid=float(df.true.sum() / df.latest_paid.sum()), single=float(df.single.mean()),
                   cv_mack_med=float(np.median(df.mack_se / np.abs(df.cl).clip(lower=1e-9))))
R["desc"] = desc
R["neg_incr_share"] = float((df.neg_incr > 0).mean())
R["tri_per_company"] = {int(k): int(v) for k, v in df.groupby("grcode").size().value_counts().sort_index().items()}

# ------------------------------------------------------------------ 2. point accuracy
pts = {"cl": "Chain ladder (own factors)", "pool": "Pooled line factors", "gbm": "Gradient boosting (pooled)"}
acc = {}
for k, lab in pts.items():
    e = (df[k] - df.true) / df.prem
    acc[k] = dict(label=lab, mae=float(e.abs().mean()), medae=float(e.abs().median()), rmse=float(np.sqrt((e ** 2).mean())),
                  bias=float(e.mean()), bias_med=float(e.median()), agg=float(df[k].sum() / df.true.sum()),
                  over_share=float((df[k] > df.true).mean()),
                  by_grp={g: dict(mae=float(e[df.grp == g].abs().mean()), medae=float(e[df.grp == g].abs().median()),
                                  agg=float(df[k][df.grp == g].sum() / df.true[df.grp == g].sum())) for g in sorted(set(df.grp))},
                  by_size={g: dict(mae=float(e[df["size"] == g].abs().mean()), medae=float(e[df["size"] == g].abs().median()))
                           for g in ["Small", "Medium", "Large"]})
for k in ("gbm_smear", "gbm_ratio", "gbm_bag_mean"):
    e = (df[k] - df.true) / df.prem
    acc[k] = dict(mae=float(e.abs().mean()), medae=float(e.abs().median()), rmse=float(np.sqrt((e ** 2).mean())),
                  bias=float(e.mean()), agg=float(df[k].sum() / df.true.sum()))
# paired comparisons (cluster bootstrap over companies)
for a, b in (("gbm", "cl"), ("pool", "cl"), ("gbm", "pool")):
    dlt = ((df[a] - df.true).abs() - (df[b] - df.true).abs()).values / df.prem.values
    acc[f"{a}_vs_{b}"] = dict(diff_mae=float(dlt.mean()), ci=cluster_ci(dlt, comp),
                              diff_med=float(np.median(dlt)), ci_med=cluster_ci(dlt, comp, f=np.median),
                              wins=float((dlt < 0).mean()), wilcoxon_p=float(stats.wilcoxon(dlt).pvalue))
acc["gbm_bag_cv_med"] = float(np.median(df.gbm_bag_sd / df.gbm.abs()))
R["acc"] = acc
R["model_selection"] = json.load(open(os.path.join(HERE, "model_selection.json")))

# ------------------------------------------------------------------ 3. coverage of upper bounds
METH = {
    "mack": ("Mack, lognormal", None),
    "odp": ("ODP bootstrap", None),
    "cp_cl_mack": ("Conformal: chain ladder, Mack scale", ("cl", "s_mack", None)),
    "cp_cl_prem": ("Conformal: chain ladder, premium scale", ("cl", "s_prem", None)),
    "cp_gbm_mack": ("Conformal: boosting, Mack scale", ("gbm", "s_mack", None)),
    "cp_gbm_prem": ("Conformal: boosting, premium scale", ("gbm", "s_prem", None)),
    "cp_pool_mack": ("Conformal: pooled factors, Mack scale", ("pool", "s_mack", None)),
    "mond_cl_mack": ("Conformal by line group: chain ladder, Mack scale", ("cl", "s_mack", "grp")),
    "mond_gbm_mack": ("Conformal by line group: boosting, Mack scale", ("gbm", "s_mack", "grp")),
    "size_cl_mack": ("Conformal by size class: chain ladder, Mack scale", ("cl", "s_mack", "size")),
}
bounds = {}; cov = {}
for key, (lab, spec) in METH.items():
    cov[key] = dict(label=lab, lv={})
    for p in LEVELS:
        if key == "mack": U = df[f"mack_q{p}"].values
        elif key == "odp": U = df[f"odp_q{p}"].values
        else: U = loco_bound(spec[0], spec[1], p, spec[2])
        bounds[(key, p)] = U
        cov[key]["lv"][str(p)] = evaluate(U, p)
# pinball differences against the two parametric benchmarks
cmp_ = {}
for key in METH:
    if key in ("mack", "odp"): continue
    cmp_[key] = {}
    for p in LEVELS:
        a = cov[key]["lv"][str(p)]
        if "_pb" not in a: continue
        for ref in ("mack", "odp"):
            dlt = a["_pb"] - cov[ref]["lv"][str(p)]["_pb"]
            cmp_[key][f"{ref}_{p}"] = dict(diff=float(dlt.mean()), ci=cluster_ci(dlt, comp))
for key in cov:
    for p in cov[key]["lv"]:
        cov[key]["lv"][p].pop("_pb", None)
R["cov"] = cov; R["pinball_cmp"] = cmp_

# binomial tests of nominal coverage (triangle level; dependence within company ignored here, see cluster CI)
bt = {}
for key in ("mack", "odp", "cp_cl_mack", "cp_gbm_mack", "mond_cl_mack"):
    bt[key] = {}
    for p in LEVELS:
        k = int((df.true.values <= bounds[(key, p)]).sum())
        bt[key][str(p)] = float(stats.binomtest(k, n, p).pvalue)
R["binom_p"] = bt

# conformal quantiles themselves (full-sample, for interpretation): multiples of the Mack standard error
sM = ((df.true - df.cl) / df.s_mack).values; sG = ((df.true - df.gbm) / df.s_mack).values
R["quantiles"] = dict(
    cl={str(p): float(cq(sM, p)) for p in LEVELS}, gbm={str(p): float(cq(sG, p)) for p in LEVELS},
    normal={str(p): float(stats.norm.ppf(p)) for p in LEVELS},
    cl_by_grp={g: {str(p): float(cq(sM[df.grp.values == g], p)) for p in LEVELS} for g in sorted(set(df.grp))},
    score_median=float(np.median(sM)), score_mean=float(np.mean(sM)), score_sd=float(np.std(sM, ddof=1)),
    score_skew=float(stats.skew(sM)), share_abs_gt2=float((np.abs(sM) > 2).mean()),
    share_lt_m2=float((sM < -2).mean()), share_gt_2=float((sM > 2).mean()))
kw = stats.kruskal(*[sM[df.grp.values == g] for g in sorted(set(df.grp))])
kws = stats.kruskal(*[sM[df["size"].values == g] for g in ["Small", "Medium", "Large"]])
R["exch"] = dict(kw_grp_stat=float(kw.statistic), kw_grp_p=float(kw.pvalue), kw_size_stat=float(kws.statistic), kw_size_p=float(kws.pvalue),
                 med_by_grp={g: float(np.median(sM[df.grp.values == g])) for g in sorted(set(df.grp))},
                 med_by_size={g: float(np.median(sM[df["size"].values == g])) for g in ["Small", "Medium", "Large"]},
                 iqr_by_size={g: float(stats.iqr(sM[df["size"].values == g])) for g in ["Small", "Medium", "Large"]})

# ------------------------------------------------------------------ 4. robustness
rob = {}
# 4a. repeated random company splits (70 % calibration / 30 % test)
rs = {k: {str(p): [] for p in LEVELS} for k in ("cp_cl_mack", "cp_gbm_mack")}
for b in range(1000):
    te_c = rng.choice(ucomp, size=int(round(0.3 * len(ucomp))), replace=False)
    te = np.isin(comp, te_c)
    for k, pt in (("cp_cl_mack", "cl"), ("cp_gbm_mack", "gbm")):
        s = ((df.true - df[pt]) / df.s_mack).values
        for p in LEVELS:
            q = cq(s[~te], p)
            rs[k][str(p)].append(float((s[te] <= q).mean()))
rob["splits"] = {k: {p: dict(mean=float(np.mean(v)), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)),
                             sd=float(np.std(v))) for p, v in d.items()} for k, d in rs.items()}
# 4b. calibration-set size
cs = {}
s = sM
for m in (20, 40, 80, 120):
    cs[str(m)] = {}
    for p in (0.75, 0.90, 0.95):
        v = []; inf = 0
        for b in range(2000):
            perm = rng.permutation(ucomp)
            cal_idx = np.where(np.isin(comp, perm[:len(ucomp) // 2]))[0]
            te_idx = np.where(np.isin(comp, perm[len(ucomp) // 2:]))[0]
            cal = rng.choice(cal_idx, size=m, replace=False)
            q = cq(s[cal], p)
            if not np.isfinite(q): inf += 1; continue
            v.append((s[te_idx] <= q).mean())
        cs[str(m)][str(p)] = dict(mean=float(np.mean(v)) if v else None, sd=float(np.std(v)) if v else None,
                                  q05=float(np.quantile(v, .05)) if v else None, infinite=float(inf / 2000))
rob["cal_size"] = cs
# 4c. deployable calibration: scores from the as-of-1992 sub-triangles (all realised by 1996)
s5 = ((df.true5 - df.cl5) / df.s5).values
tt = {}
for p in LEVELS:
    q = cq(s5, p); U = df.cl.values + q * df.s_mack.values
    e = evaluate(U, p); e.pop("_pb", None); e["q"] = float(q); tt[str(p)] = e
    qg = {g: cq(s5[df.grp.values == g], p) for g in sorted(set(df.grp))}
    Ug = df.cl.values + df.grp.map(qg).values * df.s_mack.values
    eg = evaluate(Ug, p); eg.pop("_pb", None); tt[str(p)]["by_group_cal"] = dict(coverage=eg["coverage"], ci=eg["ci"], finite=eg["finite"])
rob["temporal"] = dict(lv=tt, score5_median=float(np.median(s5)), score_median=float(np.median(sM)),
                       ks_p=float(stats.ks_2samp(s5, sM).pvalue), ks_stat=float(stats.ks_2samp(s5, sM).statistic),
                       mack5={str(p): float((df.true5 <= df.cl5 + stats.norm.ppf(p) * df.s5).mean()) for p in LEVELS})
# 4d. case-incurred triangles
di = df[df.inc_ok].copy(); di["s_inc"] = np.maximum(di.se_inc, 1e-4 * di.prem)
inc = dict(n=int(len(di)))
for p in LEVELS:
    U = loco_bound("cl_inc", "s_inc", p, None, data=di, true="true_inc")
    e = evaluate(U, p, data=di, true="true_inc"); e.pop("_pb", None)
    Um = np.array([x for x in (di.cl_inc + stats.norm.ppf(p) * di.s_inc)])
    inc[str(p)] = dict(cp=e["coverage"], cp_ci=e["ci"], mack_normal=float((di.true_inc.values <= Um).mean()), finite=e["finite"])
ei = (di.cl_inc - di.true_inc) / di.prem
inc["mae"] = float(ei.abs().mean()); inc["agg_bias"] = float((di.cl_inc.sum() - di.true_inc.sum()) / di.prem.sum())
rob["incurred"] = inc
# 4e. sample filters
filt = {"Net premium of at least 5,000 file units": df.prem >= 5000, "Single entities only": df.single == 1,
        "Groups only": df.single == 0, "No negative paid increments": df.neg_incr == 0,
        "Excluding private passenger auto": df.line != "ppauto",
        "One triangle per company (largest)": df.prem == df.groupby("grcode").prem.transform("max")}
fl = {}
for lab, msk in filt.items():
    sub = df[msk.values].copy(); fl[lab] = dict(n=int(len(sub)))
    for p in (0.75, 0.90, 0.95):
        U = loco_bound("cl", "s_mack", p, None, data=sub); Ug = loco_bound("gbm", "s_mack", p, None, data=sub)
        fl[lab][str(p)] = dict(cp_cl=float((sub.true.values <= U).mean()), cp_gbm=float((sub.true.values <= Ug).mean()),
                               mack=float((sub.true.values <= sub[f"mack_q{p}"].values).mean()),
                               odp=float((sub.true.values <= sub[f"odp_q{p}"].values).mean()))
rob["filters"] = fl
# 4f. alternative learner specifications
alt = {}
for k in ("gbm_smear", "gbm_ratio", "gbm_bag_mean"):
    alt[k] = {}
    for p in (0.75, 0.90, 0.95):
        U = loco_bound(k, "s_mack", p)
        e = evaluate(U, p, ci=False); alt[k][str(p)] = dict(coverage=e["coverage"], pinball=e.get("pinball"))
rob["alt_learner"] = alt
# 4g. two-sided 90 % interval (lower and upper 5 % conformal quantiles)
two = {}
for k, pt in (("cl", "cl"), ("gbm", "gbm")):
    s = ((df.true - df[pt]) / df.s_mack).values; inside = np.zeros(n); width = np.zeros(n)
    for c in ucomp:
        te = comp == c; cal = s[~te]; hi = cq(cal, 0.95); lo = -cq(-cal, 0.95)
        inside[te] = (s[te] >= lo) & (s[te] <= hi); width[te] = (hi - lo) * df.s_mack.values[te] / df.prem.values[te]
    two[k] = dict(coverage=float(inside.mean()), ci=cluster_ci(inside, comp), width_med=float(np.median(width)))
z = stats.norm.ppf(0.95)
mk_in = ((df.true >= df.cl - z * df.mack_se) & (df.true <= df.cl + z * df.mack_se)).values.astype(float)
two["mack_normal"] = dict(coverage=float(mk_in.mean()), ci=cluster_ci(mk_in, comp), width_med=float(np.median(2 * z * df.mack_se / df.prem)))
rob["two_sided"] = two
R["rob"] = rob

# aggregate view: sum of bounds against sum of realised amounts, by line group
agg = {}
for key in ("mack", "odp", "cp_cl_mack", "cp_gbm_mack"):
    agg[key] = {str(p): dict(total=float(np.sum(bounds[(key, p)]) / df.true.sum())) for p in LEVELS if np.isfinite(bounds[(key, p)]).all()}
R["agg"] = agg

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
np.save(os.path.join(HERE, "scores.npy"), np.c_[sM, sG, s5])
df.drop(columns=["name"]).to_csv(os.path.join(HERE, "triangle_level_results.csv"), index=False)

# ------------------------------------------------------------------ console summary
print("n", n, "companies", len(ucomp))
print("POINT", {k: (round(acc[k]["mae"], 4), round(acc[k]["medae"], 4), round(acc[k]["agg"], 3)) for k in pts})
print("gbm vs cl", acc["gbm_vs_cl"])
for key in cov:
    print(f"{key:14s}", " ".join(f"{p}:{cov[key]['lv'][str(p)]['coverage']:.3f}" + (f"/pb{cov[key]['lv'][str(p)]['pinball']:.4f}" if 'pinball' in cov[key]['lv'][str(p)] else "/inf") for p in LEVELS))
print("quantiles", R["quantiles"]["cl"], R["quantiles"]["gbm"])
print("exch", R["exch"])
print("temporal", {p: (round(v["coverage"], 3), round(v["q"], 2), round(v["by_group_cal"]["coverage"], 3)) for p, v in tt.items()}, rob["temporal"]["ks_p"])
print("incurred", inc)
print("two-sided", two)
print("splits", rob["splits"]["cp_cl_mack"])
print("calsize", cs)
