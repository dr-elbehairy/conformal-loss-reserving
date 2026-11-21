# -*- coding: utf-8 -*-
"""Core estimates: data selection, chain ladder / Mack, ODP bootstrap, pooled gradient boosting.
Input : data/*_pos.csv  (CAS loss reserving database, NAIC Schedule P, accident years 1988-1997)
Output: core.pkl (per-triangle point forecasts, scales, benchmark quantiles, true outstanding amounts)
Everything used for fitting lies in the upper triangle (calendar year <= 1997). The lower triangle is used
only as the realised outcome."""
import os, re, sys, json, pickle, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
LINES = ["ppauto", "comauto", "wkcomp", "othliab", "prodliab", "medmal"]
LEVELS = [0.75, 0.90, 0.95, 0.995]
B = 2000
SEED = 20261121
I = 10
# cohort switch: "1997" = accident years 1988-1997 (Meyers and Shi files); "2007" = accident years 1998-2007 (S&P files)
COHORT = "2007" if "2007" in sys.argv[1:] else "1997"
DATADIR, FILEPAT, TAG = ("data2007", "{}_pos_98-07.csv", "_2007") if COHORT == "2007" else ("data", "{}_pos.csv", "")


def load(line):
    d = pd.read_csv(os.path.join(HERE, DATADIR, FILEPAT.format(line)))
    d.columns = [re.sub(r"_[A-Za-z]\d?$", "", c) for c in d.columns]
    d = d.rename(columns={"IncurredLosses": "IncurLoss", "PostedReserves2007": "PostedReserve97"})
    return d


def upper_mask(n=I):
    return np.add.outer(np.arange(n), np.arange(n)) <= n - 1


UP = upper_mask()


# ------------------------------------------------------------------ chain ladder and Mack (1993)
def chain_ladder(C, n=None):
    """C: n x n cumulative array, only the upper triangle is read. Returns factors, projected square,
    reserve, Mack standard error of the total reserve."""
    n = C.shape[0] if n is None else n
    f = np.ones(n - 1); S = np.zeros(n - 1); sig2 = np.zeros(n - 1)
    for j in range(n - 1):
        rows = np.arange(0, n - j - 1)
        S[j] = C[rows, j].sum()
        f[j] = C[rows, j + 1].sum() / S[j]
        if len(rows) > 1:
            sig2[j] = (C[rows, j] * (C[rows, j + 1] / C[rows, j] - f[j]) ** 2).sum() / (len(rows) - 1)
    if n >= 4:
        a, b = sig2[n - 4], sig2[n - 3]
        sig2[n - 2] = min(b * b / a, min(a, b)) if a > 0 else 0.0
    else:
        sig2[n - 2] = sig2[n - 3] if n == 3 else 0.0
    proj = C.astype(float).copy()
    for i in range(1, n):
        for j in range(n - i, n):
            proj[i, j] = proj[i, j - 1] * f[j - 1]
    ult = proj[:, n - 1]
    latest = np.array([C[i, n - 1 - i] for i in range(n)], dtype=float)
    res_i = ult - latest
    msep_i = np.zeros(n); total = 0.0
    for i in range(1, n):
        ks = np.arange(n - 1 - i, n - 1)
        msep_i[i] = ult[i] ** 2 * (sig2[ks] / f[ks] ** 2 * (1.0 / proj[i, ks] + 1.0 / S[ks])).sum()
        cov = ult[i] * ult[i + 1:].sum() * (2 * sig2[ks] / (f[ks] ** 2 * S[ks])).sum()
        total += msep_i[i] + cov
    return f, proj, res_i.sum(), float(np.sqrt(max(total, 0.0)))


def lognormal_q(mean, se, p):
    """Upper quantile of a lognormal matched to (mean, se); normal fallback when the mean is not positive."""
    if mean > 0 and se > 0:
        s2 = np.log(1 + (se / mean) ** 2)
        return float(np.exp(np.log(mean) - s2 / 2 + np.sqrt(s2) * stats.norm.ppf(p)))
    return float(mean + se * stats.norm.ppf(p))


# ------------------------------------------------------------------ ODP bootstrap (England & Verrall, 2002)
def odp_bootstrap(C, rng, nb=B):
    n = C.shape[0]
    f, proj, _, _ = chain_ladder(C)
    fit = np.zeros((n, n))
    for i in range(n):
        fit[i, n - 1 - i] = C[i, n - 1 - i]
        for j in range(n - 2 - i, -1, -1):
            fit[i, j] = fit[i, j + 1] / f[j]
    m = np.diff(np.concatenate([np.zeros((n, 1)), fit], axis=1), axis=1)
    x = np.diff(np.concatenate([np.zeros((n, 1)), np.where(UP, C, 0.0)], axis=1), axis=1)
    am = np.maximum(np.abs(m), 1e-9)
    r = (x - m) / np.sqrt(am)
    use = UP.copy(); use[0, n - 1] = False; use[n - 1, 0] = False      # corner residuals are zero by construction
    nobs = n * (n + 1) // 2; p = 2 * n - 1
    phi = (r[UP] ** 2).sum() / (nobs - p)
    pool = r[use] * np.sqrt(nobs / (nobs - p))
    rs = rng.choice(pool, size=(nb, n, n), replace=True)
    xs = np.where(UP, m + rs * np.sqrt(am), 0.0)
    Cs = np.cumsum(xs, axis=2)
    res = np.zeros(nb)
    fs = np.ones((nb, n - 1))
    for j in range(n - 1):
        rows = np.arange(0, n - j - 1)
        den = Cs[:, rows, j].sum(axis=1)
        num = Cs[:, rows, j + 1].sum(axis=1)
        fs[:, j] = np.where(np.abs(den) > 1e-9, num / np.where(np.abs(den) > 1e-9, den, 1.0), f[j])
    fs = np.clip(fs, 0.2, 50.0)
    # forecast from the pseudo triangle (its own diagonal and its own factors), then add process error
    # independently to every future incremental cell, as in England and Verrall (2002, Appendix 3)
    for i in range(1, n):
        cur = Cs[:, i, n - 1 - i].copy()
        for j in range(n - 1 - i, n - 1):
            nxt = cur * fs[:, j]
            mu = nxt - cur
            a = np.maximum(np.abs(mu), 1e-9)
            res += rng.gamma(shape=a / max(phi, 1e-9), scale=max(phi, 1e-9)) * np.sign(mu)
            cur = nxt
    return res, float(phi)


# ------------------------------------------------------------------ data selection
def build_panel():
    tri = []; flow = {}
    for line in LINES:
        d = load(line)
        n_all = d.GRCODE.nunique(); kept = 0
        for g, x in d.groupby("GRCODE"):
            Pm = x.pivot(index="AccidentYear", columns="DevelopmentLag", values="CumPaidLoss")
            Im = x.pivot(index="AccidentYear", columns="DevelopmentLag", values="IncurLoss")
            Bm = x.pivot(index="AccidentYear", columns="DevelopmentLag", values="BulkLoss")
            if Pm.shape != (I, I) or Pm.isna().any().any():
                continue
            pr = x.groupby("AccidentYear").EarnedPremNet.first().values.astype(float)
            Pv = Pm.values.astype(float)
            if not ((Pv > 0).all() and (pr > 0).all()):
                continue
            kept += 1
            tri.append(dict(line=line, grcode=int(g), name=str(x.GRNAME.iloc[0]), single=int(x.Single.iloc[0]),
                            P=Pv, CI=(Im.values - Bm.values).astype(float), prem=pr,
                            posted97=float(x.PostedReserve97.iloc[0])))
        flow[line] = dict(entities=int(n_all), retained=int(kept))
    return tri, flow


# ------------------------------------------------------------------ pooled gradient boosting on link ratios
FEATS = ["lag", "line", "ay", "log_prem", "paid_lr", "prev_link", "own_logf", "log_size"]


def train_rows(tri, spec="ratio"):
    """spec='ratio': target is the development ratio minus one, so that a constant fit per lag with volume
    weights reproduces the chain-ladder factor exactly. spec='log': target is the log ratio (robustness)."""
    rows = []
    for t, T in enumerate(tri):
        C = T["P"]; pr = T["prem"]; lc = LINES.index(T["line"]); size = np.log(pr.mean())
        for j in range(I - 1):
            rr = np.arange(0, I - j - 1)
            num, den = C[rr, j + 1].sum(), C[rr, j].sum()
            for i in rr:
                own = np.nan
                if len(rr) > 1:
                    own = np.log(max((num - C[i, j + 1]) / (den - C[i, j]), 1e-3))
                prev = np.log(C[i, j] / C[i, j - 1]) if j >= 1 else np.nan
                yv = np.log(C[i, j + 1] / C[i, j]) if spec == "log" else C[i, j + 1] / C[i, j] - 1.0
                rows.append((t, T["grcode"], j, lc, i, np.log(pr[i]), C[i, j] / pr[i], prev, own, size,
                             yv, C[i, j] / den, C[i, j], C[i, j + 1]))
    return pd.DataFrame(rows, columns=["t", "grcode"] + FEATS + ["y", "w", "c0", "c1"])


def smearing(model, df):
    """Retransformation correction (Duan, 1983) by line and lag, estimated on the training rows only:
    the volume-weighted mean of the fitted ratios is made equal to the volume-weighted observed ratio."""
    fit = np.exp(model.predict(df[FEATS].values))
    g = pd.DataFrame(dict(line=df["line"].values, lag=df["lag"].values, num=df["c1"].values, den=df["c0"].values * fit))
    a = g.groupby(["line", "lag"])[["num", "den"]].sum()
    return (a["num"] / a["den"]).to_dict()


def gbm_fit(df, params):
    m = HistGradientBoostingRegressor(loss="squared_error", categorical_features=[1], random_state=SEED, **params)
    m.fit(df[FEATS].values, df["y"].values, sample_weight=df["w"].values)
    return m


def gbm_tune(df):
    grid = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=ml, l2_regularization=1.0)
            for d in (2, 3) for lr in (0.03, 0.1) for it in (200, 400) for ml in (20, 50)]
    gkf = GroupKFold(n_splits=5); out = []
    for prm in grid:
        err = 0.0; wsum = 0.0
        for tr, te in gkf.split(df, groups=df.grcode):
            m = gbm_fit(df.iloc[tr], prm)
            e = (m.predict(df.iloc[te][FEATS].values) - df.iloc[te]["y"].values) ** 2
            err += (e * df.iloc[te]["w"].values).sum(); wsum += df.iloc[te]["w"].values.sum()
        out.append((err / wsum, prm))
    out.sort(key=lambda z: z[0])
    return out


def gbm_project(model, tri, spec="ratio", smear=None):
    """Roll every triangle forward to lag 10 with the fitted link model."""
    res = np.zeros(len(tri))
    cur = [T["P"].astype(float).copy() for T in tri]
    fown = [np.log(np.maximum(chain_ladder(T["P"])[0], 1e-3)) for T in tri]
    for j in range(I - 1):
        X = []; idx = []
        for t, T in enumerate(tri):
            pr = T["prem"]; lc = LINES.index(T["line"]); size = np.log(pr.mean())
            for i in range(I - 1 - j, I):
                C = cur[t]
                prev = np.log(max(C[i, j], 1e-9) / max(C[i, j - 1], 1e-9)) if j >= 1 else np.nan
                X.append((j, lc, i, np.log(pr[i]), C[i, j] / pr[i], prev, fown[t][j], size)); idx.append((t, i))
        raw = model.predict(np.array(X, dtype=float))
        fac = np.exp(raw) if spec == "log" else 1.0 + raw
        if smear is not None:
            fac = fac * np.array([smear.get((int(x[1]), int(x[0])), 1.0) for x in X])
        fac = np.clip(fac, 0.5, 20.0)
        for (t, i), y in zip(idx, fac):
            cur[t][i, j + 1] = cur[t][i, j] * y
    for t, T in enumerate(tri):
        latest = np.array([T["P"][i, I - 1 - i] for i in range(I)])
        res[t] = (cur[t][:, I - 1] - latest).sum()
    return res


def pooled_line_factors(tri):
    out = {}
    for line in LINES:
        num = np.zeros(I - 1); den = np.zeros(I - 1)
        for T in tri:
            if T["line"] != line: continue
            for j in range(I - 1):
                rr = np.arange(0, I - j - 1)
                num[j] += T["P"][rr, j + 1].sum(); den[j] += T["P"][rr, j].sum()
        out[line] = num / den
    return out


def reserve_with_factors(C, f):
    r = 0.0
    for i in range(1, I):
        r += C[i, I - 1 - i] * (np.prod(f[I - 1 - i:]) - 1.0)
    return r


def main():
    rng = np.random.default_rng(SEED)
    tri, flow = build_panel()
    print("triangles retained:", len(tri), flow)
    rec = []
    for t, T in enumerate(tri):
        C = T["P"]
        latest = np.array([C[i, I - 1 - i] for i in range(I)])
        true_res = float((C[:, I - 1] - latest).sum())
        f, proj, r_cl, se = chain_ladder(C)
        boot, phi = odp_bootstrap(C, rng)
        # as-of-1992 sub-problem (accident years 1988-1992, lags 1-5): fully observed by 1996
        C5 = C[:5, :5]
        latest5 = np.array([C5[i, 4 - i] for i in range(5)])
        true5 = float((C5[:, 4] - latest5).sum())
        _, _, r5, se5 = chain_ladder(C5)
        # case-incurred triangle (robustness)
        CI = T["CI"]; inc_ok = bool((CI[UP] > 0).all() and (CI[:, I - 1] > 0).all())
        r_inc = se_inc = true_inc = np.nan
        if inc_ok:
            lat_i = np.array([CI[i, I - 1 - i] for i in range(I)])
            true_inc = float((CI[:, I - 1] - lat_i).sum())
            _, _, r_inc, se_inc = chain_ladder(CI)
        d = dict(t=t, line=T["line"], grcode=T["grcode"], name=T["name"], single=T["single"],
                 prem=float(T["prem"].mean()), prem97=float(T["prem"][-1]), latest_paid=float(latest.sum()),
                 true=true_res, cl=float(r_cl), mack_se=se, odp_mean=float(boot.mean()), odp_sd=float(boot.std(ddof=1)),
                 odp_phi=phi, true5=true5, cl5=float(r5), se5=float(se5),
                 inc_ok=inc_ok, true_inc=true_inc, cl_inc=float(r_inc), se_inc=float(se_inc),
                 neg_incr=int((np.diff(C, axis=1)[UP[:, 1:]] < 0).sum()))
        for p in LEVELS:
            d[f"mack_q{p}"] = lognormal_q(r_cl, se, p)
            d[f"odp_q{p}"] = float(np.quantile(boot, p))
        rec.append(d)
    df = pd.DataFrame(rec)

    # pooled line factors (industry benchmark) and gradient boosting
    plf = pooled_line_factors(tri)
    df["pool"] = [reserve_with_factors(T["P"], plf[T["line"]]) for T in tri]
    tr = train_rows(tri, "log")
    tune = gbm_tune(tr)
    best = tune[0][1]
    print("GBM tuned:", best, "cv weighted mse", tune[0][0], "worst", tune[-1][0])
    model = gbm_fit(tr, best)
    sm = smearing(model, tr)
    print("smearing factors range:", min(sm.values()), max(sm.values()))
    df["gbm"] = gbm_project(model, tri, "log")                # main specification (chosen in 00_model_selection.py)
    df["gbm_smear"] = gbm_project(model, tri, "log", sm)      # with retransformation correction (robustness)
    trr = train_rows(tri, "ratio")
    tune_log = gbm_tune(trr)
    df["gbm_ratio"] = gbm_project(gbm_fit(trr, tune_log[0][1]), tri, "ratio")   # raw-ratio target
    # seed stability of the learner
    alt = []
    for s in range(1, 6):
        m = HistGradientBoostingRegressor(loss="squared_error", categorical_features=[1], random_state=SEED + s, **best)
        bs = tr.sample(frac=1.0, replace=True, random_state=SEED + s)
        m.fit(bs[FEATS].values, bs["y"].values, sample_weight=bs["w"].values)
        alt.append(gbm_project(m, tri, "log"))
    df["gbm_bag_sd"] = np.std(np.array(alt), axis=0, ddof=1)
    df["gbm_bag_mean"] = np.mean(np.array(alt), axis=0)

    pickle.dump(dict(df=df, flow=flow, tune=[(float(a), b) for a, b in tune], best=best, best_ratio=tune_log[0][1], smear={f"{k[0]}_{k[1]}": float(v) for k, v in sm.items()}, n_rows=len(tr),
                     levels=LEVELS, B=B, seed=SEED), open(os.path.join(HERE, f"core{TAG}.pkl"), "wb"))
    df.drop(columns=["name"]).to_csv(os.path.join(HERE, f"triangle_level_results{TAG}.csv"), index=False)
    print(df.groupby("line")[["true", "cl", "pool", "gbm", "gbm_smear", "gbm_ratio", "mack_se"]].sum().round(0))
    print(df[["true", "cl", "pool", "gbm", "gbm_smear"]].describe().round(0))


if __name__ == "__main__":
    main()
