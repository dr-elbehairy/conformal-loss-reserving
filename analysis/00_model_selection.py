# -*- coding: utf-8 -*-
"""Time-respecting choice of the learner specification, using information available at year-end 1997 only.
Fit every candidate on the triangle as of 1996 (the 1997 diagonal removed) and score the one-step-ahead
forecast of the 1997 diagonal. No cell of the lower triangle is touched."""
import os, json, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("core", os.path.join(HERE, "01_core.py")); core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
I = 10; K = 8          # last known calendar index in the validation design (0-based: i + j <= 8)
LINES = core.LINES; FEATS = core.FEATS

def rows_asof(tri, kind):
    out = []
    for t, T in enumerate(tri):
        C = T["P"]; pr = T["prem"]; lc = LINES.index(T["line"]); size = np.log(pr[:I - 1].mean())
        for j in range(K):
            rr = np.arange(0, K - j)                       # rows with (i, j+1) known as of 1996
            num, den = C[rr, j + 1].sum(), C[rr, j].sum()
            for i in rr:
                own = np.log(max((num - C[i, j + 1]) / (den - C[i, j]), 1e-3)) if len(rr) > 1 else np.nan
                prev = np.log(C[i, j] / C[i, j - 1]) if j >= 1 else np.nan
                y = np.log(C[i, j + 1] / C[i, j]) if kind == "log" else C[i, j + 1] / C[i, j] - 1.0
                out.append((t, T["grcode"], j, lc, i, np.log(pr[i]), C[i, j] / pr[i], prev, own, size, y, C[i, j] / den, C[i, j], C[i, j + 1]))
    return pd.DataFrame(out, columns=["t", "grcode"] + FEATS + ["y", "w", "c0", "c1"])

def targets(tri):
    X = []; meta = []
    for t, T in enumerate(tri):
        C = T["P"]; pr = T["prem"]; lc = LINES.index(T["line"]); size = np.log(pr[:I - 1].mean())
        for i in range(0, I - 1):
            j = K - i                                       # cell on the 1996 diagonal; target is (i, j+1) on 1997
            if j < 0 or j > I - 2: continue
            rr = np.arange(0, K - j)
            own = np.log(max(C[rr, j + 1].sum() / C[rr, j].sum(), 1e-3)) if len(rr) > 0 else np.nan
            prev = np.log(C[i, j] / C[i, j - 1]) if j >= 1 else np.nan
            X.append((j, lc, i, np.log(pr[i]), C[i, j] / pr[i], prev, own, size)); meta.append((t, i, j, C[i, j], C[i, j + 1], own))
    return np.array(X, dtype=float), pd.DataFrame(meta, columns=["t", "i", "j", "c0", "c1", "own"])

tri, _ = core.build_panel()
X, M = targets(tri)
res = {}
# chain ladder as of 1996 (own factors); cells with no own factor (lag 9) are skipped for every candidate
ok = M.own.notna().values
M = M[ok].reset_index(drop=True); X = X[ok]
pred = {"chain ladder": M.c0 * np.exp(M.own)}
# pooled line factors as of 1996
plf = {}
for l in LINES:
    num = np.zeros(K); den = np.zeros(K)
    for T in tri:
        if T["line"] != l: continue
        for j in range(K):
            rr = np.arange(0, K - j); num[j] += T["P"][rr, j + 1].sum(); den[j] += T["P"][rr, j].sum()
    plf[l] = num / den
pred["pooled line factors"] = np.array([M.c0[k] * plf[tri[M.t[k]]["line"]][M.j[k]] for k in range(len(M))])
for kind, label in (("log", "GBM log"), ("log", "GBM log + retransformation"), ("ratio", "GBM ratio")):
    tr = rows_asof(tri, kind)
    prm = core.gbm_tune(tr)[0][1]
    m = core.gbm_fit(tr, prm)
    raw = m.predict(X)
    fac = np.exp(raw) if kind == "log" else 1.0 + raw
    if "retransformation" in label:
        sm = core.smearing(m, tr)
        fac = fac * np.array([sm.get((int(x[1]), int(x[0])), 1.0) for x in X])
    pred[label] = M.c0.values * np.clip(fac, 0.5, 20.0)
out = {}
for k, p in pred.items():
    d = pd.DataFrame(dict(t=M.t, inc_hat=np.asarray(p) - M.c0, inc=M.c1 - M.c0, c0=M.c0)).groupby("t").sum()
    rel = (d.inc_hat - d.inc).abs() / d.c0            # error in next-year paid, relative to paid to date
    out[k] = dict(mean_abs_rel=float(rel.mean()), median_abs_rel=float(rel.median()),
                  agg_ratio=float(d.inc_hat.sum() / d.inc.sum()), rmse_rel=float(np.sqrt(((d.inc_hat - d.inc) ** 2 / d.c0 ** 2).mean())))
    print(f"{k:32s}", {a: round(b, 4) for a, b in out[k].items()})
json.dump(out, open(os.path.join(HERE, "model_selection.json"), "w"), indent=1)
