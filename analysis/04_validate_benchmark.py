# -*- coding: utf-8 -*-
"""Validation of the chain-ladder, Mack and ODP-bootstrap code against two published examples.
(1) Taylor and Ashe triangle: Mack (1993) prints an overall chain-ladder reserve of 18,681 (thousands) and a standard error of 13 %.
(2) Automatic facultative general liability triangle: England and Verrall (2002, Tables 1 and 33) print a chain-ladder reserve of
    52,135, a bootstrap mean of 53,210 and a bootstrap prediction error of 19,267 (36 %), against an analytic 18,193 (35 %)."""
import os, json, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("core", os.path.join(HERE, "01_core.py")); core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
TA = [[357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
      [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046],
      [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405],
      [310608, 1108250, 776189, 1562400, 272482, 352053, 206286],
      [443160, 693190, 991983, 769488, 504851, 470639],
      [396132, 937085, 847498, 805037, 705960],
      [440832, 847631, 1131398, 1063269],
      [359480, 1061648, 1443370],
      [376686, 986608],
      [344014]]
EV = [[5012, 3257, 2638, 898, 1734, 2642, 1828, 599, 54, 172],
      [106, 4179, 1111, 5270, 3116, 1817, -103, 673, 535],
      [3410, 5582, 4881, 2268, 2594, 3479, 649, 603],
      [5655, 5900, 4211, 5500, 2159, 2658, 984],
      [1092, 8473, 6271, 6333, 3786, 225],
      [1513, 4932, 5257, 1233, 2917],
      [557, 3463, 6926, 1368],
      [1351, 5596, 6165],
      [3133, 2262],
      [2063]]
out = {}
for name, inc, pub in (("taylor_ashe", TA, dict(reserve=18681000, se_pct=13)), ("england_verrall_2002", EV, dict(reserve=52135, boot_mean=53210, boot_se=19267, boot_pct=36, analytic_se=18193))):
    C = np.zeros((10, 10))
    for i, r in enumerate(inc): C[i, :len(r)] = np.cumsum(r)
    f, proj, res, se = core.chain_ladder(C)
    boot, phi = core.odp_bootstrap(C, np.random.default_rng(1), nb=20000)
    out[name] = dict(published=pub, factors=[round(float(x), 3) for x in f], reserve_total=round(float(res)), mack_se_total=round(float(se)),
                     mack_cv=round(float(se / res), 4), odp_boot_mean=round(float(boot.mean())), odp_boot_sd=round(float(boot.std(ddof=1))),
                     odp_phi=round(float(phi), 1), odp_boot_cv=round(float(boot.std(ddof=1) / res), 4))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "benchmark_validation.json"), "w"), indent=1)
