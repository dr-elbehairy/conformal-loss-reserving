# -*- coding: utf-8 -*-
"""Downloads the twelve line-of-business files from the loss reserving data page of the Casualty Actuarial Society.
The files are not redistributed in this repository. Source page:
https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p"""
import os, subprocess, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://www.casact.org/sites/default/files/"
OLD = {l: f"2021-04/{l}_pos.csv" for l in ("ppauto", "comauto", "wkcomp", "othliab", "prodliab", "medmal")}
NEW = {"ppauto": "2026-03/ppauto_pos98-07%20%281%29.csv", "comauto": "2026-03/comauto_pos_98-07.csv", "wkcomp": "2026-03/wkcomp_pos_98-07.csv",
       "othliab": "2026-03/othliab_pos_98-07.csv", "prodliab": "2026-03/prodliab_pos_98-07.csv", "medmal": "2026-03/medmal_pos_98-07.csv"}
def get(url, path):
    if os.path.exists(path): return
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-client)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f: f.write(r.read())
    except Exception:
        # some Python installations have no root certificates; fall back to the system curl, which verifies TLS normally
        subprocess.run(["curl", "-sS", "-L", "-A", "Mozilla/5.0 (research-client)", "-o", path, url], check=True)
    print("downloaded", os.path.basename(path))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True); os.makedirs(os.path.join(HERE, "data2007"), exist_ok=True)
for l, u in OLD.items(): get(BASE + u, os.path.join(HERE, "data", f"{l}_pos.csv"))
for l, u in NEW.items(): get(BASE + u, os.path.join(HERE, "data2007", f"{l}_pos_98-07.csv"))
