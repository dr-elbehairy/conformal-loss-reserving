"""Reproduces every number, table input and figure of the paper. Run from the repository root: python run_all.py"""
import subprocess, sys, os
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
for s in (["00_download_data.py"], ["04_validate_benchmark.py"], ["00_model_selection.py"], ["01_core.py"], ["01_core.py", "2007"],
          ["02_conformal.py"], ["05_out_of_time.py"], ["03_figures.py"]):
    print(">>", " ".join(s)); subprocess.run([sys.executable, os.path.join(A, s[0])] + s[1:], check=True)
