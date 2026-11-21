# Distribution-free prediction intervals for loss reserves

Replication code for the paper *Distribution-Free Prediction Intervals for Chain-Ladder and Machine-Learning Loss Reserves: Evidence for Solvency II and the IFRS 17 Risk Adjustment* (M. Elbehairy, Suez Canal University, 2026).

The paper asks whether the confidence levels attached to loss reserves are attained in realised run-off, at two valuation dates (year-end 1997 and year-end 2007), and whether split conformal calibration across insurers repairs them.

## Scripts (folder `analysis/`)

| Script | Purpose |
|---|---|
| `00_download_data.py` | Downloads the twelve Schedule P files from the CAS data page into `analysis/data/` and `analysis/data2007/`. |
| `04_validate_benchmark.py` | Checks the chain-ladder, Mack and bootstrap code against the examples printed in Mack (1993) and England and Verrall (2002). |
| `00_model_selection.py` | Chooses the specification of the gradient-boosting learner with a 1996-to-1997 holdout that uses no later data. |
| `01_core.py` | Sample selection, chain ladder and Mack standard error, over-dispersed Poisson bootstrap, pooled line factors, pooled gradient boosting. Run with the argument `2007` for the second cohort. |
| `02_conformal.py` | First cohort: leave-one-company-out split conformal bounds, coverage and quantile-score evaluation with cluster bootstrap, short design of 1992, robustness checks. Writes `results.json`. |
| `05_out_of_time.py` | Second cohort: benchmark coverage, transfer of the 1997 multipliers, short design of 2002, pooled design. Writes `results_2007.json`. |
| `03_figures.py` | Figures 1 to 5 (written to `figures/`). |

`analysis/results.json`, `analysis/results_2007.json`, `analysis/model_selection.json`, `analysis/benchmark_validation.json` and the two `triangle_level_results*.csv` files are the outputs used in the paper.

## Reproduce

```
pip install -r requirements.txt
python run_all.py
```

Random seed 20261121. Run time is a few minutes on a laptop.

## Data

The data are not redistributed here. They were accessed from the loss reserving data page of the Casualty Actuarial Society:
https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p

The accident-year 1988 to 1997 files were compiled by Glenn G. Meyers and Peng Shi. The accident-year 1998 to 2007 files were compiled for the CAS by S&P Global Market Intelligence.

## Licence

MIT for the code.
