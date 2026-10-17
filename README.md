# StopRisk

Finite-evidence envelopes for a specified shutdown-based AI-control model.

**Status:** research manuscript and tested prototype, not a published or accepted
paper. This is a model-conditional calculation over a fixed tested attack menu.
It does not certify a real deployment or cover untested attacks.

The manuscript is by Shivam Gupta, Independent Researcher. It includes proofs,
four figures, three tables, a 240,000-dataset synthetic study, exact coverage
enumeration, and a secondary reconstruction from public monitoring aggregates.
No new LLM calls were performed; synthetic results and externally generated
aggregate results are clearly separated.

## What is implemented

- Exact Clopper-Pearson rectangles for decisive probability and conditional harm
  share, including zero decisive observations.
- Lower and upper Bellman envelopes for history-adaptive selection among fixed
  stationary attack kernels; one confidence event covers every horizon.
- A separately calibrated marginal-hazard baseline.
- A block evaluator for the optimistic affine envelope, with a bounded-switch
  characterization and stable geometric calculations for tiny hazards.

## Reproduction

Use Python 3.11 or later. The executed package versions are pinned in
`requirements-lock.txt`; Tectonic builds the PDF. No credentials are required.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/run_study.py
python scripts/make_figures.py
tectonic manuscript/main.tex
```

The final manuscript is `StopRisk_Shivam_Gupta.pdf`. All figures are regenerated
from the supplied CSVs or stated analytical kernels. Timing and runtime fields
will change when the study is rerun; seeded count tables should not change with
the same NumPy version. A fresh manuscript build writes `manuscript/main.pdf`.

## Study result

Both methods have conservative coverage. Neither uniformly gives narrower
intervals. In the rare synthetic condition, 73.72% of datasets with 1,000
observations per attack give zero plug-in risk when the modeled risk is 1.89%.
The public-data case does not establish an advantage from switching: its first
attacker is strongest at every tested horizon. It also does not validate the
IID sampling or effective-shutdown assumptions.

## Files

| Path | Purpose |
| --- | --- |
| `src/stoprisk/core.py` | Counts, confidence rectangles, reference recursion |
| `src/stoprisk/fast.py` | Optional optimistic-envelope accelerator |
| `tests/` | Independent rational, LP, enumeration, decimal checks |
| `scripts/run_study.py` | Offline primary study and public count reconstruction |
| `scripts/make_figures.py` | Four figures and exploratory accelerator comparison |
| `data/` | Original attributed CSV and explicitly derived counts |
| `results/synthetic_*.npz` | Generated categorical count tables, not model outputs |
| `results/*.csv` | Numerical outcomes, exact coverage, public analysis |
| `PROTOCOL.txt`, `DEVIATIONS.txt` | Prospective internal protocol and amendments |
| `manuscript/` | Editable LaTeX source |
| `SHA256SUMS.txt` | Final file hashes |

## Assumptions and limits

Evaluation draws are IID within an attack, sample sizes and the candidate family
are fixed independently of the evaluation outcomes, and an action is selected
before the next outcome. Kernels are stationary and transfer to deployment.
Catch and harm are correctly distinguished, mutually exclusive, absorbing events;
a catch immediately prevents future harm. A fixed score threshold is not a hard
audit budget. The rectangular bound need not be sharp for stationary kernels.
The accelerator uses floating point, not rigorous interval arithmetic.

## Attribution

The original public aggregate CSV is from Andre Williams,
[trusted-monitoring-factorial-research](https://github.com/Andre-Williams22/trusted-monitoring-factorial-research),
commit `3e420c9a8180163eafaf35db17da0973d49fbd4a`,
`logs/monitor_exp/identity_all_viable.csv`. It is unmodified and its Apache-2.0
license is retained in `data/UPSTREAM_LICENSE.txt`. Derived counts are marked as
such. Underlying raw model transcripts were unavailable in the checkout. The
primary model formula is established in AI-control and competing-risk literature;
see the manuscript's references for the exact scope of the proposed contribution.

## License

Original code: MIT, copyright 2026 Shivam Gupta. Upstream data retains its original
license and attribution. The manuscript is not covered by the code license.
