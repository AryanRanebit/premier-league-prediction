# Experiment rules (autoresearch-style loop, scaled to this project)

Idea borrowed from karpathy/autoresearch: propose a change, run a fixed evaluation,
keep it only if the metric improves, otherwise revert, and log every try.

- Editable: `candidate.py` (CFG) only, via `run_experiment.py "description" key=value`.
- Locked: `evaluate.py`, everything under `src/`, `data/`, `results/`, and the frozen
  2026-27 protocols under `experiments/`. Do not edit them to get a better number.
- Metric: mean scoreline NLL on the tune seasons 2021-22 to 2024-25, lower is better.
- Keep rule: tune-mean improves by at least 0.005 and at most one tune season gets worse.
- 2025-26 is logged but never used to decide. It was used in earlier reports, so it is not
  a clean holdout. The only untouched test is the frozen 2026-27 checkpoint (GW19 / season end).
- Dixon-Coles reference (results/dc_nll.json): 3.0702 on the tune seasons, 2.9112 on 2025-26.
- One run takes about 5 seconds on one CPU core. No GPU, no paid API, no unattended loop:
  an agent or a person runs a fixed list of experiments and every row goes into results.tsv.
