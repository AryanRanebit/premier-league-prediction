# autoresearch-style NN experiments

Challenger branch work, separate from the frozen forecast and the Dixon-Coles main prediction.
See `program.md` for the rules and `results.tsv` for every run (kept and discarded).

Run: `python3 autoresearch_pl/run_experiment.py "what I tried" l2=1e-3`
Needs the data from `scripts/fetch_data.py`. The baseline row reproduces the main-branch
rich NN exactly (per-season NLL 3.016 / 2.990 / 3.226 / 3.148 / 2.951, five-season mean 3.066).

## Result of the first 10 experiments (selection on 2021-22 to 2024-25 only)

| | tune mean NLL | 2025-26 (already seen) |
|---|---|---|
| Dixon-Coles (results/dc_nll.json) | 3.0702 | 2.9112 |
| NN baseline (main) | 3.0953 | 2.9509 |
| NN kept: L2 3e-3 + hidden 24/12 | 3.0787 | 2.9239 |

Read this honestly:
- The kept config still does not beat Dixon-Coles on the tune seasons (3.0787 vs 3.0702).
- 1 of 10 tries was kept. It combined two ideas after both were seen to help a little on
  their own (smaller net alone gave 3.0853 but failed the "at most one season worse" rule).
  That makes it a post-hoc pick, so its gain (-0.017) is not an independent estimate.
- Gains are small next to the season-to-season spread (NLL ranges 2.99 to 3.23 across seasons).
- The 2025-26 column moved the same way, but it is not a test set.
- Nothing here changes the frozen 2026-27 forecast. A fair final check is the frozen GW19 /
  season-end protocol, scored once, with this config declared in advance.
