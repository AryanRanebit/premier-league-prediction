# Next-9 checkpoint — frozen three-model predictions (games 11-19 per club)

Frozen 2026-11-21, before any of these matches kicked off (verified
2026-11-21T05:54Z; earliest kickoff 2026-11-21T12:30Z). This complements,
not replaces, the season-end evaluation in `../season_end_evaluation/`.

## What is frozen

`next9_2627_predictions.json` — for each of the 90 fixtures (the union of
every club's next 9 announced fixtures after the 2026-11-08 cutoff, all 20
clubs at exactly 10 played):

- full per-model score matrices (the record of record — these ARE the
  predictions; nothing is "recomputed later"),
- 1X2 probabilities, expected goals, top-3 most likely scorelines,
- per-club expected points over its 9 matches per model.

## Models (histories NOT identical across families — read before comparing)

| Key | Model | Training data | Grid |
|---|---|---|---|
| `dc` | Production Dixon-Coles, loaded from the frozen `results/forecast_full_v3.json` (protocol boost=1.0, half-life 730d, asof 2026-11-19 = same 100 matches) | its own full-history weighted fit | 11x11 (goals 0-10) |
| `nn_rich` | NN, repo rich features | 2020-21..2026-11-08, arm B early stop, 5 seeds | 7x7 (6 = 6+) |
| `nn_xg` | NN, rich + rolling xG | identical window to nn_rich (matched control design from `../xg/`) | 7x7 (6 = 6+) |

The points-stretch/draw post-processing illustrations are NOT part of this
comparison.

## Fixture window

Each club's next 9 ANNOUNCED fixtures (source: FotMob announced schedule,
cross-checked 100/100 against football-data for played matches). All clubs
have announced fixtures covering games 11-19; every club's 9th-next fixture
is currently announced for 2027-01-01..2027-01-03. Postponements may move
actual dates: evaluation happens when every club has actually PLAYED 19
matches, verified against football-data/FotMob at that time — not on a fixed
round date. If a fixture in this set is postponed and a different match is
played in its place before a club's 19th game, score the 90 frozen fixtures
that were actually played within each club's first-19 window and report the
count; do not silently substitute.

## Scoring (fixed now)

When all 20 clubs have 19 played:
1. **1X2 log-loss and Brier score** per model over the played frozen
   fixtures (outcome vs p_home/p_draw/p_away).
2. **Exact-scoreline NLL** per model (actual score's cell probability;
   NN cell "6" covers 6+ goals, DC has exact cells to 10).
3. **Points MAE**: |expected next-9 points − actual next-9 points| per club,
   averaged, per model.
Report all three per model; no single headline "accuracy".

Regeneration: `exp_next9_2627.py` reproduces this file bit-for-bit given the
pinned inputs: the football-data CSVs (auto-fetched into `data/` by the
common harness), `experiments/xg/data/pl_xg_matched.csv` (not redistributed;
regenerate with the network retrieval scripts `experiments/xg/download_xg.py`
then `match_xg.py` per `experiments/README.md`), and the frozen DC snapshot
`results/forecast_full_v3.json`. Without the xG CSV the script exits with the
retrieval instructions before writing anything. Verified 2026-11-21: a full
repo-layout regeneration reproduced the committed JSON exactly.
