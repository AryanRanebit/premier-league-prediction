# Next-9 checkpoint — frozen three-model predictions (games 11-19 per club)

Frozen 2026-11-21, before any of these matches kicked off (verified
2026-11-21T05:54Z; earliest kickoff 2026-11-21T12:30Z). This complements,
not replaces, the season-end evaluation in `../season_end_evaluation/`.

## Protocol version

**v2 is the scoring record.** `next9_2627_predictions_v2.json` holds
full-precision probabilities. `next9_2627_predictions.json` (v1, 5-decimal
rounding, which produced exact zeros in tiny DC cells) is preserved
unchanged for provenance but superseded for scoring. v2 changes:
full precision; `dc_binned7x7` added per fixture (DC 11x11 aggregated onto
the common 7x7 event space, bin 6 = 6+ goals); input pinning by SHA256;
postponement scoring rules fixed below.

## What is frozen

For each of the 90 fixtures (the union of every club's next 9 announced
fixtures after the 2026-11-08 cutoff, all 20 clubs at exactly 10 played):

- full per-model score matrices at full precision (the record of record —
  these ARE the predictions; nothing is "recomputed later"),
- `dc_binned7x7`: DC aggregated onto the common 7x7 grid for scoreline NLL,
- 1X2 probabilities, expected goals, top-3 most likely scorelines,
- per-club expected points over its 9 matches per model.

All model cells are strictly positive (DC: exp-link bivariate Poisson; NN:
softmax over 49 cells). If a future regeneration ever yielded a zero cell,
scoring clips at eps=1e-9 before log, uniformly for all models (predeclared,
not post-hoc).

## Models (histories NOT identical across families — read before comparing)

| Key | Model | Training data | Grid |
|---|---|---|---|
| `dc` | Production Dixon-Coles, loaded from the frozen `results/forecast_full_v3.json` (protocol boost=1.0, half-life 730d, asof 2026-11-19 = same 100 matches) | its own full-history weighted fit | 11x11 (goals 0-10) |
| `nn_rich` | NN, repo rich features | 2020-21..2026-11-08, arm B early stop, 5 seeds | 7x7 (6 = 6+) |
| `nn_xg` | NN, rich + rolling xG | identical window to nn_rich (matched control design from `../xg/`) | 7x7 (6 = 6+) |

The points-stretch/draw post-processing illustrations are NOT part of this
comparison.

## Fixture window and sources

Each club's next 9 ANNOUNCED fixtures. Sources:
- Fixtures/schedule: FotMob league JSON,
  `https://www.fotmob.com/api/data/leagues?id=47&ccode3=IND&season=2026/2027`,
  fetched 2026-11-21; every club's 9th-next fixture currently announced for
  2027-01-01..2027-01-03.
- Played-match cross-check: football-data.co.uk E0 CSVs,
  `https://www.football-data.co.uk/mmz4281/<season>/E0.csv` (100/100 played
  matches match FotMob at the cutoff).
- Pre-match verification: all 90 fixtures unplayed as of 2026-11-21T05:54Z
  (FotMob); earliest kickoff 2026-11-21T12:30Z (Man City-Fulham).

## Input pinning (no live-CSV reproduction claims)

SHA256 of every input file is in `INPUT_HASHES.txt`. football-data CSVs
update in-season, and the vendor xG extract is not redistributed, so a
future bit-exact regeneration requires the pinned private input snapshot
(held with the review materials, hashes matching INPUT_HASHES.txt) plus
`results/forecast_full_v3.json`. Without the xG CSV the script exits with
the retrieval instructions before writing anything.

## Scoring (fixed now)

Evaluation happens per fixture as the frozen 90 complete; final evaluation
when every club has actually PLAYED 19 matches, verified live at that time.

1. **1X2 log-loss and Brier score** per model over completed frozen
   fixtures (outcome vs p_home/p_draw/p_away).
2. **Exact-scoreline NLL** per model on the COMMON 7x7 event space:
   `nn_rich`/`nn_xg` natively, `dc` via `dc_binned7x7`; actual scores with
   6+ goals map to bin 6 for all three models. (The raw 11x11 DC matrix is
   not NLL-comparable to the NN 7x7 grids; the binned version is.)
3. **Points MAE**: |expected next-9 points − actual points| per club, where
   actual points are earned ONLY in that club's frozen 9 fixtures, per
   model.

Postponements: score the fixed 90 frozen fixtures as they complete and
report coverage (n/90). If clubs reach 19 played while frozen fixtures
remain unplayed, report a partial matched-fixture evaluation and wait for
the remaining frozen fixture(s); never substitute a different match, and
never compare expected points for 9 frozen fixtures against actual points
from a different set of fixtures.

Report all three metrics per model plus coverage; no single headline
"accuracy".

## Regeneration

`exp_next9_2627.py` regenerates the v2 file. Verified 2026-11-21: a full
repo-layout run reproduced v1 bit-exactly, and v2 rounded to 5 decimals
equals v1 exactly (worst cell diff 0.0), with all matrices summing to 1 and
strictly positive (min cell ~1.4e-15).
