# Experiments — November 2026 research round

One day of local experiments on top of the v3 baseline (branch
`review-v3-baseline`, commit `89c4838`). **Nothing here is adopted.** The
production forecast (`results/forecast_full_v3.json`, Dixon-Coles) and the v3
NN code are unchanged. Every experiment below ran locally on free data and
free compute; this directory is the review bundle.

## What's here

| Directory | Question | Verdict |
|---|---|---|
| `manager_ctx/` | Do manager, schedule, or style features add skill? | No reliable improvement — **not adopted** |
| `euro_stylec/` | Do European-workload or lagged high-recovery features add skill? | Small, inconsistent — **not adopted** |
| `halflife/` | Is the 730-day recency half-life right? | Mixed (position MAE vs NLL trade off) — **730d kept** |
| `xg/` | Do rolling xG features add skill? | Small, mixed (−0.04 MAE / +0.007 NLL) — **not adopted** |
| `forecast_2627_gw10/` | What does the NN predict for 2026-27 after 10 games/club? | Experimental snapshot — **not adopted** |
| `postprocessing/` | Can the forecast's points spread/draw rate be adjusted to match history? | Reference-fit illustrations — **not validated, not adopted** |

## Two protocols — do not conflate

- **GW19 backtest (primary):** for each test season 2021-22..2025-26, train
  only on matches up to that season's official gameweek-19 cutoff, predict
  the remaining matches, simulate the final table 4,000 times. 5 seeds.
  Arm B = chronological 80/20 split with early stopping (best validation
  checkpoint). The 250-epoch arm is known to overfit and is reference only.
- **First-100 / GW10 snapshot:** train through 2026-11-08 (the first 100
  matches of 2026-27, every club exactly 10 played), simulate the remaining
  280 announced fixtures. This is a forecast, not an accuracy score.

Metrics: **scoreline NLL** (negative log-probability of the actual scoreline;
lower = better) and **position MAE** (mean absolute error of expected finish
vs actual, in table places; lower = better).

## Headline numbers (GW19 arm B, 5-season averages)

- Full-history rich baseline: big-six MAE 1.598, all-20 MAE 1.864, NLL 3.034
- +manager/schedule (ctx): 1.502 / 1.808 / 3.030 — hints, not replicated
- +European workload + high recovery (euroC): 1.545 / 1.877 / 3.026
- Recency grid (730d control): 1.598 / 1.864 / 3.034; 120d: 1.598 / 1.801 /
  3.104 (positions better, probabilities worse)
- **xG (matched short-history design, both arms train 2020-21 onward):**
  control 1.551 / 1.830 / 3.061, +xG 1.514 / 1.791 / 3.068. The
  training-window effect alone is as large as the feature effect — the
  matched control is what makes this readable.

Consistent reading across all of these: on the same five reused validation
seasons, deltas are small and sign-inconsistent. These are **not improvement
claims**. Reusing the same five seasons for every keep/reject decision is
itself a form of adaptive overfitting, which is why the next evaluation
should be a frozen pre-registered forecast scored on future matches.

## Post-processing illustrations (postprocessing/)

The after-10-games NN forecast produces a more bunched table than recent
real seasons (simulated champion ~80 pts vs 85-100 in reality; draw rate
23.8%). Two global, match-consistent post-processes were tested:

- `exp_drawcal_2627.py` — multiplies draw cells by one global factor solved
  to a target season draw rate. Matches totals/draws by construction; does
  not widen the spread.
- `exp_spreadcal_2627.py` / `exp_spreadcal5_2627.py` — one global
  strength-stretch parameter on goal supremacy, solved to the 2025-26 table
  spread (15.4 pts std) or the 2021-22..2025-26 five-season mean spread
  (17.97). Moves points (champion 80.1 → 83.3 / 88.4), barely moves ranks.

**These are illustrations fit to a chosen historical reference, not learned
model behavior and not validated predictions.** They are kept separate from
every forecast artifact.

## Data provenance and licensing

- Match results/statistics: football-data.co.uk (see `scripts/fetch_data.py`,
  `DATA.md`).
- Manager stints: compiled from Wikipedia ("List of Premier League
  managers"), `manager_ctx/manager_stints.csv` — 521 dated stints, full
  coverage of all 8,560 team-matches 2015-16..2026-27.
- European fixtures: `euro_stylec/pl_clubs_european_fixtures_2015_2027.csv` —
  938 club-matches compiled from ESPN's API and cross-checked against
  FotMob; played matches only, the 2021-12-09 Spurs-Rennes awarded game is
  excluded, domestic cups excluded.
- FotMob-derived stats (possession won final 3rd, match xG): **raw data is
  not redistributed here** — free-to-view vendor data without an established
  reuse license. `xg/download_xg.py` retrieves match-level xG from FotMob's
  public endpoints for local research use; the census (2,380/2,380 matches
  covered 2020-21 onward, zero missing) is in `xg/SUMMARY.md`. Coverage
  starts 2020-21; the abandoned 2023-12-16 Bournemouth-Luton match has no
  xG (the replayed fixture does).
- Squad possession (styleA feature): `common/fbref_possession.json` —
  season possession percentages compiled from FBref squad pages on
  2026-11-21 (9 seasons, 2017-18..2025-26).
- FBref removed all advanced stats (including xG) on 2026-01-20 after its
  data provider terminated access; historical advanced data was deleted.
  Do not rely on old FBref xG scraping guides.

## Reproduction

1. Fetch football-data CSVs per `DATA.md` / `scripts/fetch_data.py`.
2. Retrieve vendor data locally (see each experiment's header + scripts).
3. Each experiment directory has a harness (`exp_*.py`) with usage in its
   docstring; shared infrastructure is in `common/`. Result JSONs from the
   original runs are in each `results/` directory. All runs use 5 seeds and
   rng 42, so results are exactly reproducible.
4. Python stdlib + numpy only; see `requirements.txt`.

## Honest limitations (read before citing any number)

- All validation used the same five seasons (2021-22..2025-26), repeatedly.
  Small average deltas may be noise or adaptive tuning to those seasons.
- Early stopping addresses the visible training overfit; it does not prove
  overfitting is gone.
- The xG provider/model identity behind FotMob's served values is
  unverified; don't mix xG sources.
- Schedule features count PL matches only (cups excluded) unless stated;
  European workload counts played matches only.
- Promoted clubs have no prior-season top-flight values; seasonal features
  mean-fill them by design.
- Probabilities are model outputs, not verified calibration.
