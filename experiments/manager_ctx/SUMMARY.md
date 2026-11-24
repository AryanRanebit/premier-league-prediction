# Manager / schedule / tactical-style features — experimental tests (21 Nov 2026)

Four feature sets tested on top of the repo's rich NN features (commit 89c4838),
all local, no GitHub changes, free data and compute. PRIMARY comparison is the
early-stopping arm of the 5-season GW19 backtest (the 250-epoch arm is known to
overfit and is reported as reference only).

## Feature sets (all causal, no invented values)

1. **ctx = manager + schedule** (7 per team):
   manager: log tenure days, log matches under current manager, new-manager
   (<=90 days) flag, log prior PL matches managed, manager career PL PPG.
   schedule: days since previous match, matches in previous 14 days.
2. **styleA = season-aggregate prior style**: prior-PL-season possession %.
   For a match in season N the value is the completed season N-1 aggregate.
   No current-season or future aggregate is used anywhere.
3. **styleB = pre-match rolling style**: rolling-10 shots share, SOT share,
   corners share (for / (for + against)) from football-data match stats.
   These are dominance proxies derived from existing data, NOT measured
   possession/pressing.
4. **styleAB** = styleA + styleB.

## Data provenance and coverage (auditable)

- Managers: Wikipedia "List of Premier League managers", fetched 21 Nov 2026.
  Snapshot: wiki_mgrs.html; parsed: manager_stints.csv (521 dated stints incl.
  caretakers). Strict date-interval containment covers 8,560/8,560 team-matches
  2015-16..2026-27, zero gaps/fallbacks. Cross-validated (Arteta, Klopp,
  Guardiola; Tifo-consistent 2026-27 entries: De Zerbi Spurs, Frank Palace,
  Andrews Brentford, Alonso Chelsea; Pochettino to Man Utd 18 Nov 2026
  confirmed via manutd.com / AP / AFP / BBC).
  Constraints: appointment date = in-charge date (boundary-day matches choose
  one side by rule); manager history before 2015-16 is outside the window
  (left-censored); manager PPG is computed from raw match results
  (3/1/0 points), not official tables, so points deductions never enter it.
- Schedule features count PREMIER LEAGUE MATCHES ONLY. Cup and European
  fixtures are not in the data, so "matches in last 14 days" understates true
  congestion for clubs in Europe - this is a known partial proxy, not the
  full fixture-congestion signal.
- Possession: FBref squad possession tables 2017-18..2025-26 (9 seasons),
  fetched 21 Nov 2026. Snapshots: fb_1718.md .. fb_2526.md; parsed:
  fbref_possession.json. Feature available for matches from 2018-19 onward;
  2015-16..2017-18 training matches are mean-filled. Per test season, 17/20
  squads have a prior-season value; the 3 promoted clubs are mean-filled
  (their prior season was outside the PL). Only the Poss column was
  extractable from FBref pages via the fetcher; deeper columns (attacking-third
  touches, progressive distance) rendered empty - NOT confirmed absent from
  FBref, just not retrievable in this run. FBref pressing/pressure pages:
  UNVERIFIED (repeated fetch timeouts / a 403), not confirmed absent.

## Results - PRIMARY: GW19 cutoffs, early-stopping arm B (matched)

| mode | Big Six pos err | all-20 pos err | scoreline NLL |
|---|---|---|---|
| rich (current) | 1.598 | 1.864 | 3.034 |
| ctx (manager+schedule) | 1.502 | 1.808 | 3.030 |
| styleA (prior poss %) | 1.574 | 1.853 | 3.030 |
| styleB (rolling shares) | 1.577 | 1.898 | 3.019 |
| styleAB | 1.555 | 1.875 | 3.022 |

Per-season NLL for styleB vs rich: better in 4 of 5 seasons
(2.968/3.022, 2.966/2.971, 3.281/3.249 worse, 2.963/3.006, 2.915/2.924).
All differences are small against season-to-season spread; none is an
established improvement.

## Results - reference arms (250 epochs, known overfit)

GW19, arm A: rich 1.391/1.795/3.534 | ctx 1.381/1.992/3.782 |
styleA 1.438/1.881/3.539 | styleB 1.477/1.898/3.508 | styleAB 1.315/1.900/3.623

100-match-cutoff protocol (250 epochs): rich 2.75/2.53/3.629 -> ctx
2.66/2.56/3.848. Control re-run reproduced the repo's recorded v3 numbers
exactly.

2026-27 first-10 preseason test (250 epochs): rich 44/100, NLL 3.60, table
4.12, points 3.37 -> ctx 44/100, NLL 4.86, table 4.76, points 4.07.

## Verdict (corrected framing)

Under the matched early-stopping comparison, no feature set shows a reliable
or material improvement. The manager+schedule set has the largest table-error
gains under that arm (Big Six 1.50 vs 1.60, all-20 1.81 vs 1.86) with NLL
unchanged; rolling shot/corner shares improve NLL slightly in 4 of 5 seasons.
These are hints, not established gains: they do not replicate consistently
across protocols and seasons, and under the (overfit) 250-epoch recipe the
same features hurt probability calibration.

This is a negative-ablation result ONLY: it shows these particular features
did not add measurable skill in this setup. It does NOT show that manager or
tactical effects are already captured by form - that mechanism was not tested
and the earlier wording claiming it is retracted.

Recommendation: do not adopt any of these features now. Keep the bundle as
documented experiments. Honest next steps if he wants more: (a) true fixture
congestion needs European/cup match dates (not freely in our current data);
(b) per-match xG/pressing would need a paid or scraping-heavy source;
(c) a "matches since manager change" interaction with form is a cleaner test
of the new-manager-bounce hypothesis than raw features.

## Bundle contents
- exp_context_gw100.py, exp_context_gw19.py, first10_ctx.py - manager+schedule harnesses
- exp_style_gw19.py - tactical-style harness (modes styleA/styleB/styleAB)
- manager_stints.csv, wiki_stints_raw.json, wiki_mgrs.html - manager data + snapshot
- fbref_possession.json, fb_1718.md..fb_2526.md - possession data + snapshots
- ctx_results/*.json - all result JSONs (ctx, styleA, styleB, styleAB, rich rerun, first10)
- ctx_comparison.png - chart of the primary early-stopping comparison
