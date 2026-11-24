# European fixture dataset for PL clubs — coverage and verification note

File: `pl_clubs_european_fixtures_2015_2027.csv` (built 21 Nov 2026).
Scope: every UEFA Champions League, Europa League and Conference League match
(including qualifying / play-off rounds) involving a club that played in the
Premier League that season, 2015-16 through 2026-27. 938 rows:
902 played, 35 scheduled (future, 2026-27), 1 awarded-not-played.
Columns: season, competition, date, home, away, stage, pl_club, source, status.

## Sources (all verified live 21 Nov 2026)

- UCL + UEL: ESPN soccer core API (unofficial but public).
  Season/type/event enumeration, e.g.
  `https://sports.core.api.espn.com/v2/sports/soccer/leagues/uefa.champions/seasons/2021/types/1/events`
  5,698 events pulled for 2015..2026 seasons; 855 involve English clubs.
  ESPN includes qualifying rounds for seasons 2015-16..2019-20 only.
- Tottenham 2020-21 UEL qualifying (3 matches, the only English-club
  qualifiers ESPN misses from 2020-21 onward): tottenhamhotspur.com
  announcements (17 Sep Lokomotiv Plovdiv A, 24 Sep Shkendija A,
  1 Oct Maccabi Haifa H).
- UECL competition proper 2021-22..2026-27: FotMob league fixture pages
  (`https://www.fotmob.com/en-GB/leagues/10216/fixtures/conference-league?season=YYYY/YYYY`).
- UECL August play-off legs (not in FotMob competition listings), 5 ties,
  each verified against named sources:
  West Ham–Viborg 18/25 Aug 2022; Hibernian–Aston Villa 23/31 Aug 2023
  (Sky Sports/ESPN match reports); Chelsea–Servette 22/29 Aug 2024
  (chelseafc.com); Crystal Palace–Fredrikstad 21/28 Aug 2025 (cpfc.co.uk);
  Tromso–Brighton 20/27 Aug 2026 (BBC Sport). Wikipedia qualifying-phase
  articles corroborate round dates.

## Verification checks performed

- Participation matrix cross-checked per season: 2015-16..2026-27, every PL
  club's UCL/UEL/UECL participation listed; clubs not listed are verified
  non-participants, not missing data. 2026-27: no English club in UCL/UEL
  qualifying (Wikipedia qualifying articles); Brighton entered UECL at the
  play-off round.
- Awarded / cancelled / unplayed: ESPN status checked for all 855
  English-club events (800 FT, 11 AET, 9 pens, 32 scheduled; zero cancelled
  or awarded). One unplayed awarded match exists in the UECL data:
  Tottenham v Rennes, 9 Dec 2021 — never played, UEFA awarded Rennes a 3-0
  win (uefa.com, 20 Dec 2021). Marked `awarded_not_played`; it must NOT
  count as physical workload. FotMob presents it as a normal finished 0-3,
  so this needs the explicit flag.
- Postponed/rescheduled matches: source dates are actual played dates.
  Spot-verified: Arsenal v PSV (postponed from 8 Sep to 20 Oct 2022 after
  Queen Elizabeth II's death) appears on 20 Oct 2022.
- Duplicates: none; each match is one row. When both clubs are PL clubs
  (11 matches: e.g. Liverpool–Tottenham 2019 UCL final, Chelsea–Arsenal
  2019 UEL final) the single row lists both in `pl_club` (slash-separated).
- Dates are UTC calendar dates from the sources.

## Explicit exclusions (not in scope, not missing)

FA Cup, EFL Cup, Community Shield, UEFA Super Cup, FIFA Club World Cup
(Chelsea participated June–July 2025). Domestic cups would need their own
source pass if workload features should include them.

## Licensing caveat

All sources are free to view. That does not establish a reuse or
redistribution license: FotMob and ESPN expose this data without a stated
bulk license. Treat the CSV as a local research artifact, not something to
republish as data.

## Known boundaries

- Pre-play-off UECL qualifying rounds contain no English clubs in any season.
- UEFA match dates before 2015-16 are outside the model's training window.
- FotMob season-aggregate pressing metric lives in
  `fotmob_poss_won_final3rd_by_season.csv` (see parent report for details).
