# Data

All match data comes from **football-data.co.uk**, which publishes Premier League
season files (`E0.csv`, seasons 2015-16 through 2026-27) as publicly accessible
downloads at https://www.football-data.co.uk/data.php

The site states **no reuse license**: its only legal page is a liability disclaimer
(https://football-data.co.uk/disclaimer.php, verified 19 Nov 2026) that says nothing
about data reuse. Reuse terms are therefore unresolved. Our posture: attribute the
source, link it, and do **not** redistribute the CSVs - the fetch script below
downloads them from the source instead.

- Fetch everything: `python3 scripts/fetch_data.py` (writes `data/E0_<season>.csv`)
- The CSVs are **not committed** to this repo. Please download them via the script
  and credit football-data.co.uk if you reuse them.
- Snapshot used for the published results: 2026-27 season through **8 Nov 2026**
  (100 matches played). Later snapshots change the current-season forecast.
