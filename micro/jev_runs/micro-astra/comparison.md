# SMAC-Hard map smoke tests

Base opponent, shared team vision, fixed-step simulation, abilities disabled. Three episodes per map are exploratory results, not reliable win-rate estimates.

Maps marked dev are developer scenarios. Archived maps and dev scenarios use explicit attack-move extensions where the upstream base script has no implementation.

| Map | Policy | Wins/completed | Results | Mean kills | Mean survivors | JEV USD | Astra plans |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 3m | [jev](3m/jev/summary.json) | 0/3 | loss, loss, loss | 2 | 0 | 0.01831 | 1 |
| 3s_vs_3z | [jev](3s_vs_3z/jev/summary.json) | 0/3 | loss, loss, loss | 1.67 | 0 | 0.03835 | 1 |

One Astra map plan is reused in every JEV request across all episodes. JEV cost estimates exclude Codex account usage; Astra token counts are in each planner_result.
