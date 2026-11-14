| method | in-dist AUROC [CI] | LOCO AUROC [CI] | transfer AUROC [CI] | within-area transfer AUROC | gap in-dist - transfer [CI] | AUPRC in-dist | AUPRC transfer | n_pos/n_neg in-dist; transfer | n_redraw/n_boot in-dist; transfer |
|---|---|---|---|---|---|---|---|---|---|
| probe (nested in-dist / selected layer on transfer) | 0.799 [0.752, 0.855] | 0.660 [0.619, 0.712] | 0.780 [0.665, 0.873] | 0.789 (7 areas) | +0.056 [-0.045, 0.191] | 0.766 (chance 0.500) | 0.594 (chance 0.322) | 288/288; 107/225 | 0/1000; 0/1000 |
| layer-0 probe | 0.807 [0.752, 0.864] | 0.624 [0.571, 0.680] | 0.640 [0.520, 0.762] | 0.646 (7 areas) | +0.167 [0.032, 0.301] | 0.794 (chance 0.500) | 0.423 (chance 0.322) | 288/288; 107/225 | 0/1000; 0/1000 |
| lexical | 0.819 [0.763, 0.885] | 0.615 [0.562, 0.679] | 0.759 [0.668, 0.844] | 0.763 (7 areas) | +0.060 [-0.033, 0.173] | 0.815 (chance 0.500) | 0.543 (chance 0.322) | 288/288; 107/225 | 0/1000; 0/1000 |
| length | 0.507 [0.489, 0.527] | 0.504 [0.488, 0.521] | 0.493 [0.369, 0.626] | 0.514 (7 areas) | +0.014 [-0.125, 0.144] | 0.514 (chance 0.500) | 0.337 (chance 0.322) | 288/288; 107/225 | 0/1000; 0/1000 |
| shuffled-label probe | 0.495 [0.481, 0.510] | n/a | 0.446 [0.320, 0.587] | 0.441 (7 areas) | +0.049 [-0.078, 0.193] | 0.502 (chance 0.500) | 0.284 (chance 0.322) | 288/288; 107/225 | 0/1000; 0/1000 |
| topic prior | n/a | n/a | 0.106 [0.043, 0.194] | 0.114 (7 areas) | n/a | n/a | 0.209 (chance 0.322) | n/a; 107/225 | n/a; 0/1000 |
| prompt-risky prior | n/a | n/a | 0.798 [0.700, 0.891] | 0.795 (7 areas) | n/a | n/a | 0.557 (chance 0.322) | n/a; 107/225 | n/a; 0/1000 |
| judge | 0.606 [0.566, 0.653] | n/a | 0.668 [0.522, 0.793] | 0.649 (7 areas) | -0.061 [-0.184, 0.102] | 0.591 (chance 0.500) | 0.571 (chance 0.322) | 288/288; 107/225 | 0/1000; 0/1000 |

Bootstrap: percentile cluster bootstrap, n_boot = 1000, seed = 0.
AUPRC is shown with its chance level (positive fraction).
Pooled LOCO mixes out-of-fold scores of different leave-one-CWE-out models, so score scales differ across CWEs; see per_cwe.md for per-CWE values.

Notes:
- judge gap: n_redraw/n_boot = 0/1000
- judge: no scores_judge_loco_*.csv (LOCO)
- layer-0 probe LOCO: n_redraw/n_boot = 0/1000
- layer-0 probe gap: n_redraw/n_boot = 0/1000
- length LOCO: n_redraw/n_boot = 0/1000
- length gap: n_redraw/n_boot = 0/1000
- lexical LOCO: n_redraw/n_boot = 0/1000
- lexical gap: n_redraw/n_boot = 0/1000
- probe (nested in-dist / selected layer on transfer) LOCO: n_redraw/n_boot = 0/1000
- probe (nested in-dist / selected layer on transfer) gap: n_redraw/n_boot = 0/1000
- shuffled-label probe gap: n_redraw/n_boot = 0/1000
- shuffled-label probe: no scores_probe_shuffled_loco_*.csv (LOCO)
