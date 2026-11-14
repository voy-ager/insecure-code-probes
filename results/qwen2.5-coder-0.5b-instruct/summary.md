| method | in-dist AUROC [CI] | LOCO AUROC [CI] | transfer AUROC [CI] | within-area transfer AUROC | gap in-dist - transfer [CI] | AUPRC in-dist | AUPRC transfer | n_pos/n_neg in-dist; transfer | n_redraw/n_boot in-dist; transfer |
|---|---|---|---|---|---|---|---|---|---|
| probe (nested in-dist / selected layer on transfer) | 0.736 [0.694, 0.787] | 0.611 [0.565, 0.665] | 0.638 [0.512, 0.748] | 0.605 (7 areas) | +0.133 [0.014, 0.280] | 0.724 (chance 0.500) | 0.445 (chance 0.315) | 288/288; 69/150 | 0/1000; 0/1000 |
| layer-0 probe | 0.797 [0.744, 0.853] | 0.609 [0.560, 0.658] | 0.568 [0.418, 0.724] | 0.575 (7 areas) | +0.230 [0.070, 0.386] | 0.775 (chance 0.500) | 0.380 (chance 0.315) | 288/288; 69/150 | 0/1000; 0/1000 |
| lexical | 0.819 [0.763, 0.885] | 0.615 [0.562, 0.679] | 0.603 [0.482, 0.746] | 0.648 (7 areas) | +0.216 [0.074, 0.357] | 0.815 (chance 0.500) | 0.433 (chance 0.315) | 288/288; 69/150 | 0/1000; 0/1000 |
| length | 0.507 [0.489, 0.527] | 0.504 [0.488, 0.521] | 0.570 [0.457, 0.661] | 0.544 (7 areas) | -0.063 [-0.148, 0.038] | 0.514 (chance 0.500) | 0.356 (chance 0.315) | 288/288; 69/150 | 0/1000; 0/1000 |
| shuffled-label probe | 0.516 [0.504, 0.530] | n/a | 0.557 [0.415, 0.709] | 0.594 (7 areas) | -0.041 [-0.196, 0.103] | 0.519 (chance 0.500) | 0.347 (chance 0.315) | 288/288; 69/150 | 0/1000; 0/1000 |
| topic prior | n/a | n/a | 0.210 [0.103, 0.339] | 0.119 (7 areas) | n/a | n/a | 0.218 (chance 0.315) | n/a; 69/150 | n/a; 0/1000 |
| prompt-risky prior | n/a | n/a | 0.743 [0.621, 0.853] | 0.754 (7 areas) | n/a | n/a | 0.491 (chance 0.315) | n/a; 69/150 | n/a; 0/1000 |
| judge | 0.515 [0.489, 0.545] | n/a | 0.470 [0.358, 0.603] | 0.507 (7 areas) | +0.045 [-0.099, 0.167] | 0.533 (chance 0.500) | 0.309 (chance 0.315) | 288/288; 69/150 | 0/1000; 0/1000 |

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
