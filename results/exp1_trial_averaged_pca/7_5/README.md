# Experiment 1, session 7_5 headline numbers

Preprocessing: detrend (per-neuron linear) -> z-score; truncate to 75 frames; trial-average per area.
Trial mask: |treadmill| > 1.0 dropped; n_clean = 453.
Class composition (clean): {'Clip': 377, 'Monet2': 38, 'Trippy': 38}.

## Per-area headline (all-neurons analysis)

| Area | n_neurons | silhouette (p) | CV accuracy (p) | chance |
|---|---|---|---|---|
| V1 | 5485 | +0.021 (p=0.001) | 0.989 (p=0.001) | 0.832 |
| AL | 414 | -0.018 (p=0.001) | 0.947 (p=0.001) | 0.832 |
| LM | 1262 | +0.031 (p=0.001) | 0.980 (p=0.001) | 0.832 |
| RL | 1033 | +0.011 (p=0.001) | 0.978 (p=0.001) | 0.832 |
