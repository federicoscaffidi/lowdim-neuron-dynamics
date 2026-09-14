# Option 3 — session 7_5 headline numbers

Preprocessing: detrend → z-score; truncate to 75 frames; running-outlier filter (|treadmill| > 1.0) → 453 clean trials.
Class composition (clean): {'Clip': 377, 'Monet2': 38, 'Trippy': 38}.

## Headline: Monet2↔Trippy by area (full-feature metric)

| Area | n_neurons | max distance | null p95 | p-value | onset (ms) |
|---|---|---|---|---|---|
| V1 | 5485 | 19.960 | 21.464 | 0.366 | 7067 |
| AL | 414 | 5.305 | 6.121 | 0.762 | n/a |
| LM | 1262 | 9.478 | 10.627 | 0.475 | n/a |
| RL | 1033 | 8.675 | 9.211 | 0.287 | n/a |
