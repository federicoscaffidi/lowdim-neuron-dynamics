# Experiment 2, session 7_5 headline numbers

Preprocessing: detrend → z-score; truncate to 75 frames; running-outlier filter (|treadmill| > 1.0) → 453 clean trials.
Class composition (clean): {'Clip': 377, 'Monet2': 38, 'Trippy': 38}.
Null: 1000 label permutations of the max-over-frames distance; p_holm = Holm over the 12 (area × pair) tests of the full-feature metric. `max distance` is the raw value (noise floor sqrt(N·(1/n_A+1/n_B))); `cv distance` is the split-half bias-corrected value. Onset = first frame at which the observed distance exceeds the null's 95th percentile; only reported when p_holm < 0.05.

## Headline: Monet2↔Trippy by area (full-feature metric)

| Area | n_neurons | max distance | cv distance | null p95 | p | p_holm | onset (ms) |
|---|---|---|---|---|---|---|---|
| V1 | 5485 | 19.960 | 12.449 | 21.460 | 0.338 | 1.000 | n/a |
| AL | 414 | 5.305 | 3.163 | 6.170 | 0.776 | 1.000 | n/a |
| LM | 1262 | 9.478 | 5.924 | 10.617 | 0.533 | 1.000 | n/a |
| RL | 1033 | 8.675 | 5.934 | 9.283 | 0.308 | 1.000 | n/a |
