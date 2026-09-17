# Experiment 3, session 7_5 headline numbers

Pure within-Clip analysis: 3 categories × 75 frames, detrend → z-score, treadmill QC threshold = 1.0.
Class composition (clean): {'Cinematic': 127, 'sports1m': 127, 'Rendered': 123}; 237 unique clips.
Null: 1000 permutations of category labels *between clips* (repeats of one clip stay together); p_holm = Holm over the 12 (area × pair) tests of the full-feature metric. `max distance` is the raw value (noise floor sqrt(N·(1/n_A+1/n_B))); `cv distance` is the split-half bias-corrected value. Onset = first frame at which the observed distance exceeds the null's 95th percentile; only reported when p_holm < 0.05.

## Cinematic↔sports1m (full-feature metric)

| Area | n_neurons | max distance | cv distance | null p95 | p | p_holm | onset (ms) |
|---|---|---|---|---|---|---|---|
| V1 | 5485 | 13.387 | 6.331 | 13.007 | 0.014 | 0.098 | n/a |
| AL | 414 | 3.710 | 2.021 | 3.982 | 0.201 | 0.699 | n/a |
| LM | 1262 | 6.680 | 2.909 | 6.721 | 0.060 | 0.336 | n/a |
| RL | 1033 | 5.363 | 2.558 | 5.715 | 0.315 | 0.699 | n/a |

## Cinematic↔Rendered (full-feature metric)

| Area | n_neurons | max distance | cv distance | null p95 | p | p_holm | onset (ms) |
|---|---|---|---|---|---|---|---|
| V1 | 5485 | 12.580 | 4.592 | 13.083 | 0.195 | 0.699 | n/a |
| AL | 414 | 4.748 | 3.191 | 3.993 | 0.001 | 0.012 | 400 |
| LM | 1262 | 6.803 | 3.675 | 6.818 | 0.056 | 0.336 | n/a |
| RL | 1033 | 6.016 | 2.774 | 5.717 | 0.007 | 0.056 | n/a |

## Rendered↔sports1m (full-feature metric)

| Area | n_neurons | max distance | cv distance | null p95 | p | p_holm | onset (ms) |
|---|---|---|---|---|---|---|---|
| V1 | 5485 | 12.560 | 6.979 | 13.023 | 0.175 | 0.699 | n/a |
| AL | 414 | 5.200 | 3.735 | 4.034 | 0.001 | 0.012 | 400 |
| LM | 1262 | 7.339 | 3.770 | 6.718 | 0.002 | 0.020 | 400 |
| RL | 1033 | 6.153 | 2.902 | 5.705 | 0.006 | 0.054 | n/a |

