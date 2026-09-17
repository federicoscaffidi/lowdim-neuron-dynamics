# lowdim-neuron-dynamics

Does the identity of a visual stimulus show up as separable structure in the
population activity of mouse visual cortex, and does that depend on the
cortical area? This repository asks the question with linear methods (PCA,
Euclidean trajectory distances, a linear decoder) on one session of the
MICrONS functional dataset: 8,194 neurons in V1, AL, LM and RL responding to
natural movie clips (Clip) and two parametric noise stimuli (Monet2, Trippy).

## Results in one paragraph

All numbers are from session `7_5` and are reproduced by the notebooks in this
repository (Python 3.11.14, pinned packages; every CSV under `results/` was
regenerated on 2026-09-16). Stimulus class is linearly decodable from the
trial-averaged population activity in every area (5-fold cross-validated
logistic regression, V1 0.989, LM 0.980, RL 0.978, AL 0.947 against a
majority-class chance of 0.832; every area beats its label-permutation null,
Holm-adjusted p = 0.004). The same class structure is barely visible in the
first three principal components (class-balanced silhouette between −0.018
and +0.031, above a null that is itself negative), because class identity
accounts for a small share of trial-to-trial variance: the top three PCs
explain 16.4 % of V1's variance and 95 % needs 309 of 453 components. Keeping
the time axis (75 frames, 7.5 Hz) and testing the distance between
class-mean trajectories against a permutation null of the maximum over
frames, no area separates the two parametric stimuli, Monet2 and Trippy
(smallest Holm-adjusted p = 0.12 for any pair in any area), and the ordering
of raw distances between pairs is the ordering of their sampling-noise
floors, not a result. Within the Clip class, with labels permuted between the
237 unique clips rather than between the 377 trials, three of twelve
area × category-pair tests survive Holm correction (AL Cinematic↔Rendered,
AL Rendered↔sports1m, LM Rendered↔sports1m; onset 400 ms), none in V1.

## Headline numbers

Empirical p = (k+1)/(n_shuffles+1) with 1000 shuffles, so 0.001 is the floor;
`p_holm` is Holm-adjusted over the 4 areas of a metric (experiment 1) or the
12 area × pair tests of a metric (experiments 2–3).

**Experiment 1** — trial-averaged activity, per area
(`results/exp1_trial_averaged_pca/7_5/silhouette_scores.csv`):

| quantity | V1 (5485) | AL (414) | LM (1262) | RL (1033) |
| --- | --- | --- | --- | --- |
| 5-fold CV logistic-regression accuracy, Clip/Monet2/Trippy | 0.989 | 0.947 | 0.980 | 0.978 |
| … null p95 / p / p_holm | 0.826 / 0.001 / 0.004 | 0.806 / 0.001 / 0.004 | 0.819 / 0.001 / 0.004 | 0.817 / 0.001 / 0.004 |
| balanced silhouette in top-3 PCs | +0.021 | −0.018 | +0.031 | +0.011 |
| … null p95 / p / p_holm | −0.030 / 0.001 / 0.004 | −0.033 / 0.001 / 0.004 | −0.031 / 0.001 / 0.004 | −0.031 / 0.001 / 0.004 |
| accuracy at 414 neurons (median of 20 subsamples; AL unsubsampled) | 0.973 | 0.947 | 0.969 | 0.966 |
| silhouette at 414 neurons (median of 20) | +0.004 | −0.018 | +0.023 | +0.013 |

Majority-class chance for the accuracy is 0.832 (377/453). In the
equal-population block 20/20 subsamples are significant per area at
`p_holm = 0.03` (100 shuffles each). A `log1p` variance-stabilising
transform changes no verdict (accuracies 0.960–0.989, silhouettes
−0.017 to +0.022).

**Experiment 2** — time-resolved distance between class-mean trajectories,
full neuron space, Monet2↔Trippy
(`results/exp2_stim_trajectories/7_5/trajectory_metrics.csv`):

| area | max raw distance | bias-corrected | null p95 | p | p_holm |
| --- | --- | --- | --- | --- | --- |
| V1 | 19.96 | 12.45 | 21.46 | 0.34 | 1.00 |
| AL | 5.30 | 3.16 | 6.17 | 0.78 | 1.00 |
| LM | 9.48 | 5.92 | 10.62 | 0.53 | 1.00 |
| RL | 8.68 | 5.93 | 9.28 | 0.31 | 1.00 |

No pair in any area reaches `p_holm < 0.05` on the full-feature metric
(smallest: RL Clip↔Trippy, p = 0.010, p_holm = 0.12; V1 Clip↔Trippy,
p = 0.029, p_holm = 0.32). No onset is therefore reported.

**Experiment 3** — same pipeline within Clip, labels = content category,
labels permuted between the 237 unique clips
(`results/exp3_clip_categories/7_5/trajectory_metrics.csv`), full-feature
metric, `p / p_holm` per cell:

| pair | V1 | AL | LM | RL |
| --- | --- | --- | --- | --- |
| Cinematic↔sports1m | 0.014 / 0.098 | 0.201 / 0.699 | 0.060 / 0.336 | 0.315 / 0.699 |
| Cinematic↔Rendered | 0.195 / 0.699 | **0.001 / 0.012** | 0.056 / 0.336 | 0.007 / 0.056 |
| Rendered↔sports1m | 0.175 / 0.699 | **0.001 / 0.012** | **0.002 / 0.020** | 0.006 / 0.054 |

Three of twelve tests survive Holm (bold); their onsets are all at frame 3
(400 ms). Bias-corrected maximum distances for those three: AL
Cinematic↔Rendered 3.19, AL Rendered↔sports1m 3.74, LM Rendered↔sports1m
3.77 (raw 4.75 / 5.20 / 7.34).

The top-3-PC rows of both files are significant almost everywhere; they are a
visualisation check only — the PCA is fit on the class-mean trajectories being
compared — and are not reported here.

Neuron counts: V1 5485, AL 414, LM 1262, RL 1033. Clean trials: 453 of 464
(Clip 377, Monet2 38, Trippy 38); within Clip: Cinematic 127, sports1m 127,
Rendered 123, from 237 unique clips (6 shown 10×, 86 shown 2×).

## Quickstart: clean clone → one reproduced number

Target: the V1 classifier accuracy in experiment 1, committed value
`0.988962148962149` (`silhouette_scores.csv`, row `V1,5485,classifier,…,all_neurons`).

```bash
git clone https://github.com/federicoscaffidi/lowdim-neuron-dynamics.git
cd lowdim-neuron-dynamics
python3.11 -m venv .venv && source .venv/bin/activate     # 3.10+ required (microns_datacleaner)
pip install -r requirements.txt                          # pinned from the environment of the re-run
```

**Data.** The notebooks read `${MICRONS_DATADIR}/functional/microns_functional.h5`
(20,638,694,845 bytes, 14 sessions); `MICRONS_DATADIR` defaults to `data`,
the git-ignored folder inside the repo, so with no configuration the file
lands at `data/functional/microns_functional.h5`. The `microns_datacleaner` package that
the loaders wrap (0.2.1.7) downloads this file from

    https://huggingface.co/datasets/NeuroBLab/MICrONS/resolve/main/microns.h5

(`microns_datacleaner/downloader.py`, `download_functional_data`); HF dataset
revision `79c7c55fec8484ebffd1cef67cfa433e63f32a03` at the time of writing.
This is the file all committed results were produced from.

The first code cell after the constants in `notebooks/eda_preproc.ipynb`
downloads it automatically if it is missing (no CAVE credentials needed) and
checks the byte count. The package downloader cannot resume, so on a flaky
connection prefer:

```bash
mkdir -p data/functional
curl -L -C - -o data/functional/microns_functional.h5 \
  https://huggingface.co/datasets/NeuroBLab/MICrONS/resolve/main/microns.h5
```

To keep the file elsewhere (e.g. shared between repositories), set
`MICRONS_DATADIR=/path/to/dir` before starting the kernel; the notebooks
then read `/path/to/dir/functional/microns_functional.h5`.

**Run.** The notebooks import `scripts.*` and write `figures/…`, `results/…`
relative to the **repo root**; the first code cell of every notebook
`chdir`s up one level if the kernel was started inside `notebooks/`, so both
VS Code and JupyterLab work. From the root:

```bash
python -c "
import nbformat, nbclient
nb = nbformat.read('notebooks/experiment1.ipynb', as_version=4)
nbclient.NotebookClient(nb, timeout=-1, resources={'metadata': {'path': '.'}}).execute()
"
```

The committed outputs were produced by running the notebooks interactively
(VS Code) in the same environment; the 1000-shuffle classifier nulls make
experiment 1 the slow one (order of an hour on a laptop; the exact time was
not recorded).

**Check.**

```bash
grep '^V1,5485,classifier.*,all_neurons,' results/exp1_trial_averaged_pca/7_5/silhouette_scores.csv
# expected: V1,5485,classifier,0.988962148962149,0.8035396825396826,...,all_neurons,0.0039960039960039,4.0,,
```

Exact equality is only expected with the pinned numpy/scikit-learn versions.

## Repo map

```text
notebooks/
  eda_preproc.ipynb      data download; cross-session screen (14 sessions) + session 7_5 deep dive; figures only
  experiment1.ipynb      trial-averaged PCA per area; silhouette + CV-classifier + shuffle nulls
  experiment2.ipynb      time-resolved trajectories (75 frames) per area × stimulus class
  experiment3.ipynb      experiment-2 pipeline within Clip, labels = Cinematic/sports1m/Rendered
scripts/
  microns_eda.py         loaders (MicronsFunctionalReader + h5py), per-session summaries, EDA plots
  experiment1_utils.py   preprocessing, per-area PCA, silhouette/classifier + nulls, Holm, population matching
  experiment2_utils.py   trajectory construction, PCA, distance metrics, shuffle null, split-half distance, plots
  experiment3_utils.py   clip-category labels + inventory, catalog plots, cross-area plot
  neuro_palette.py       colours + matplotlib style shared by all figures
  convert_svgs.py        SVG → PDF for the three vector figures used in the report (needs cairosvg)
results/
  exp1_trial_averaged_pca/7_5/   silhouette_scores.csv, README.md (headline table)
  exp2_stim_trajectories/7_5/    trajectory_metrics.csv, README.md
  exp3_clip_categories/7_5/      trajectory_metrics.csv, clip_inventory.csv, README.md
  */cross_session_summary.csv    one row-set per session run (currently only 7_5)
figures/
  eda/                   cross-session screen + session 7_5 deep dive (numbered by EDA section)
  eda/presentation/      slide versions (svg/png/pdf) of three EDA figures
  eda/stim_examples/     one raw frame per Clip category (svg)
  exp*/7_5/              per-experiment figures, numbered by notebook part; slide versions in presentation/
data/                    git-ignored; the raw H5 is downloaded to data/functional/ (see Quickstart)
```

Experiments 2 and 3 also write a `trajectories.npz` cache (14 MB each) next
to their CSV; nothing reads it back, so it is git-ignored and regenerated by
running the notebook.

## Method (what the code does)

Every notebook operates on session `7_5`. Trials whose mean absolute treadmill
speed exceeds 1.0 are dropped (11 of 464; this lowers the trial-wise
correlation between population response and running from −0.39 to −0.22);
each neuron's time series is linearly detrended (photobleaching, ≈45 %
decline over the session) and z-scored; trials are truncated to 75 frames;
neurons are split by cortical area.

Experiment 1 averages each trial to one vector per trial, fits PCA per area,
and scores class separability with a class-balanced silhouette on the top 3
PCs (20 balanced subsamples of 38 trials per class) and 5-fold stratified
cross-validated multinomial logistic regression on the full neuron matrix,
each against a 1000-shuffle label-permutation null (100 inside the
equal-population control). Controls: every area subsampled to AL's 414
neurons (20 draws) and a `log1p` sensitivity run.

Experiment 2 keeps the time axis: per area and class it builds a
trial-averaged 75 × N trajectory, fits PCA on the stacked trajectories, and
measures frame-wise Euclidean distance between class pairs in the full neuron
space and in the top-3 PC space. Significance per pair: the observed maximum
over frames against a 1000-permutation null of the maximum, so the 75 frames
are corrected for by construction (max-T; it controls the same family-wise
error rate as Bonferroni's 0.05/75 but is less conservative); Holm over the
12 area × pair tests of each metric; onset latency = first frame whose
observed distance exceeds the null's 95th percentile, reported only for
Holm-significant pairs. Because the raw distance between two noisy class
means carries a floor of `sqrt(N·(1/n_A + 1/n_B))` in z-scored units (for V1,
≈17 for Monet2↔Trippy at 38/38 trials, ≈12.6 for Clip↔Monet2 at 377/38), a
split-half bias-corrected distance (`⟨mean_A¹ − mean_B¹, mean_A² − mean_B²⟩`
over 20 random splits, unbiased for the squared distance) is reported next
to it and drawn as the dashed curve in the figures. Controls: equal
population (414 neurons, 20 draws) and Clip-trial subsampling (377 → 38
trials, 20 draws; the raw Clip↔Monet2 distance rises from 14.6 to 20.1 while
the bias-corrected one moves from 9.9 to 10.5).

Experiment 3 runs the experiment 2 pipeline on Clip trials only, with labels
from the movie's content category (`short_movie_name` in the H5), and
permutes labels between clips rather than between trials, since repeats of
one clip are not independent trials. All randomness is derived from
`RANDOM_SEED = 42`.

## Limitations

- One session (`7_5`) of fourteen was analysed; the `cross_session_summary.csv`
  files and the "cross-session" figures contain that session only. All
  fourteen sessions are recordings of the **same animal** (the MICrONS volume
  is one mouse), so a multi-session run would test consistency across
  recordings, not across animals.
- The three stimulus classes are not controlled for low-level image
  statistics (mean luminance, RMS contrast, motion energy). Any class
  separation reported here is consistent with the areas encoding such
  features; nothing here supports "naturalness" or "content category" as the
  encoded variable. The frames are in the H5 (`/videos/`), so this control is
  feasible; it is not implemented.
- "Cinematic" in experiment 3 is 8 films (127 trials); a category difference
  is not separable from a film-set difference with this design. Category
  results also rest on AL, the smallest population (414 neurons), where the
  noise floor is lowest.
- Cross-area magnitude comparisons are limited to the equal-population
  controls, which compute no null per subsample; their rows in the CSVs are
  descriptive. The all-neurons distances differ across areas mostly through
  `N` (5485 vs 414 neurons), not through the brain.
- The linear decoder shows that class identity is recoverable; it does not
  locate the encoding in a low-dimensional subspace, and no supervised or
  nonlinear embedding was run. PCA is unsupervised, so a weak silhouette in
  its first components is not a contradiction of the decoder result.
- The notebooks were run top to bottom on 2026-09-16 with the pinned
  packages; `SESSION`, the treadmill threshold (1.0) and the 75-frame
  truncation are constants set per notebook.
- The raw dataset is not distributed with the repository (20.6 GB); see the
  Data section for the source.

## References

- The MICrONS Consortium. *Functional connectomics spanning multiple areas of
  mouse visual cortex*. Nature, 2025.
- P. Dayan and L. F. Abbott. *Theoretical Neuroscience*. MIT Press, 2001.
- S. Schneider, J. H. Lee, M. W. Mathis. *Learnable latent embeddings for
  joint behavioural and neural analysis*. Nature 617, 360–368 (2023).
- V. Buendia, *Microns-DataCleaner* 0.2.1.7 (PyPI; since renamed
  *Microns-Combiner*): `MicronsFunctionalReader` and the functional-data
  downloader used here.

## Licence

GPL-3.0 — see `LICENSE`.
