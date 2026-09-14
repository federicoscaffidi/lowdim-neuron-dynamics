# lowdim-neuron-dynamics

PCA-based analysis of how stimulus classes separate in population activity of
mouse visual cortex (areas V1, AL, LM, RL), using one session (`7_5`) of the
MICrONS functional dataset.

> Status (2026-09-15): all numbers below are the values committed before the
> planned correctness review and re-run. They will be refreshed after the
> re-run. Sections marked `<TODO: Federico>` need a decision or a claim only
> the authors can make.

## Headline result

`<TODO: Federico>` — pick which of the committed numbers is the headline and
state what it means. Candidates, straight from the committed CSVs
(session 7_5, all neurons, empirical p = (k+1)/(n_shuffles+1) with 100 shuffles,
so 0.0099 is the floor):

| experiment | quantity | V1 | AL | LM | RL | source |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5-fold CV logistic-regression accuracy, Clip/Monet2/Trippy (null p95) | 0.989 (0.823) | 0.947 (0.804) | 0.980 (0.819) | 0.978 (0.819) | `results/exp1_trial_averaged_pca/7_5/silhouette_scores.csv` |
| 1 | balanced silhouette in top-3 PCs (null p95) | 0.021 (−0.030) | −0.018 (−0.035) | 0.031 (−0.030) | 0.011 (−0.032) | same |
| 2 | max full-feature trajectory distance Monet2↔Trippy, observed / null p95 / p | 19.96 / 21.46 / 0.37 | 5.30 / 6.12 / 0.76 | 9.48 / 10.63 / 0.48 | 8.68 / 9.21 / 0.29 | `results/exp2_stim_trajectories/7_5/trajectory_metrics.csv` |
| 3 | max full-feature distance Cinematic↔sports1m, observed / null p95 / p / onset | 13.39 / 11.52 / 0.0099 / 267 ms | 3.71 / 3.32 / 0.0099 / 267 ms | 6.68 / 5.72 / 0.0099 / 267 ms | 5.36 / 5.06 / 0.0099 / 267 ms | `results/exp3_clip_categories/7_5/trajectory_metrics.csv` |

Neuron counts: V1 5485, AL 414, LM 1262, RL 1033. Clean trials: 453
(Clip 377, Monet2 38, Trippy 38); within Clip: Cinematic 127, sports1m 127,
Rendered 123.

## Quickstart: clean clone → one reproduced number

Target: the V1 classifier accuracy in experiment 1, committed value
`0.988962148962149` (`silhouette_scores.csv`, row `V1,classifier,all_neurons`).

```bash
git clone https://github.com/federicoscaffidi/lowdim-neuron-dynamics.git
cd lowdim-neuron-dynamics
python3.12 -m venv .venv && source .venv/bin/activate     # 3.11 also recorded as working
pip install -r requirements.txt                          # versions are guesses; see file header
```

**Data.** The notebooks read `${MICRONS_DATADIR}/functional/microns_functional.h5`
(~20.6 GB). Source of the file: `<TODO: Federico>` (URL/DOI and the exact
release; nothing in the repo records it). Then:

```bash
export MICRONS_DATADIR=/path/to/dir            # default if unset: ../neuroscience
```

**Run.** The notebooks import `scripts.*` and write `figures/…`, `results/…`
relative to the **repo root**, so the kernel's working directory must be the
repo root (this is VS Code's default for notebooks; it is *not* JupyterLab's).
From the root:

```bash
python -c "
import nbformat, nbclient
nb = nbformat.read('notebooks/experiment1.ipynb', as_version=4)
nbclient.NotebookClient(nb, timeout=-1, resources={'metadata': {'path': '.'}}).execute()
"
```

`<TODO: Federico>` — runtime on your machine, and confirm this command from a
clean clone (it has not been run here: no working environment exists on this
machine).

**Check.**

```bash
grep '^V1,5485,classifier' results/exp1_trial_averaged_pca/7_5/silhouette_scores.csv
# expected (pre-rerun): V1,5485,classifier,0.988962148962149,...
```

Exact equality is only expected with the same numpy/scikit-learn versions;
see the note in `requirements.txt`.

## Repo map

```text
notebooks/
  eda_preproc.ipynb      cross-session screen (14 sessions) + session 7_5 deep dive; figures only
  experiment1.ipynb      trial-averaged PCA per area; silhouette + CV-classifier + shuffle nulls
  experiment2.ipynb      time-resolved trajectories (75 frames) per area × stimulus class
  experiment3.ipynb      experiment-2 pipeline within Clip, labels = Cinematic/sports1m/Rendered
scripts/
  microns_eda.py         loaders (MicronsFunctionalReader + h5py), per-session summaries, EDA plots
  experiment1_utils.py   preprocessing, per-area PCA, silhouette/classifier + nulls, population matching
  experiment2_utils.py   trajectory construction, PCA, distance metrics, bootstrap/null, plots
  experiment3_utils.py   clip-category labels + inventory, catalog plots, cross-area plot
  neuro_palette.py       colours + matplotlib style shared by all figures
  fetch_results.py       restores the deposited .npz caches and verifies them against data/MANIFEST.tsv
results/
  exp1_trial_averaged_pca/7_5/   silhouette_scores.csv, README.md (headline table)
  exp2_stim_trajectories/7_5/    trajectory_metrics.csv, trajectories.npz*, README.md
  exp3_clip_categories/7_5/      trajectory_metrics.csv, clip_inventory.csv, trajectories.npz*, README.md
  */cross_session_summary.csv    one row-set per session run (currently only 7_5)
figures/                 EDA figures at top level; per-experiment figures under figures/exp*/7_5/
data/                    deposit docs (README, MANIFEST.*); raw H5 lives here locally, ignored
convert_svgs.py          SVG → PDF for the three vector figures used in the report (needs cairosvg)
```

`*` = in the external deposit (see `data/README.md`), not in git history after
the Phase 3 cleanup.

## Method (what the code does)

Every notebook operates on session `7_5`. Trials whose mean absolute treadmill
speed exceeds 1.0 are dropped; each neuron's time series is linearly detrended
and z-scored; trials are truncated to 75 frames; neurons are split by cortical
area. Experiment 1 averages each trial to one vector per trial, fits PCA per
area, and scores class separability with a class-balanced silhouette on the top
3 PCs (20 balanced subsamples) and 5-fold stratified cross-validated
multinomial logistic regression on the full neuron matrix, each against a
100-shuffle label-permutation null; controls: every area subsampled to AL's
414 neurons (20 draws) and a `log1p` sensitivity run. Experiment 2 keeps the
time axis: per area and class it builds a trial-averaged 75 × N trajectory,
fits PCA on the stacked trajectories, and measures frame-wise Euclidean
distance between class pairs in the full neuron space and in the top-3 PC
space, with 1000 class-stratified bootstrap envelopes, a 100-shuffle null on
the maximum distance, an onset-latency estimate, an equal-population control,
and a Clip-trial subsampling control. Experiment 3 runs the experiment 2
pipeline on Clip trials only, with labels from the movie's content category.
All randomness is derived from `RANDOM_SEED = 42`.

## Limitations

- One session (`7_5`) of fourteen was analysed; the `cross_session_summary.csv`
  files and the "cross-session" figures contain that session only.
- The committed notebooks were edited after their last execution (import cells
  and the cells that write `trajectories.npz` have no execution count), so the
  committed outputs are from an earlier notebook state. A full re-run is
  planned.
- `SESSION`, the treadmill threshold (1.0) and the 75-frame truncation are
  hard-coded per notebook.
- The raw dataset is not distributed and its source is not recorded here.
- `<TODO: Federico>` — scientific limitations (what the analysis cannot
  conclude) belong here; not derivable from the code.

## References

- The MICrONS Consortium. *Functional connectomics spanning multiple areas of
  mouse visual cortex*. Nature, 2025.
- P. Dayan and L. F. Abbott. *Theoretical Neuroscience*. MIT Press, 2001.
- S. Schneider, J. H. Lee, M. W. Mathis. *Learnable latent embeddings for
  joint behavioural and neural analysis*. Nature 617, 360–368 (2023).
- `microns_datacleaner` (Python package used for loading): `<TODO: Federico>`
  citation / repository URL.

## Licence

GPL-3.0 — see `LICENSE`.
