# lowdim-neuron-dynamics

Study the dynamics of neurons in the MICrONS functional dataset through the lens of dimensionality-reduction algorithms (PCA, UMAP, t-SNE, CEBRA).

## Layout

```
notebooks/   four Jupyter notebooks (EDA + three experiments)
scripts/     helper modules imported by the notebooks
data/        local data (gitignored; see Setup)
figures/     generated plots
results/     per-session CSVs and headline READMEs
report/      LaTeX final report
```

## Notebooks

All notebooks operate on session `7_5` (closest to the cohort median by neuron count) and are session-swappable: change the `SESSION` constant at the top of any notebook to replicate on another session.

- [`notebooks/eda_preproc.ipynb`](notebooks/eda_preproc.ipynb): exploratory data analysis. Cross-session overview, per-session deep dive (structure, stimuli, neural responses, behavior, preprocessing diagnostics), running-outlier removal, and recommended preprocessing for the dimensionality-reduction phase. Backed by [`scripts/microns_eda.py`](scripts/microns_eda.py).
- [`notebooks/experiment1.ipynb`](notebooks/experiment1.ipynb): **Experiment 1, trial-averaged PCA.** Per-cortical-area PCA on trial-mean responses, with class-balanced silhouette in the top three PCs and 5-fold cross-validated logistic regression accuracy on the full feature matrix as separability metrics. Includes an equal-population control (subsample every area to AL's neuron count) and a `log(1+x)` sensitivity check. Backed by [`scripts/experiment1_utils.py`](scripts/experiment1_utils.py).
- [`notebooks/experiment2.ipynb`](notebooks/experiment2.ipynb): **Experiment 2, time-resolved trajectories.** Per-area, per-stimulus trial-averaged trajectories with the time axis preserved. Stacked-stimulus PCA, full-feature and top-3 PC distance metrics, bootstrap envelopes (B=1000), shuffle null on max pairwise distance (n=100), equal-population control, and a Clip-trial subsampling sanity check. Resolves the Monet2 / Trippy collapse seen in Experiment 1 by keeping within-trial dynamics. Backed by [`scripts/experiment2_utils.py`](scripts/experiment2_utils.py).
- [`notebooks/experiment3.ipynb`](notebooks/experiment3.ipynb): **Experiment 3, Clip-category trajectories.** The Experiment 2 pipeline applied within the Clip class, using the three content subcategories (`Cinematic`, `sports1m`, `Rendered`) as labels. Includes a Part 1 catalog (per-category exemplars, frame-strips, and a 240-panel grid of all unique Clip movies) so the analysis labels are interpretable. Backed by [`scripts/experiment3_utils.py`](scripts/experiment3_utils.py).

## Report

The final write-up lives in [`report/report.tex`](report/report.tex) (compiled to `report.pdf`). Body is two pages; figures and the title page are extra.

## Setup

```bash
pip install -r requirements.txt
```

The `microns.h5` file is ~19 GB and lives outside the repo (the `.gitignore` excludes `*.h5`). Two ways to point the notebooks at it:

1. Set an environment variable to the directory containing the H5:
   ```bash
   export MICRONS_DATADIR=/path/to/dir/containing/h5
   ```
2. Or accept the default `DATADIR = "../neuroscience"` used by the notebooks (assumes the repo is cloned beside a `neuroscience/` directory containing the H5).

`MicronsFunctionalReader` looks for the file at `{DATADIR}/functional/microns_functional.h5`. The notebooks' data-layout cell automatically creates a symlink from that path to a `microns.h5` (or `microns_functional.h5`) found elsewhere in `DATADIR`, so the 19 GB file does not need to be moved.
