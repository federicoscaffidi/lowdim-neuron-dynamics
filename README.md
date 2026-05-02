# lowdim-neuron-dynamics

Study the dynamics of neurons in the functional dataset through the lens of dimensionality reduction algorithms, like PCA, UMAP, t-SNE and CEBRA.

## Notebooks

- [`ai_prototype/analysis.ipynb`](ai_prototype/analysis.ipynb) — PCA / eigenspectrum / principal angles on session `7_4`.
- [`eda_microns.ipynb`](eda_microns.ipynb) — Exploratory data analysis on session `7_5` (closest-to-median by neuron count). Cross-session overview, per-session deep-dive (structure, stimuli, neural responses, behavior, preprocessing diagnostics), running-outlier removal, and recommended preprocessing for the dim-reduction phase. Backed by the [`microns_eda`](microns_eda.py) helper module.
- [`option2_pca.ipynb`](option2_pca.ipynb) — **PCA phase, Option 2.** Trial-averaged PCA per cortical area on session `7_5`, with balanced silhouette in top-3 PCs and 5-fold cross-validated logistic regression accuracy on the full trial-averaged matrix as separability metrics. Includes equal-population control (subsample to AL's neuron count) and `log(1+x)` sensitivity check. Session-swappable: change the `SESSION` constant at the top to replicate on another session. Backed by the [`option2_pca_utils`](option2_pca_utils.py) helper module.

Design and plan documents:

- [`docs/specs/2026-04-30-microns-eda-design.md`](docs/specs/2026-04-30-microns-eda-design.md)
- [`docs/plans/2026-04-30-microns-eda-implementation.md`](docs/plans/2026-04-30-microns-eda-implementation.md)
- [`docs/specs/2026-05-02-pca-design.md`](docs/specs/2026-05-02-pca-design.md)
- [`docs/plans/2026-05-02-pca-implementation.md`](docs/plans/2026-05-02-pca-implementation.md)

## Setup

```bash
pip install -r requirements.txt
```

The `microns.h5` file is large (~19 GB) and lives outside the repo (the `.gitignore` excludes `*.h5`). Two options:

1. Point the notebook at your existing copy by setting an environment variable:

   ```bash
   export MICRONS_DATADIR=/path/to/dir/containing/h5
   ```

2. Or accept the default `DATADIR = "../neuroscience"` in `eda_microns.ipynb` (the notebook expects this repo to be cloned beside a `neuroscience/` folder containing the H5).

`MicronsFunctionalReader` looks for the file at `{DATADIR}/functional/microns_functional.h5`. The notebook's data-layout cell automatically creates a symlink from that path to a `microns.h5` (or `microns_functional.h5`) it finds elsewhere in `DATADIR`, so you don't have to move the 19 GB file.
