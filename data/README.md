# Results deposit

Large, regenerable result caches are kept out of the git history and in an
external deposit. Everything else (CSVs, figures, notebooks) is tracked in
the repo.

**Contents.** The two `trajectories.npz` caches written by
`notebooks/experiment2.ipynb` and `notebooks/experiment3.ipynb`
(session `7_5`; 13 arrays each, one per area × stimulus class, `np.savez`,
uncompressed). Re-running those notebooks regenerates them.

**DOI.** `<TODO: Federico>` — not yet published. Until then
`scripts/fetch_results.py` exits with a message saying so.

**Fetch.**

```bash
python scripts/fetch_results.py            # download, extract to repo paths, verify
python scripts/fetch_results.py --verify   # check files already on disk
```

The per-file list with byte counts and SHA-256 is in `MANIFEST.tsv`
(`MANIFEST.sha256` is the same data in `shasum -c` format).

**Size by directory.**

| directory | files | MB |
|---|---:|---:|
| `results/exp2_stim_trajectories/7_5/` | 1 | 14.1 |
| `results/exp3_clip_categories/7_5/` | 1 | 14.1 |
| total | 2 | 28.1 |

**Not in the deposit.** The raw MICrONS file
(`functional/microns_functional.h5`, ~20.6 GB) is *not* distributed here.
It is expected under `$MICRONS_DATADIR` (see the top-level README); this
`data/` directory is ignored by git apart from these three files.
