#!/usr/bin/env python3
"""Convert the three SVG figures used by the (external) report into PDFs.

Run from anywhere: paths resolve relative to the repo root (this file lives in
``scripts/``). Needs ``cairosvg`` (in requirements.txt). Re-run if the source
SVGs change.
"""
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parent.parent
SVGS = [
    "figures/eda/presentation/cohort_z_scores.svg",
    "figures/exp2_stim_trajectories/7_5/presentation/trajectories_2x2_simple.svg",
    "figures/exp3_clip_categories/7_5/presentation/trajectories_2x2_simple.svg",
]

for rel in SVGS:
    src = ROOT / rel
    dst = src.with_suffix(".pdf")
    if not src.exists():
        print(f"  MISSING {src}")
        continue
    cairosvg.svg2pdf(url=str(src), write_to=str(dst))
    print(f"  wrote {dst.relative_to(ROOT)}")
