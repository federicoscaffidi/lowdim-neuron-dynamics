#!/usr/bin/env /opt/homebrew/bin/python3.11
"""Convert the SVGs referenced by report.tex into PDFs (vector-preserving).

Run once, then `pdflatex` can include the resulting PDFs directly. Re-run if
the source SVGs change.
"""
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parent.parent
SVGS = [
    "figures/presentation/cohort_z_scores.svg",
    "figures/option3/7_5/presentation/trajectories_2x2_simple.svg",
    "figures/option4/7_5/presentation/trajectories_2x2_simple.svg",
]

for rel in SVGS:
    src = ROOT / rel
    dst = src.with_suffix(".pdf")
    if not src.exists():
        print(f"  MISSING {src}")
        continue
    cairosvg.svg2pdf(url=str(src), write_to=str(dst))
    print(f"  wrote {dst.relative_to(ROOT)}")
