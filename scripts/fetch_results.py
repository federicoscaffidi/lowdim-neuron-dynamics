#!/usr/bin/env python3
"""Fetch the deposited result artifacts and restore them to their repo paths.

The repo tracks source and small CSVs; the large regenerable caches
(``results/**/trajectories.npz``) live in an external deposit. This script
downloads that deposit archive, extracts exactly the files listed in
``data/MANIFEST.tsv`` to the paths given there, and verifies size + sha256.

Stdlib only. Run from anywhere; paths resolve relative to the repo root.

    python scripts/fetch_results.py              # download + extract + verify
    python scripts/fetch_results.py --verify     # only check files already on disk
    python scripts/fetch_results.py --archive X  # use a local .tar.gz instead of downloading

The archive is a gzip tarball whose member names are the manifest paths
(e.g. ``results/exp2_stim_trajectories/7_5/trajectories.npz``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

# <TODO: Federico> fill in once the deposit is published (Zenodo/OSF/...).
# Leave as None until then; the script fails with a readable message.
DEPOSIT_DOI: str | None = None
DEPOSIT_URL: str | None = None  # direct download URL of the .tar.gz

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "MANIFEST.tsv"
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads"
ARCHIVE_NAME = "lowdim-neuron-dynamics-results.tar.gz"


def read_manifest(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["path", "bytes", "sha256"]:
            sys.exit(f"unexpected manifest header in {path}: {header}")
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            rel, nbytes, digest = ln.split("\t")
            rows.append((rel, int(nbytes), digest))
    return rows


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(rows: list[tuple[str, int, str]]) -> int:
    """Check every manifest row; return the number of failures."""
    failures = 0
    for rel, nbytes, digest in rows:
        p = REPO_ROOT / rel
        if not p.exists():
            print(f"  MISSING   {rel}")
            failures += 1
            continue
        size = p.stat().st_size
        if size != nbytes:
            print(f"  SIZE      {rel}: {size} bytes, manifest says {nbytes}")
            failures += 1
            continue
        actual = sha256_of(p)
        if actual != digest:
            print(f"  SHA256    {rel}: mismatch")
            failures += 1
            continue
        print(f"  OK        {rel}")
    print(f"verified {len(rows) - failures}/{len(rows)} files")
    return failures


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n        -> {dest}")
    try:
        with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.URLError as e:
        sys.exit(f"download failed: {e}")


def extract(archive: Path, wanted: set[str]) -> None:
    """Extract only manifest members, refusing paths that escape the repo."""
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        missing = sorted(wanted - members.keys())
        if missing:
            sys.exit(
                "archive does not contain every manifest path; missing:\n  "
                + "\n  ".join(missing)
            )
        for rel in sorted(wanted):
            m = members[rel]
            if m.name.startswith("/") or ".." in Path(m.name).parts or not m.isfile():
                sys.exit(f"refusing unsafe archive member: {m.name!r}")
            target = REPO_ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            with target.open("wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            print(f"  extracted {rel}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="only verify files already on disk; no download")
    ap.add_argument("--archive", type=Path, default=None,
                    help="local .tar.gz to extract instead of downloading")
    ap.add_argument("--url", default=None,
                    help="override DEPOSIT_URL for this run")
    args = ap.parse_args(argv)

    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    rows = read_manifest(MANIFEST)
    wanted = {rel for rel, _, _ in rows}
    print(f"manifest: {MANIFEST.relative_to(REPO_ROOT)} ({len(rows)} files)")

    if args.verify:
        return 1 if verify(rows) else 0

    archive = args.archive
    if archive is None:
        url = args.url or DEPOSIT_URL
        if not url:
            sys.exit(
                "The results deposit is not yet published.\n"
                "  DEPOSIT_URL is unset in scripts/fetch_results.py"
                + (f" (DOI: {DEPOSIT_DOI})" if DEPOSIT_DOI else "")
                + ".\n"
                "  Either pass --url <direct .tar.gz link>, or --archive <local .tar.gz>,\n"
                "  or regenerate the files by running the notebooks (see README)."
            )
        archive = DOWNLOAD_DIR / ARCHIVE_NAME
        download(url, archive)
    elif not archive.exists():
        sys.exit(f"archive not found: {archive}")

    extract(archive, wanted)
    return 1 if verify(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
