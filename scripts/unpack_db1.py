"""Unpack NinaPro DB1 subject zips into data/raw/.

Drop s1.zip ... s27.zip into data/raw/ and run:
    uv run python scripts/unpack_db1.py

The zips usually nest the .mat files inside a subfolder named after the subject.
This script extracts each zip, moves the .mat files up to data/raw/, and removes
the now-empty subfolder and the original zip.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def unpack_zip(zip_path: Path) -> list[Path]:
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
        extracted = [RAW / name for name in zf.namelist()]
    return extracted


def flatten_mat_files() -> int:
    moved = 0
    for mat in RAW.rglob("*.mat"):
        if mat.parent == RAW:
            continue
        target = RAW / mat.name
        if target.exists():
            mat.unlink()
        else:
            shutil.move(str(mat), str(target))
            moved += 1
    return moved


def remove_empty_subdirs() -> None:
    for p in sorted(RAW.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()


def main() -> int:
    # Match s1.zip ... s27.zip strictly so unrelated zips in data/raw/ are left alone.
    import re
    zips = sorted(p for p in RAW.glob("s*.zip") if re.fullmatch(r"s\d+\.zip", p.name))
    if not zips:
        print(f"No s*.zip files found in {RAW}.")
        print("Download subject zips from ninapro.hevs.ch/DB1.html and drop them there.")
        return 1

    for z in zips:
        print(f"unpacking {z.name}")
        unpack_zip(z)
        z.unlink()

    moved = flatten_mat_files()
    remove_empty_subdirs()

    mats = sorted(RAW.glob("S*_A1_E*.mat"))
    print(f"{moved} .mat files flattened, {len(mats)} total in {RAW}")
    if len(mats) % 3 != 0:
        print("warning: subject count is not a multiple of 3 — some subjects may be missing an exercise file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
