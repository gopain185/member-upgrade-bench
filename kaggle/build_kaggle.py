"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only]

writes kaggle/dataset/ (the grid results and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package
source and reruns the grid).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/ensemble-member-upgrade-benchmark"
NOTEBOOK_ID = "locdunhat/how-much-of-a-member-s-solo-gain-reaches-the-blend"
TITLE = "How Much of a Member's Solo Gain Reaches the Blend?"


def build_dataset(public: bool) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["replace.csv", "add.csv"]:
        (out / name).write_bytes((ROOT / "results" / name).read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Ensemble Member Upgrade Benchmark",
        "id": DATASET_ID,
        "subtitle": "What a soft-voting blend gains when one member improves or a new one joins",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "simulations"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in sorted((ROOT / "upgradebench").glob("*.py"))}


SETUP = """

import os, sys
ROOT = "/kaggle/working/member-upgrade-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("member-upgrade-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
"""

RUN = """import time
import pandas as pd
from upgradebench import run as bench
pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
t = time.time()
rep = bench.summarise(bench.replace_grid(), ["k", "rho", "rho_t", "share", "gain"])
add = bench.summarise(bench.add_grid(), ["k", "rho", "rho_t", "share", "solo"])
print(len(rep), "replace cells,", len(add), "add cells, in", round(time.time() - t, 1), "s")
same = rep[rep.rho_t == "same"]
"""

RETENTION = """same[same.gain == 0.044].pivot_table(index=["k", "rho"], columns="share", values="retention").round(3)
"""

BY_GAIN = """same[(same.k == 4) & (same.rho == 0.7)].pivot_table(index="gain", columns="share", values="retention").round(3)
"""

LOWER = """rep[(rep.rho_t == "lower") & (rep.gain == 0.044)].pivot_table(index=["k", "rho"], columns="share", values="retention").round(3)
"""

DETECT = """same[(same.gain == 0.044) & (same.share == 0.2)][["k", "rho", "rest_acc", "blend_before", "blend_after", "d_blend", "flip",
                                                     "sd_public", "z_public", "rows_needed"]].round(4)
"""

ADD = """a = add[(add.k == 4) & (add.rho_t == "same")]
a.pivot_table(index=["rho", "solo"], columns="share", values="d_blend").round(4)
"""

BEST = """a[a.share == 0.2].pivot_table(index="rho", columns="solo", values="best_share").round(2)
"""

COMPARE = """import glob
hits = glob.glob("/kaggle/input/**/replace.csv", recursive=True)
if hits:
    saved = pd.read_csv(hits[0])
    keys = ["k", "rho", "rho_t", "share", "gain"]
    m = rep.merge(saved, on=keys, suffixes=("", "_saved"))
    print("rows matched:", len(m), "of", len(rep), "| largest difference in d_blend:", float((m.d_blend - m.d_blend_saved).abs().max()))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("markdown", "## 1. The grid\n\nReplace: the target goes from solo 0.52 to 0.52 + gain at fixed share. Add: a new member joins. "
                     "k = 2 / 4 / 8 other members of solo 0.60, rho = 0.3-0.9, 36,000 rows, 4 seeds."),
        ("code", RUN),
        ("markdown", "## 2. Retention: blend gain / solo gain, for a solo gain of 0.044"),
        ("code", RETENTION),
        ("markdown", "## 3. Retention does not depend on the size of the gain (k = 4, rho = 0.7)"),
        ("code", BY_GAIN),
        ("markdown", "## 4. A target with less correlated errors (half the rho)"),
        ("code", LOWER),
        ("markdown", "## 5. Could one submission see it? (solo gain 0.044, share 0.2, 1,200 public rows)"),
        ("code", DETECT),
        ("markdown", "## 6. Adding a new member: change in blend accuracy (k = 4)"),
        ("code", ADD),
        ("code", BEST),
        ("markdown", "## 7. Against the saved runs"),
        ("code", COMPARE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "member-upgrade-bench.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "member-upgrade-bench.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public))
    print("wrote", build_notebook(public))
