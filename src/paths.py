"""Repository paths, resolved from this file's location.

Keeps the scripts runnable from anywhere — `make all` from the repository root,
`python src/run_analysis.py` from the root, or from inside `src/`.
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.path.join(ROOT, "data")
OUTPUT = os.path.join(ROOT, "output")
FIGURES = os.path.join(ROOT, "figures")

for _d in (DATA, OUTPUT, FIGURES):
    os.makedirs(_d, exist_ok=True)
