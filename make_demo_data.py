"""Create a small demo subset of NASA CMAPSS FD001 (engines 1-10) for deployment.

The dashboard only uses the first 10 engines, so this subset is all it needs.
Run from the project root:  python make_demo_data.py
"""
from pathlib import Path

src = Path("data/raw/train_FD001.txt")
dst = Path("data/demo/train_FD001_demo.txt")
dst.parent.mkdir(parents=True, exist_ok=True)

kept = [line for line in src.read_text().splitlines()
        if line.strip() and int(line.split()[0]) <= 10]
dst.write_text("\n".join(kept) + "\n")
print(f"Wrote {len(kept)} rows (engines 1-10) to {dst}")
