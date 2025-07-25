#!/usr/bin/env python3
"""gpu_syncnet_batch.py

Batch‑score lip‑sync for every *.mp4 clip in a directory using SyncNet on a
single GPU **with a clean, global progress‑bar**. All internal tqdm bars from
SyncNet itself are silenced, and FFmpeg runs fully muted.

If an output CSV already exists, any clips whose *stem* is already present are
skipped so that you can resume interrupted runs safely.

Usage
-----
Simply run without arguments if you keep the default paths::

    $ python gpu_syncnet_batch.py

The defaults assume:
* `syncnet_python/` lives alongside this script (or adjust SYNCNET_DIR)
* Your dataset is under `/home/web/data/CelebVHQ/processed/*.mp4`
* Results are written to `/home/web/data/CelebVHQ/sync_scores.csv`

Edit the constants near the top if your layout differs.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration — adjust if needed
# -----------------------------------------------------------------------------
SYNCNET_DIR = Path("./syncnet_python")
CLIPS_DIR = Path("/home/web/data/CelebVHQ/processed")
OUT_CSV = Path("/home/web/data/CelebVHQ/sync_scores.csv")

THRESH = 6.0  # accept if min‑dist ≤ THRESH
BATCH_SIZE = 20
VSHIFT = 15
# -----------------------------------------------------------------------------

# Make SyncNet repo importable
sys.path.insert(0, str(SYNCNET_DIR))
from SyncNetInstance import SyncNetInstance  # type: ignore  # noqa: E402

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@contextmanager
def suppress_inner_tqdm():
    """Silence any tqdm bars created inside imported libraries (SyncNet)."""
    import tqdm as _tqdm

    original = _tqdm.tqdm

    def hidden_tqdm(*args, **kwargs):  # type: ignore[override]
        kwargs["disable"] = True
        return original(*args, **kwargs)

    _tqdm.tqdm = hidden_tqdm  # type: ignore[assignment]
    try:
        yield
    finally:
        _tqdm.tqdm = original  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    # 1) Load SyncNet once on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️  CUDA not available; running on CPU will be very slow.")
    net = SyncNetInstance().to(device)
    net.loadParameters(SYNCNET_DIR / "data" / "syncnet_v2.model")
    net.eval()

    # 2) Prepare / resume CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if OUT_CSV.exists():
        with OUT_CSV.open() as f:
            done = {row["clip"] for row in csv.DictReader(f)}

    csv_file = OUT_CSV.open("a", newline="")
    writer = csv.writer(csv_file)
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        writer.writerow(["clip", "median_dist", "is_synced", "offset"])

    # 3) Gather clips and iterate with global tqdm
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    progress = tqdm(clips, desc="Sync", unit="vid", ascii=True)

    for mp4 in progress:
        stem = mp4.stem
        if stem in done:
            continue  # already processed

        # # a) Convert to temporary AVI
        tmp_dir = Path(tempfile.mkdtemp(prefix="sync_"))

        # b) Run SyncNet (silencing its own bars)
        opt = SimpleNamespace(
            tmp_dir=str(tmp_dir),
            reference=stem,
            batch_size=BATCH_SIZE,
            vshift=VSHIFT,
        )

        with torch.no_grad(), suppress_inner_tqdm():
            offset, confidence, frame_dists = net.evaluate(opt, str(mp4), verbose=False)  # type: ignore[arg-type]

        median_dist = float(np.median(frame_dists))
        writer.writerow([stem, f"{median_dist:.3f}", int(median_dist <= THRESH), offset])
        csv_file.flush()  # ensure recoverability

        # c) Clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)

    csv_file.close()
    print("✅  Finished – results in", OUT_CSV)


if __name__ == "__main__":
    main()
