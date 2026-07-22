#!/usr/bin/env python3
"""
compose_via_ezyseq_sim.py

Calls Carter's ACTUAL ezy_seq balance="sample" selection engine
(_select_by_sample_targets -- the literal function tag_region_abundance_by_FMT
uses internally for balance="sample") directly, invoked as a subprocess from
simulate_ground_truth.R, rather than a hand-translated R reimplementation of
the same algorithm (which turned out to silently diverge -- see CLAUDE.md,
"we really messed up" entry, 2026-07-20).

Bypasses tag_region_abundance_by_FMT's own hi/lo derivation step (which
computes mean +/- dev*SD directly from the INPUT data's own per-sample
region fractions), because this simulation deliberately anchors its
composition-imbalance magnitude to the MANUSCRIPT'S OWN reported real-data
statistic (26.4% +/- 7.4%), not to the raw simulated pool's own composition
(raw_roi_share=0.45 is a subsampling-headroom knob, not a realistic
target). hi_frac/lo_frac are therefore computed in R and passed in here as
pre-computed targets; everything downstream of that (the actual per-sample
feasibility computation and subsampling) is Carter's unmodified code.

Usage:
    python3 compose_via_ezyseq_sim.py <cell_meta_csv> <out_csv> <hi_frac> <lo_frac> <seed>
      cell_meta_csv: columns Cell,sample_ID,Treatment,region
                     (Treatment in {"Treatment","Control"})
      out_csv: one column, 'Cell' -- the selected (kept) cell IDs
"""
import sys
import numpy as np
import pandas as pd

# Requires the repo's ezy_seq package (EzySeq_Library/mypythonlibrary) to be
# installed, e.g. `pip install -e EzySeq_Library/mypythonlibrary` from the
# repo root. Calling Carter's real selection engine directly, rather than a
# hand-translated copy, is the whole point of this script -- see the module
# docstring above.
from ezy_seq.ezyfunctions import _select_by_sample_targets

cell_meta_csv = sys.argv[1]
out_csv       = sys.argv[2]
hi_frac       = float(sys.argv[3])
lo_frac       = float(sys.argv[4])
seed          = int(sys.argv[5])

df = pd.read_csv(cell_meta_csv, dtype={"Cell": str, "sample_ID": str,
                                        "Treatment": str, "region": str})
df = df.set_index("Cell", drop=False)

target_by_fmt = {
    "Treatment": {"Cortex": hi_frac},
    "Control":   {"Cortex": lo_frac},
}

rng = np.random.default_rng(seed)
selected = _select_by_sample_targets(
    df, target_by_fmt, total_per_fmt=None,
    region_col="region", fmt_col="Treatment", sample_col="sample_ID",
    rng=rng, fill="baseline",
)

if not selected:
    raise ValueError("No cells selected -- infeasible target at this dev.")

out = pd.DataFrame({"Cell": sorted(selected)})
out.to_csv(out_csv, index=False)
print(f"Selected {len(selected)} / {len(df)} cells -> {out_csv}")
