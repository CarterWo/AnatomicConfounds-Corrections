#!/usr/bin/env python3
"""
aggregate_seeds.py
==================
Collapse the ``LMM_all.ipynb`` per-seed DE outputs across composition-engineering
iterations into one summary per analysis.

LMM_all writes results to::

    <lmm_results_dir>/dev_<dev>/seed_<seed>/{Pseudobulk_Validation,
                                             Local_Regional_Analysis,
                                             Global_CT_Analysis}/<file>.csv

where, per deviation magnitude, DESeq2 spans all tagged seeds (e.g. 100) and the
Dream LMM spans only the seeds it ran (e.g. 3). This script groups the
identically named files across seeds and writes aggregated results to::

    <lmm_results_dir>/aggregated/dev_<dev>/<subdir>/<file>.csv

Each aggregated file keeps the standard DE columns (logFC = across-seed mean,
P.Value / adj.P.Val = medians) plus cross-seed stability metrics: ``sig_frac``
(selection frequency), ``logFC_sd``, ``logFC_lo``/``logFC_hi`` (2.5-97.5% band),
``dir_consistency``, ``n_seeds`` and ``present_frac``. So it drops straight into
any per-file downstream visualisation, now one file per (dev, analysis).

The seeds are re-subsamples of the SAME cells (not independent replicates), so
p-values are summarised by their distribution rather than pooled -- see
``ezy_seq.lmm.aggregate_seed_files`` for the rationale.

Usage::

    python Post_DE_Processing/aggregate_seeds.py [--framework NAME] [--alpha 0.05] [--results-dir DIR]

Paths come from the selected framework's ``outputs.lmm_results_dir`` in config.yaml
(framework chosen by --framework / $PIPELINE_FRAMEWORK / active_framework) unless
overridden with --results-dir.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

# Use the repo-root compatibility shim, which loads the bundled ezy_seq.lmm
# directly — without the heavy squidpy/dask package import, and independent of
# any older pip-installed ezy_seq copy that might lack aggregate_seed_directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lmm_comparison_utils as lmm


def find_config(name: str = "config.yaml") -> Path:
    """Search the cwd, its parents, and the script's parents for config.yaml."""
    candidates = [Path.cwd(), *Path.cwd().parents,
                  Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for d in candidates:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found near {Path.cwd()} or the script directory")


def select_framework(cfg: dict, name: str | None = None) -> dict:
    """Resolve which frameworks.<name> block to use.

    Order: explicit arg -> $PIPELINE_FRAMEWORK -> config active_framework. Falls
    back to a flat/legacy config (top-level inputs/outputs/composition).
    """
    if "frameworks" not in cfg:
        return cfg
    name = name or os.environ.get("PIPELINE_FRAMEWORK") or cfg.get("active_framework")
    if not name:
        raise SystemExit("No framework selected: pass --framework, set "
                         "PIPELINE_FRAMEWORK, or set active_framework in config.yaml.")
    if name not in cfg["frameworks"]:
        raise SystemExit(f"Framework {name!r} not in config. "
                         f"Available: {list(cfg['frameworks'])}")
    print(f"Framework: {name}")
    return cfg["frameworks"][name]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate LMM_all per-seed DE results across iterations.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for the selection-frequency metric (default 0.05).")
    ap.add_argument("--results-dir", default=None,
                    help="Override the framework's outputs.lmm_results_dir.")
    ap.add_argument("--framework", default=None,
                    help="frameworks.<name> block to use (else PIPELINE_FRAMEWORK / active_framework).")
    args = ap.parse_args()

    cfg_path = find_config()
    cfg = yaml.safe_load(open(cfg_path))
    outputs = select_framework(cfg, args.framework)["outputs"]
    results_root = Path(args.results_dir or outputs["lmm_results_dir"])
    if not results_root.is_absolute():
        results_root = cfg_path.parent / results_root

    print(f"Aggregating per-seed DE results under: {results_root}")
    written = lmm.aggregate_seed_directory(results_root, alpha=args.alpha)
    print(f"\nWrote {len(written)} aggregated files to {results_root / 'aggregated'}")


if __name__ == "__main__":
    main()
