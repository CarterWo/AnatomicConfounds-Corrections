"""
plot_utils.py
=============
Shared plotting helpers for LMM model comparison visualizations, plus the
framework resolver used by every post-DE script so their inputs/outputs are
labelled per framework (genotype / fmt).
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from scipy.optimize import brentq


# =============================================================================
# FRAMEWORK RESOLUTION  (shared by the post-DE scripts)
# =============================================================================

def find_config(name: str = "config.yaml") -> Path:
    """Search cwd, its parents, and the script's parents for config.yaml."""
    here = Path(__file__).resolve()
    for d in [Path.cwd(), *Path.cwd().parents, here.parent, *here.parents]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found near {Path.cwd()} or {here.parent}")


def _framework_from_argv(default=None):
    """Read (and strip) a --framework NAME / --framework=NAME flag from sys.argv.

    These figure scripts don't use argparse, so we pull the flag out by hand and
    remove it, leaving the rest of sys.argv untouched (subprocess relaunches that
    re-exec the file therefore won't choke on it).
    """
    name = default
    argv = sys.argv[1:]
    keep, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--framework" and i + 1 < len(argv):
            name = argv[i + 1]; i += 2; continue
        if a.startswith("--framework="):
            name = a.split("=", 1)[1]; i += 1; continue
        keep.append(a); i += 1
    sys.argv[1:] = keep
    return name


def select_framework(cfg: dict, name=None):
    """Return (framework_block, framework_name).

    Order: explicit arg -> $PIPELINE_FRAMEWORK -> config active_framework. Falls
    back to a flat/legacy config (top-level inputs/outputs/composition).
    """
    if "frameworks" not in cfg:
        return cfg, (cfg.get("active_framework") or "default")
    name = name or os.environ.get("PIPELINE_FRAMEWORK") or cfg.get("active_framework")
    if not name:
        raise SystemExit("No framework selected: pass --framework, set "
                         "PIPELINE_FRAMEWORK, or set active_framework in config.yaml.")
    if name not in cfg["frameworks"]:
        raise SystemExit(f"Framework {name!r} not in config. "
                         f"Available: {list(cfg['frameworks'])}")
    return cfg["frameworks"][name], name


def resolve_framework(name=None):
    """Load config.yaml and pick the framework.

    Returns (framework_block, framework_name, config_dir:Path). Framework comes
    from `name` -> --framework -> $PIPELINE_FRAMEWORK -> active_framework.
    """
    cfg_path = find_config()
    cfg = yaml.safe_load(open(cfg_path))
    if name is None:
        name = _framework_from_argv()
    fw, fw_name = select_framework(cfg, name)
    print(f"[framework] {fw_name}")
    return fw, fw_name, cfg_path.parent


def resolve_results_root(name=None):
    """Return (framework_name, results_root:Path).

    results_root is the framework's ``outputs.lmm_results_dir`` (the labelled DE
    results folder, e.g. LMM_output/genotype), resolved to an absolute path.
    """
    fw, fw_name, cfg_dir = resolve_framework(name)
    root = Path(fw["outputs"]["lmm_results_dir"])
    if not root.is_absolute():
        root = cfg_dir / root
    return fw_name, root


def draw_proportional_venn(ax, n_left, n_right, n_both,
                            col_left, col_right, col_both,
                            left_label=None, right_label=None,
                            title="", logfc_thresh=None,
                            **kwargs):
    """
    Draw a proportional Venn diagram where the intersection area of the two
    equal circles scales linearly with  n_both / (n_left + n_right + n_both).

    The circles are dynamically sized to fill the available axes space
    regardless of overlap degree.

    When n_both == 0  the circles sit flush (just touching, no overlap).
    When n_both == total the circles are fully concentric (d = 0).
    """
    total = n_left + n_right + n_both

    # Solve for normalised half-distance t = d/(2r) in (0,1)
    if total > 0 and 0 < n_both < total:
        f = n_both / total

        def _eq(t):
            return 2.0 * np.arccos(t) - 2.0 * t * np.sqrt(max(1.0 - t * t, 0.0)) - f * np.pi

        t_sol = brentq(_eq, 1e-9, 1.0 - 1e-9)
    elif total > 0 and n_both >= total:
        # Every gene is significant in both: circles fully concentric (d = 0).
        # brentq can't be used here — at f = 1 the bracket collapses onto the root.
        t_sol = 0.0
    else:
        t_sol = 1.0  # circles just touching: d = 2r

    # --- Dynamic sizing to fill the axes ---
    # Get axes size in display (pixel) coords to determine aspect
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_w, ax_h = bbox.width, bbox.height  # inches

    # The footprint of two overlapping circles of radius r with half-distance t*r:
    #   width  = 2r + d = 2r(1 + t)
    #   height = 2r
    # We want to maximise r so the footprint fits in the axes with some padding.
    pad_frac = 0.08  # 8% padding on each side
    usable_w = ax_w * (1.0 - 2 * pad_frac)
    usable_h = ax_h * (1.0 - 2 * pad_frac)

    # r from width constraint:  2r(1+t) = usable_w  =>  r = usable_w / (2(1+t))
    # r from height constraint: 2r      = usable_h  =>  r = usable_h / 2
    r_from_w = usable_w / (2.0 * (1.0 + t_sol))
    r_from_h = usable_h / 2.0
    r = min(r_from_w, r_from_h)

    d = 2.0 * r * t_sol

    # Set up axes in data coordinates: centre everything at (0, 0)
    cx_left  = -d / 2.0
    cx_right =  d / 2.0
    cy = 0.0

    # Tight limits around the footprint with padding
    half_w = r * (1.0 + t_sol)
    x_pad = half_w * pad_frac / (1.0 - 2 * pad_frac)
    y_pad = r * pad_frac / (1.0 - 2 * pad_frac)
    ax.set_xlim(-half_w - x_pad, half_w + x_pad)
    ax.set_ylim(-r - y_pad, r + y_pad)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(plt.Circle((cx_left,  cy), r, color=col_left,  alpha=0.35, zorder=1))
    ax.add_patch(plt.Circle((cx_right, cy), r, color=col_right, alpha=0.35, zorder=1))

    # Place exclusive counts in the outer portions of each circle
    tx_left  = cx_left  - r * 0.42
    tx_right = cx_right + r * 0.42

    ax.text(tx_left, cy, str(n_left), ha="center", va="center",
            fontsize=28, fontweight="bold", color=col_left, zorder=3)
    ax.text(tx_right, cy, str(n_right), ha="center", va="center",
            fontsize=28, fontweight="bold", color=col_right, zorder=3)

    ov_col = col_both if n_both > 0 else "#888"
    mid_x = (cx_left + cx_right) / 2.0
    t_ov = ax.text(mid_x, cy, str(n_both), ha="center", va="center",
                   fontsize=28, fontweight="bold", color=ov_col, zorder=3)
    if n_both > 0:
        t_ov.set_path_effects([withStroke(linewidth=3, foreground="white")])

    # Title
    _title = title
    if logfc_thresh is not None:
        _title += f"\n& |logFC| > {logfc_thresh}"
    ax.set_title(_title, fontsize=13, pad=8)
