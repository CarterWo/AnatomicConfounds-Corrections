#!/usr/bin/env python3
"""
region_composition_donuts.py
============================
Regional composition donut charts for the FMT composition-engineering design.

A single AnnData (``Final_Data/fmt/adata_full.h5ad``) now carries every engineered
subset as boolean ``obs`` columns, instead of shipping separate cortex-up /
cortex-down objects. This script rebuilds the 3x2 donut grid from those flags:

    Row G1 (Stroke Cortex Enriched)   -> obs["G1_{seed}_{dev}"]  (cortex-up)
    Row Base (Baseline / unmanipulated) -> the full object, no mask
    Row G2 (Healthy Cortex Enriched)  -> obs["G2_{seed}_{dev}"]  (cortex-down)

    Columns: Stroke_FMT | Healthy_FMT

``dev`` is the shift magnitude in SD units and ``seed`` the iteration; the pair
``(seed=0, dev=1)`` reproduces the original +/-1 SD design. Masks exist for
seeds 0-99 and devs {0.25, 0.5, 1, 1.5}.

Only the five obs columns needed are read from the (multi-GB) h5ad via h5py, so
the expression matrix is never loaded.

Usage
-----
    python region_composition_donuts.py                 # seed 0, dev 1 (default)
    python region_composition_donuts.py --seed 3 --dev 0.5
    python region_composition_donuts.py --framework fmt --out my_figure.png
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
fmt_col = "FMT"
region_col = "quint_region"
napari_col = "napari_region"
coordinate_pattern = r'^\d+,\d+,\d+$'   # rows whose region label is a raw x,y,z pixel coord

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", default="fmt",
                   help="Data framework subfolder under Final_Data (default: fmt).")
    p.add_argument("--seed", type=int, default=0,
                   help="Iteration seed embedded in the mask column name (default: 0).")
    p.add_argument("--dev", default="1",
                   help="Deviation magnitude label: 0.25 | 0.5 | 1 | 1.5 (default: 1).")
    p.add_argument("--data-path", default=None,
                   help="Override path to the h5ad (defaults to "
                        "Final_Data/<framework>/adata_full.h5ad).")
    p.add_argument("--out", default=None,
                   help="Output image path (defaults to "
                        "region_composition_donuts_seed<seed>_dev<dev>.png "
                        "next to this script).")
    return p.parse_args()


# ---------------------------------------------------------
# 2. LIGHTWEIGHT OBS LOADER (reads only the columns we need)
# ---------------------------------------------------------
def _read_obs_column(obs_group, name):
    """Return one obs column as a pandas-friendly array, decoding AnnData's
    categorical (categories/codes) and byte-string encodings."""
    node = obs_group[name]
    if isinstance(node, h5py.Group):                      # categorical
        cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
        codes = node["codes"][:]
        return pd.Categorical.from_codes(codes, categories=cats)
    arr = node[:]                                         # plain dataset
    if arr.dtype.kind in ("S", "O"):
        arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    return arr


def load_obs_subset(h5ad_path, columns):
    """Build a small DataFrame from just `columns` of the h5ad's obs group."""
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        available = set(obs.keys())
        missing = [c for c in columns if c not in available]
        if missing:
            g1_like = sorted(k for k in available if k.startswith("G1_"))
            devs = sorted({k.split("_", 2)[2] for k in g1_like})
            seeds = sorted({int(k.split("_")[1]) for k in g1_like})
            raise KeyError(
                f"Column(s) {missing} not in obs. "
                f"Available mask seeds: {seeds[:5]}...{seeds[-1] if seeds else ''}, "
                f"devs: {devs}."
            )
        idx_key = obs.attrs.get("_index", "_index")
        if isinstance(idx_key, bytes):
            idx_key = idx_key.decode()
        index = _read_obs_column(obs, idx_key)
        data = {c: _read_obs_column(obs, c) for c in columns}
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------
# 3. PREPROCESSING (mirrors the original clean/group logic on obs)
# ---------------------------------------------------------
def clean_and_group(df):
    """Drop rows with raw-coordinate region labels and collapse all cortical
    sub-regions (napari_region == 'Cortex') into a single 'Cortex' bucket."""
    mask_coords = df[region_col].astype(str).str.match(coordinate_pattern)
    out = df.loc[~mask_coords].copy()
    out["grouped_region"] = np.where(
        out[napari_col].astype(str) == "Cortex",
        "Cortex",
        out[region_col].astype(str),
    )
    return out


# ---------------------------------------------------------
# 4. PLOTTING HELPERS
# ---------------------------------------------------------
def get_clean_labels(counts, threshold=2.0):
    total = counts.sum()
    labels_list, autopct_list = [], []
    for name, val in counts.items():
        pct = (val / total) * 100
        if pct >= threshold:
            labels_list.append(name)
            autopct_list.append(f"{pct:.1f}%")
        else:
            labels_list.append("")
            autopct_list.append("")
    return labels_list, autopct_list


def main():
    args = parse_args()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = PROJECT_ROOT / "Final_Data" / args.framework / "adata_full.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(f"h5ad not found: {data_path}")

    g1_col = f"G1_{args.seed}_{args.dev}"   # cortex-up   (Stroke_FMT enriched)
    g2_col = f"G2_{args.seed}_{args.dev}"   # cortex-down (Healthy_FMT enriched)

    print(f"Loading obs columns from {data_path} ...")
    obs = load_obs_subset(
        data_path, [region_col, napari_col, fmt_col, g1_col, g2_col]
    )

    # Derive the three views from the SINGLE object via the boolean flags.
    adata_full_clean = clean_and_group(obs)                              # baseline
    cortex_up_clean = clean_and_group(obs.loc[obs[g1_col].astype(bool)])   # G1
    cortex_down_clean = clean_and_group(obs.loc[obs[g2_col].astype(bool)])  # G2

    # -----------------------------------------------------
    # DYNAMIC COLOR PALETTE
    # -----------------------------------------------------
    all_regions = (
        set(cortex_up_clean["grouped_region"].unique())
        | set(cortex_down_clean["grouped_region"].unique())
        | set(adata_full_clean["grouped_region"].unique())
    )
    unique_grouped_regions = sorted(str(r) for r in all_regions)

    fallback_palette = cm.tab20(np.linspace(0, 1, len(unique_grouped_regions)))
    manual_overrides = {
        "Hippocampus": "lightblue",
        "Cerebellum": "lightgreen",
        "Unassigned": "grey",
    }

    global_color_map = {}
    # Assign colors for everything EXCEPT the Cortex (which is data-driven below).
    for i, region in enumerate(unique_grouped_regions):
        if region != "Cortex":
            base = manual_overrides.get(region, fallback_palette[i])
            global_color_map[region] = mcolors.to_rgba(base, alpha=0.5)

    # -----------------------------------------------------
    # PLOT CONFIGURATION
    # -----------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16, 24))

    rows_config = [
        (0, cortex_up_clean, "G1: Stroke Cortex Enriched"),
        (1, adata_full_clean, "Base: Baseline (Unmanipulated)"),
        (2, cortex_down_clean, "G2: Healthy Cortex Enriched"),
    ]
    cols_config = [
        (0, "Stroke_FMT"),
        (1, "Healthy_FMT"),
    ]

    # Colors for the faded center circles
    center_colors = {
        ("G1: Stroke Cortex Enriched", "Stroke_FMT"): (249 / 255, 100 / 255, 149 / 255),
        ("G1: Stroke Cortex Enriched", "Healthy_FMT"): (85 / 255, 160 / 255, 251 / 255),
        ("Base: Baseline (Unmanipulated)", "Stroke_FMT"): "silver",
        ("Base: Baseline (Unmanipulated)", "Healthy_FMT"): "grey",
        ("G2: Healthy Cortex Enriched", "Stroke_FMT"): "gold",
        ("G2: Healthy Cortex Enriched", "Healthy_FMT"): "magenta",
    }

    # -----------------------------------------------------
    # GENERATE PLOTS
    # -----------------------------------------------------
    for row_idx, df_obj, manip_name in rows_config:
        for col_idx, fmt_group in cols_config:

            ax = axes[row_idx, col_idx]

            subset = df_obj[df_obj[fmt_col] == fmt_group]
            counts = subset["grouped_region"].value_counts().sort_index()

            if counts.empty:
                ax.set_title(f"{manip_name}\n({fmt_group}) - No Data", fontsize=14)
                ax.axis("off")
                continue

            # Data-driven Cortex color: darker when cortex is heavily enriched.
            cortex_count = counts.get("Cortex", 0)
            total_count = counts.sum()
            cortex_pct = (cortex_count / total_count) * 100 if total_count > 0 else 0
            cortex_color = "darkorange" if cortex_pct >= 25.0 else "gold"

            current_colors = []
            for r in counts.index:
                if str(r) == "Cortex":
                    current_colors.append(cortex_color)
                else:
                    current_colors.append(global_color_map.get(str(r), "lightgrey"))

            clean_labels, clean_autopct = get_clean_labels(counts, threshold=2.0)

            # Plot Donut Chart
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=clean_labels,
                autopct="%1.1f%%",
                colors=current_colors,
                startangle=140,
                pctdistance=0.85,
                labeldistance=1.05,
                wedgeprops={"width": 0.3, "edgecolor": "none", "linewidth": 0},
            )

            # Style the Cortex Wedge (thick black outline)
            for i, wedge in enumerate(wedges):
                if clean_labels[i] == "Cortex":
                    wedge.set_edgecolor("black")
                    wedge.set_linewidth(3)
                    wedge.set_zorder(10)

            # Style the region labels
            for text in texts:
                if text.get_text() == "Cortex":
                    text.set_fontsize(18)
                    text.set_fontweight("bold")
                else:
                    text.set_fontsize(10)
                text.set_zorder(15)

            # Style the percentages
            for i, t in enumerate(autotexts):
                t.set_text(clean_autopct[i])
                if clean_autopct[i]:
                    t.set_fontsize(10)
                    t.set_fontweight("bold")
                    t.set_zorder(15)
                    if clean_labels[i] == "Cortex":
                        t.set_color("black")
                        t.set_fontsize(12)
                    else:
                        t.set_color("black")

            # Hollow center (colored and faded)
            bg_color = center_colors.get((manip_name, fmt_group), "white")
            rgba_bg = mcolors.to_rgba(bg_color, alpha=0.25)
            centre_circle = plt.Circle((0, 0), 0.70, fc=rgba_bg, edgecolor="none", zorder=0)
            ax.add_artist(centre_circle)

            ax.set_title(f"{manip_name}\n({fmt_group})", fontsize=14, fontweight="bold")

    fig.suptitle(
        f"Regional composition by FMT  (seed={args.seed}, dev={args.dev})",
        fontsize=18, fontweight="bold", y=1.001,
    )
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = SCRIPT_DIR / f"region_composition_donuts_seed{args.seed}_dev{args.dev}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
