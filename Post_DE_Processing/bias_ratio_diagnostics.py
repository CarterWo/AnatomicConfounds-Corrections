#!/usr/bin/env python3
"""
bias_ratio_diagnostics.py
=========================
Technical diagnostics for the normalized effect-size disagreement metric that
the figures report as the "Bias Ratio" / "% of logFC".

For two per-gene effect estimates a (scenario/method A) and b (B), the metric is

    bias_ratio = mean_g |a_g - b_g|  /  mean_g ( (|a_g| + |b_g|) / 2 )

Because |a-b| <= |a|+|b|, this is bounded on [0, 2] (0-200%): 0 when the two
estimates agree, up to 200% when they are systematically opposite in sign. It is
a *disagreement* measure, not a causal decomposition of signal.

This script does not defend that interpretation; it characterizes the metric's
behavior, per the reviewer's request:

  * distribution of the per-gene differences (a_g - b_g), with median and IQR;
  * median absolute difference (a robust alternative to the mean numerator);
  * rank (Spearman) and linear (Pearson) correlation of a vs b, with bootstrap CIs;
  * stability when the denominator (average |effect|) is near zero:
      - the distribution of the per-gene denominator and the fraction near zero;
      - the aggregate metric recomputed while flooring the denominator;
      - the reported ratio-of-means vs the mean-of-per-gene-ratios, which diverge
        precisely when many genes have near-zero effects.

Calibration against known/simulated effects is intentionally out of scope here.

Inputs are the same DE result CSVs the figure scripts consume:
  * G1/G2 (composition):  <lmm_results_dir>/<subpath>/{Global_CT_Analysis,
      Local_Regional_Analysis,Pseudobulk_Validation}/{UP,DOWN}_<...>.csv
      (default subpath aggregated/dev_1). delta = logFC_down - logFC_up.
  * Anatomic (baseline):  <lmm_results_dir>/base/... BASE_<stem>_dream_<ann>.csv.
      delta = logFC_aware - logFC_blind  (blind_vs_napari, blind_vs_quint).

Usage:
  python Post_DE_Processing/bias_ratio_diagnostics.py [--framework NAME]
      [--results-dir DIR] [--subpath aggregated/dev_1]
      [--mode {g1g2,anatomic,both}] [--out DIR] [--n-boot 1000]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))          # plot_utils
sys.path.insert(0, str(SCRIPT_DIR.parent))   # lmm_comparison_utils shim (repo root)
from plot_utils import resolve_results_root
import lmm_comparison_utils as lmm

warnings.filterwarnings("ignore")


# ───────────────────────────── core diagnostics ─────────────────────────────
def _ratio_of_means(abs_delta, avg_mag):
    den = avg_mag.mean()
    return float(abs_delta.mean() / den) if den > 0 else np.nan


def diagnose_pair(a, b, name, n_boot=1000, seed=0):
    """Compute the full diagnostic suite for one effect-estimate pair (a vs b).

    delta is a - b (caller orders a/b to match the figure convention).
    Returns a flat dict of statistics plus a private '_arrays' entry for plotting.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    delta = a - b
    abs_delta = np.abs(delta)
    avg_mag = (np.abs(a) + np.abs(b)) / 2.0

    pos = avg_mag > 0
    pergene_ratio = np.full(n, np.nan)
    pergene_ratio[pos] = abs_delta[pos] / avg_mag[pos]      # in [0, 2] where defined

    bias_ratio     = _ratio_of_means(abs_delta, avg_mag)   # the reported metric
    mean_of_ratios = float(np.nanmean(pergene_ratio))      # denominator-sensitive form
    median_ratio   = float(np.nanmedian(pergene_ratio))
    med_mag        = float(np.median(avg_mag))
    robust_bias    = float(np.median(abs_delta) / med_mag) if med_mag > 0 else np.nan

    # correlations
    rho  = float(stats.spearmanr(a, b)[0]) if n >= 3 else np.nan
    r_pe = float(stats.pearsonr(a, b)[0])  if n >= 3 else np.nan

    # bootstrap CIs for the rank correlation and the reported metric
    rng = np.random.default_rng(seed)
    boot_rho, boot_br = [], []
    if n >= 5 and n_boot:
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            boot_rho.append(float(stats.spearmanr(a[idx], b[idx])[0]))
            boot_br.append(_ratio_of_means(abs_delta[idx], avg_mag[idx]))

    def _ci(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if v.size == 0:
            return np.nan, np.nan
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    rho_lo, rho_hi = _ci(boot_rho)
    br_lo, br_hi   = _ci(boot_br)

    # sign flips and how much of the numerator they drive
    nz = (a != 0) & (b != 0)
    flip = np.zeros(n, bool)
    flip[nz] = np.sign(a[nz]) != np.sign(b[nz])
    sign_flip_frac = float(flip[nz].mean()) if nz.any() else np.nan
    tot_num = abs_delta.sum()
    num_from_flip = float(abs_delta[flip].sum() / tot_num) if tot_num > 0 else np.nan

    # denominator near-zero fractions
    eps_levels = [0.01, 0.05, 0.1, 0.25]
    frac_below = {e: float((avg_mag < e).mean()) for e in eps_levels}

    # stability sweep: floor the denominator, recompute both forms of the metric
    qs = np.quantile(avg_mag, np.linspace(0.0, 0.9, 19)) if n else np.array([0.0])
    grid = np.unique(np.concatenate([[0.0], qs]))
    sweep_rows = []
    for e in grid:
        keep = avg_mag >= e
        if keep.sum() >= 10:
            sweep_rows.append((
                float(e),
                _ratio_of_means(abs_delta[keep], avg_mag[keep]),
                float(np.nanmean(pergene_ratio[keep])),
                int(keep.sum()),
            ))
    sweep = pd.DataFrame(sweep_rows, columns=["floor", "ratio_of_means", "mean_of_ratios", "n"])

    return {
        "name": name, "n_genes": n,
        "bias_ratio_pct": 100 * bias_ratio,
        "bias_ratio_ci_lo_pct": 100 * br_lo, "bias_ratio_ci_hi_pct": 100 * br_hi,
        "mean_of_ratios_pct": 100 * mean_of_ratios,
        "median_pergene_ratio_pct": 100 * median_ratio,
        "robust_bias_pct": 100 * robust_bias,
        "mean_abs_delta": float(abs_delta.mean()) if n else np.nan,
        "median_abs_delta": float(np.median(abs_delta)) if n else np.nan,
        "delta_mean": float(delta.mean()) if n else np.nan,
        "delta_median": float(np.median(delta)) if n else np.nan,
        "delta_q25": float(np.percentile(delta, 25)) if n else np.nan,
        "delta_q75": float(np.percentile(delta, 75)) if n else np.nan,
        "delta_sd": float(delta.std(ddof=1)) if n > 1 else np.nan,
        "spearman_r": rho, "spearman_ci_lo": rho_lo, "spearman_ci_hi": rho_hi,
        "pearson_r": r_pe,
        "sign_flip_frac": sign_flip_frac,
        "num_share_from_signflip": num_from_flip,
        "frac_denom_lt_0.01": frac_below[0.01], "frac_denom_lt_0.05": frac_below[0.05],
        "frac_denom_lt_0.1": frac_below[0.1], "frac_denom_lt_0.25": frac_below[0.25],
        "_arrays": (a, b, delta, abs_delta, avg_mag, pergene_ratio, sweep),
    }


# ───────────────────────────── plotting ─────────────────────────────
def plot_pair(res, outpath):
    a, b, delta, abs_delta, avg_mag, pergene_ratio, sweep = res["_arrays"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Bias-ratio diagnostics — {res['name']}   (n={res['n_genes']} genes)",
                 fontsize=14, fontweight="bold")

    # (1) per-gene difference distribution
    a0 = ax[0, 0]
    a0.hist(delta, bins=60, color="#4C78A8", alpha=0.85)
    a0.axvline(0, color="grey", lw=0.8, ls="--")
    a0.axvline(res["delta_median"], color="k", lw=1.5,
               label=f"median {res['delta_median']:+.3f}\nIQR [{res['delta_q25']:+.3f}, {res['delta_q75']:+.3f}]")
    a0.set_title("Per-gene difference (a − b)")
    a0.set_xlabel("Δ effect (logFC)"); a0.set_ylabel("genes"); a0.legend(fontsize=8)

    # (2) A vs B scatter with correlations
    a1 = ax[0, 1]
    a1.scatter(b, a, s=6, alpha=0.35, color="#555555")
    if a.size:
        lim = float(np.nanpercentile(np.abs(np.concatenate([a, b])), 99)) or 1.0
        a1.plot([-lim, lim], [-lim, lim], color="red", lw=1, ls="--")
        a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_title(f"A vs B   Spearman ρ={res['spearman_r']:.3f} "
                 f"[{res['spearman_ci_lo']:.3f}, {res['spearman_ci_hi']:.3f}]\n"
                 f"Pearson r={res['pearson_r']:.3f}")
    a1.set_xlabel("effect B"); a1.set_ylabel("effect A")

    # (3) denominator distribution
    a2 = ax[0, 2]
    p = avg_mag > 0
    if p.any():
        a2.hist(np.log10(avg_mag[p]), bins=60, color="#F58518", alpha=0.85)
    a2.set_title(f"Denominator = avg |effect|\n< 0.05 for {100*res['frac_denom_lt_0.05']:.1f}% of genes")
    a2.set_xlabel("log10 avg |effect|"); a2.set_ylabel("genes")

    # (4) stability vs denominator floor
    a3 = ax[1, 0]
    if len(sweep):
        a3.plot(sweep["floor"], 100 * sweep["ratio_of_means"], "-o", ms=3,
                color="#4C78A8", label="ratio of means (reported)")
        a3.plot(sweep["floor"], 100 * sweep["mean_of_ratios"], "-s", ms=3,
                color="#E45756", label="mean of per-gene ratios")
        a3b = a3.twinx()
        a3b.plot(sweep["floor"], sweep["n"], color="grey", lw=0.8, ls=":")
        a3b.set_ylabel("n genes retained", color="grey")
    a3.set_title("Stability vs near-zero denominator")
    a3.set_xlabel("denominator floor (drop avg |effect| < floor)")
    a3.set_ylabel("metric (%)"); a3.legend(fontsize=8, loc="upper left")

    # (5) per-gene ratio distribution (bounded [0, 2])
    a4 = ax[1, 1]
    pr = pergene_ratio[np.isfinite(pergene_ratio)]
    if pr.size:
        a4.hist(np.clip(pr, 0, 2), bins=60, color="#54A24B", alpha=0.85)
        a4.axvline(res["median_pergene_ratio_pct"] / 100, color="k", lw=1.5,
                   label=f"median {res['median_pergene_ratio_pct']:.0f}%")
        a4.legend(fontsize=8)
    a4.set_xlim(0, 2)
    a4.set_title("Per-gene ratio |a−b| / avg|effect|  (bounded [0, 2])")
    a4.set_xlabel("per-gene ratio"); a4.set_ylabel("genes")

    # (6) text summary
    a5 = ax[1, 2]; a5.axis("off")
    lines = [
        f"n genes ................ {res['n_genes']}",
        "",
        "REPORTED METRIC",
        f"  bias ratio ........... {res['bias_ratio_pct']:.1f}%",
        f"  95% CI (bootstrap) ... [{res['bias_ratio_ci_lo_pct']:.1f}, {res['bias_ratio_ci_hi_pct']:.1f}]%",
        "",
        "DENOMINATOR-SENSITIVE FORMS",
        f"  mean of ratios ....... {res['mean_of_ratios_pct']:.1f}%",
        f"  median per-gene ratio  {res['median_pergene_ratio_pct']:.1f}%",
        f"  robust (med/med) ..... {res['robust_bias_pct']:.1f}%",
        "",
        "PER-GENE DIFFERENCE",
        f"  mean |Δ| ............. {res['mean_abs_delta']:.4f}",
        f"  median |Δ| .......... {res['median_abs_delta']:.4f}",
        f"  Δ median [IQR] ...... {res['delta_median']:+.4f} [{res['delta_q25']:+.3f}, {res['delta_q75']:+.3f}]",
        "",
        "AGREEMENT",
        f"  Spearman ρ .......... {res['spearman_r']:.3f}",
        f"  Pearson r ........... {res['pearson_r']:.3f}",
        f"  sign-flip genes ..... {100*res['sign_flip_frac']:.1f}%  "
        f"(drive {100*res['num_share_from_signflip']:.1f}% of Σ|Δ|)",
    ]
    a5.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=10.5, transform=a5.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ───────────────────────────── loaders ─────────────────────────────
def _std(fp, model):
    return lmm.standardize_columns(lmm.load_de_file(fp), model)


def load_g1g2_pairs(base):
    """Discover UP/DOWN pairs under an aggregated dev folder. delta = down - up."""
    data, local, sc, pb = (base / "Global_CT_Analysis",
                           base / "Local_Regional_Analysis",
                           base / "SingleCell_Tests",
                           base / "Pseudobulk_Validation")
    up, down = {}, {}
    for d in (data, local, sc):
        if not d.exists():
            continue
        for fp in d.glob("*.csv"):
            p = lmm.parse_seurat_dream_filename(fp.name)
            if not p:
                continue
            # Wilcoxon is region-blind and annotation-less; label it "blind".
            ann = p["annotation"] or ("blind" if p["model"] == "wilcoxon" else None)
            if ann is None:
                continue
            key = (p["model"], lmm.simplify_celltype_name(p["cell_type"]), ann)
            (up if p["direction"] == "UP" else down)[key] = fp
    if pb.exists():
        for fp in pb.glob("*.csv"):
            p = lmm.parse_seurat_dream_filename(fp.name)
            if not p or p["model"] != "pseudobulk":
                continue
            key = ("deseq2", lmm.simplify_celltype_name(p["cell_type"]), "pb")
            (up if p["direction"] == "UP" else down)[key] = fp

    pairs = []
    for key in sorted(set(up) & set(down)):
        model, ct, ann = key
        std = "pseudobulk" if model in ("deseq2", "pseudobulk") else model
        m = pd.merge(_std(up[key], std), _std(down[key], std),
                     on="Gene", suffixes=("_up", "_down"), how="inner")
        if len(m) < 3:
            continue
        pairs.append((f"g1g2/{ct}/{model}/{ann}",
                      m["logFC_down"].to_numpy(), m["logFC_up"].to_numpy()))
    return pairs


def _parse_base_name(fname):
    """Parse BASE_<stem>_dream_<annotation>.csv -> (stem, annotation) or None."""
    stem = Path(fname).stem
    if not stem.startswith("BASE_"):
        return None
    rest = stem[len("BASE_"):]
    for ann in ("napari", "quint", "blind"):
        if rest.endswith(f"_dream_{ann}"):
            return rest[: -len(f"_dream_{ann}")], ann
    return None


def load_anatomic_pairs(base_root):
    """Discover blind-vs-aware baseline pairs. delta = aware - blind."""
    files = {}
    for d in (base_root / "Global_CT_Analysis", base_root / "Local_Regional_Analysis"):
        if not d.exists():
            continue
        for fp in d.glob("*.csv"):
            parsed = _parse_base_name(fp.name)
            if parsed:
                files[parsed] = fp

    pairs = []
    for stem in sorted({s for (s, _) in files}):
        blind = files.get((stem, "blind"))
        if blind is None:
            continue
        for aware_ann in ("napari", "quint"):
            aware = files.get((stem, aware_ann))
            if aware is None:
                continue
            m = pd.merge(_std(blind, "dream"), _std(aware, "dream"),
                         on="Gene", suffixes=("_blind", "_aware"), how="inner")
            if len(m) < 3:
                continue
            pairs.append((f"anatomic/{stem}/blind_vs_{aware_ann}",
                          m["logFC_aware"].to_numpy(), m["logFC_blind"].to_numpy()))
    return pairs


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--framework", default=None,
                    help="frameworks.<name> block (else PIPELINE_FRAMEWORK / active_framework).")
    ap.add_argument("--results-dir", default=None,
                    help="Override the framework's outputs.lmm_results_dir.")
    ap.add_argument("--subpath", default="aggregated/dev_1",
                    help="G1/G2 results subfolder under the results dir (default aggregated/dev_1).")
    ap.add_argument("--mode", choices=["g1g2", "anatomic", "both"], default="both")
    ap.add_argument("--out", default=None, help="Output dir (default figures/<framework>/bias_diagnostics).")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs (default 1000).")
    args = ap.parse_args()

    fw, root = resolve_results_root(args.framework)
    if args.results_dir:
        root = Path(args.results_dir)
    out = Path(args.out) if args.out else (SCRIPT_DIR / "figures" / fw / "bias_diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    if args.mode in ("g1g2", "both"):
        g = root / args.subpath
        if g.exists():
            pairs += load_g1g2_pairs(g)
        else:
            print(f"  [g1g2] not found (run aggregate_seeds first): {g}")
    if args.mode in ("anatomic", "both"):
        base = root / "base"
        if base.exists():
            pairs += load_anatomic_pairs(base)
        else:
            print(f"  [anatomic] not found (run DE_Base_Analysis first): {base}")

    if not pairs:
        print("No effect-estimate pairs found. Run the DE pipeline (+ aggregate_seeds) first.")
        return

    rows = []
    for name, a, b in pairs:
        res = diagnose_pair(a, b, name, n_boot=args.n_boot)
        plot_pair(res, out / f"{name.replace('/', '__')}.png")
        res.pop("_arrays")
        rows.append(res)
        print(f"  {name:48s} bias={res['bias_ratio_pct']:6.1f}%  "
              f"median|d|={res['median_abs_delta']:.3f}  rho={res['spearman_r']:.3f}  "
              f"denom<0.05={100*res['frac_denom_lt_0.05']:.0f}%")

    csv = out / "bias_ratio_diagnostics.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nWrote {len(rows)} diagnostics rows -> {csv}")
    print(f"Figures -> {out}")


if __name__ == "__main__":
    main()
