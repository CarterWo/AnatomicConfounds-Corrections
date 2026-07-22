from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np
import numpy.random as npr
import squidpy as sq
import scanpy as sc
import re
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.stats import norm
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests


def rank_DE(
    adata,
    groupby: str,
    comparisons: str | list[tuple[str, str]] = "all_vs_ref",
    reference: str | None = None,
    groups_keep: list[str] | None = None,
    cell_key: str | None = None,
    cells_keep: list[str] | None = None,
    extra_filter: dict | None = None,
    method: str = "wilcoxon",
):
    """
    Run differential expression using scanpy.tl.rank_genes_groups.

    Parameters
    ----------
    groupby : str
        Column in adata.obs to compare (e.g., "condition", "tissue").
    comparisons : "all_vs_ref" or list of (target, reference) tuples
        Which pairwise comparisons to run.
    reference : str, optional
        Required when comparisons == "all_vs_ref".
    groups_keep : list[str], optional
        Restrict to a subset of categories within `groupby`.
    cell_key : str, optional
        Column in adata.obs defining a subgroup to filter by.
    cells_keep : list[str], optional
        Values in `cell_key` to keep.
    extra_filter : dict, optional
        Additional {obs_column: allowed_values} filters applied before DE.
    method : str
        scanpy.tl.rank_genes_groups method (default: "wilcoxon").

    Returns
    -------
    pd.DataFrame
        Columns: gene, log2fc, score, p, padj, neglog10p, target, ref.
    """
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} not found in adata.obs")
    if comparisons == "all_vs_ref" and not reference:
        raise ValueError("comparisons='all_vs_ref' requires `reference` to be provided.")

    obs = adata.obs.copy()
    if not pd.api.types.is_categorical_dtype(obs[groupby]):
        obs[groupby] = pd.Categorical(obs[groupby])

    if groups_keep is not None:
        obs = obs[obs[groupby].isin(set(groups_keep))]

    if cell_key and cells_keep:
        if cell_key not in obs:
            raise KeyError(f"{cell_key!r} not found in adata.obs")
        obs = obs[obs[cell_key].isin(cells_keep)]

    if extra_filter:
        for key, allowed in extra_filter.items():
            if key not in obs:
                raise KeyError(f"{key!r} not found in adata.obs")
            obs = obs[obs[key].isin(allowed)]

    adata_filt = adata[obs.index].copy()
    cats = list(pd.Categorical(adata_filt.obs[groupby]).categories)

    if comparisons == "all_vs_ref":
        if reference not in cats:
            raise ValueError(f"reference {reference!r} not found in {groupby} categories: {cats}")
        pairs = [(c, reference) for c in cats if c != reference]
    else:
        pairs = comparisons
        for t, r in pairs:
            if t not in cats or r not in cats:
                raise ValueError(f"Pair ({t},{r}) uses unknown category; available: {cats}")

    for target, ref in pairs:
        sub = adata_filt[adata_filt.obs[groupby].isin([target, ref])].copy()
        if pd.api.types.is_categorical_dtype(sub.obs[groupby]):
            sub.obs[groupby] = sub.obs[groupby].cat.remove_unused_categories()

        sc.tl.rank_genes_groups(sub, groupby=groupby, method=method, reference=ref, pts=True)

        res = sub.uns["rank_genes_groups"]
        names    = list(res["names"][target])
        lfcs     = np.asarray(res["logfoldchanges"][target], dtype=float)
        scores   = np.asarray(res["scores"][target], dtype=float)
        p_raw    = np.asarray(res["pvals"][target], dtype=float)
        padj     = np.asarray(res["pvals_adj"][target], dtype=float)
        neglog10 = -np.log10(np.clip(res["pvals_adj"][target], 1e-300, None))

        df = pd.DataFrame({
            "gene":      names,
            "log2fc":    lfcs,
            "score":     scores,
            "p":         p_raw,
            "padj":      padj,
            "target":    target,
            "neglog10p": neglog10,
            "ref":       ref,
        })
        return df


def ensure_is_list_of_adatas(
    adatas,
    on_single_message: str = "You passed a single AnnData. Please pass a list of AnnData objects.",
) -> list[AnnData]:
    """Validate that input is a non-empty list of AnnData objects."""
    if isinstance(adatas, AnnData):
        raise TypeError(on_single_message)
    if adatas is None or not hasattr(adatas, "__iter__"):
        raise TypeError("Expected an iterable (list/tuple) of AnnData objects.")
    adatas = list(adatas)
    if not all(isinstance(a, AnnData) for a in adatas):
        raise TypeError("All items must be AnnData objects.")
    if len(adatas) == 0:
        raise ValueError("Empty list provided.")
    return adatas


def apply_annotation(
    adatas: list[AnnData],
    meta_dict: dict,
    cell_type_probablility_key=None,
    cell_type_key=None,
) -> list[AnnData]:
    """
    Apply sample/region metadata to a list of AnnData objects.

    Parameters
    ----------
    adatas : list[AnnData]
        Each object must have 'slide_ID' and 'sample_ID' columns in .obs.
    meta_dict : dict
        Nested dictionary produced by read_dictionary(). Structure:
        {slide_id: {sample_id: {'extra_meta_data': {...}, ...}}}
    cell_type_probablility_key : str, optional
        Regex pattern matching a column to rename as 'cell_type_probablility'.
    cell_type_key : str, optional
        Regex pattern matching a column to rename as 'cell_type'.

    Returns
    -------
    list[AnnData]
        Per-sample AnnData objects with metadata columns added.
    """
    split_adatas = []

    for adata in adatas:
        for slide_name in meta_dict:
            if slide_name not in adata.obs['slide_ID'].values:
                continue

            print('***********************')
            print("Processing:", slide_name)
            print('***********************')

            for sample in meta_dict[slide_name]:
                print(sample)
                if sample not in adata.obs['sample_ID'].values:
                    print(f"  Skipping {sample}: not found in current adata.")
                    continue

                sub_adata = adata[adata.obs['sample_ID'] == sample].copy()
                sample_data = meta_dict[slide_name][sample]

                if 'extra_meta_data' in sample_data:
                    for meta_key, meta_val in sample_data['extra_meta_data'].items():
                        sub_adata.obs[meta_key] = meta_val
                        print(f"    Assigned {meta_key}: {meta_val}")

                if cell_type_probablility_key is not None:
                    prob_col = [col for col in sub_adata.obs.columns
                                if re.match(cell_type_probablility_key, col)]
                    sub_adata.obs.rename(columns={prob_col[0]: 'cell_type_probablility'}, inplace=True)

                if cell_type_key is not None:
                    cluster_col = [col for col in sub_adata.obs.columns
                                   if re.match(cell_type_key, col)]
                    sub_adata.obs.rename(columns={cluster_col[0]: 'cell_type'}, inplace=True)

                split_adatas.append(sub_adata)

    return split_adatas


def find_elbow_n_genes(adatas: list) -> None:
    """
    Plot a histogram of genes detected per cell for each AnnData.

    Useful for visually choosing a minimum-gene filter threshold before
    calling filter_and_normalize().
    """
    adatas = ensure_is_list_of_adatas(adatas)
    for adata in adatas:
        if 'nFeature_RNA' not in adata.obs:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
        vals = adata.obs['nFeature_RNA'].to_numpy()

        plt.figure(figsize=(6, 4))
        plt.hist(vals[~np.isnan(vals)], bins=100,
                 label=f"AnnData Object {adata.obs['sample_ID'][0]}")
        plt.xlabel('Number of genes detected')
        plt.ylabel('Number of cells')
        plt.legend()
        plt.tight_layout()
        plt.show()


def pseudobulk_export(
    adatas,
    sample_key: str = 'sample_ID',
    group_key: str = 'FMT',
    tissue_key: str = 'tissue',
    celltype_key: str = 'cell_type',
    layer: str = 'counts',
    by_celltype: bool = False,
    out_base: str = "./pseudobulk_output",
):
    """
    Aggregate single-cell counts into pseudobulk matrices and write to CSV.

    Parameters
    ----------
    adatas : list[AnnData] or AnnData
        Input data. All objects must share the same gene set (or a union is used).
    sample_key : str
        .obs column identifying biological samples.
    group_key : str
        .obs column identifying treatment/condition groups.
    celltype_key : str
        .obs column identifying cell types (used when by_celltype=True).
    layer : str
        Layer to use for raw counts; falls back to .X if not found.
    by_celltype : bool
        If True, aggregate per sample × cell type rather than per sample.
    out_base : str or Path
        Output directory. Writes counts.csv (genes × samples) and meta.csv.

    Returns
    -------
    counts_df : pd.DataFrame
    meta_df : pd.DataFrame
    """
    print("Outputs sent to:", out_base)
    if not isinstance(adatas, (list, tuple)):
        adatas = [adatas]

    genes = None
    for ad in adatas:
        if genes is None:
            genes = list(ad.var_names)
        else:
            if not np.array_equal(np.array(genes), np.array(ad.var_names)):
                genes = sorted(set(genes).union(set(ad.var_names)))
    genes = list(genes)

    def get_counts_matrix(ad):
        if layer is not None and layer in ad.layers:
            X = ad.layers[layer]
        else:
            X = ad.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.array(X)

    pb_dict = {}
    meta_rows = []
    for ad in adatas:
        X = get_counts_matrix(ad)
        obs = ad.obs
        if list(ad.var_names) != genes:
            gene_index = {g: i for i, g in enumerate(ad.var_names)}
            big_X = np.zeros((X.shape[0], len(genes)), dtype=X.dtype)
            for j, g in enumerate(genes):
                if g in gene_index:
                    big_X[:, j] = X[:, gene_index[g]]
            X = big_X

        samples   = obs[sample_key].astype(str).values
        groups    = obs[group_key].astype(str).values if group_key in obs else np.array(['NA'] * len(samples))
        celltypes = obs[celltype_key].astype(str).values if celltype_key in obs else np.array(['NA'] * len(samples))

        for i, s in enumerate(samples):
            key = f"{s}__{celltypes[i]}" if by_celltype else s
            if key not in pb_dict:
                pb_dict[key] = np.zeros(len(genes), dtype=X.dtype)
            pb_dict[key] += X[i, :]

        unique_keys = set(f"{samples[i]}__{celltypes[i]}" if by_celltype else samples)
        for key in unique_keys:
            if by_celltype:
                sample, ct = key.split("__", 1)
                sel = (samples == sample) & (celltypes == ct)
                meta_rows.append({'sample': key, 'group': groups[sel][0], 'n_cells': int(sel.sum())})
            else:
                sel = (samples == key)
                meta_rows.append({'sample': key, 'group': groups[sel][0], 'n_cells': int(sel.sum())})

    meta_df = pd.DataFrame(meta_rows).drop_duplicates(subset='sample').set_index('sample')
    samples_order = sorted(pb_dict.keys())
    counts_mat = np.vstack([pb_dict[s] for s in samples_order]).T
    counts_df = pd.DataFrame(counts_mat, index=genes, columns=samples_order)
    counts_df = counts_df.round().astype(int)

    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    counts_fp = out_base / "counts.csv"
    meta_fp   = out_base / "meta.csv"
    counts_df.to_csv(counts_fp)
    meta_df.to_csv(meta_fp)

    print(f"Wrote counts -> {counts_fp} (rows=genes, cols=samples)")
    print(f"Wrote meta   -> {meta_fp} (rows=samples)")

    return counts_df, meta_df


def filter_and_normalize(
    adatas: list,
    min_gene_cnt: int = 15,
    min_t_cnt: int = 20,
    count_normalization_target: float = 1e4,
    min_cell_cnt: int = 100,
) -> list[AnnData]:
    """
    Filter and normalize a list of CosMx AnnData objects.

    Filtering steps applied per object:
      1. Remove cells with fewer than `min_gene_cnt` detected genes.
      2. Remove cells with fewer than `min_t_cnt` total transcript counts.
      3. Remove genes detected in fewer than `min_cell_cnt` cells.
      4. Remove negative-probe ('NegPrb') and false-code ('FalseCode') control features.
      5. Remove cells where >10% of counts originate from negative probes.

    Normalization: log1p transform followed by normalize_total to
    `count_normalization_target` counts.

    Parameters
    ----------
    adatas : list[AnnData]
    min_gene_cnt : int
        Minimum genes detected per cell.
    min_t_cnt : int
        Minimum total counts per cell.
    count_normalization_target : float
        Target sum for normalize_total.
    min_cell_cnt : int
        Minimum number of cells a gene must appear in to be retained.

    Returns
    -------
    list[AnnData]
        Raw counts stored in .layers["counts"]; .X holds log-normalized values.
    """
    adatas = ensure_is_list_of_adatas(adatas)
    new_adatas = []
    for adata in adatas:
        adata_manip = adata.copy()
        adata_manip.layers["counts"] = adata_manip.X.copy()
        # ── gene-wise first ──
        sc.pp.filter_genes(adata_manip, min_cells=min_cell_cnt)
        adata_manip = adata_manip[:, adata_manip.var["NegPrb"].fillna(False) == False]
        adata_manip = adata_manip[:, adata_manip.var["FalseCode"].fillna(False) == False].copy()
        # ── cell-wise next ──
        sc.pp.filter_cells(adata_manip, min_genes=min_gene_cnt)
        sc.pp.filter_cells(adata_manip, min_counts=min_t_cnt)
        adata_manip = adata_manip[adata_manip.obs.pct_counts_NegPrb <= 10, :]
        sc.pp.normalize_total(adata_manip, target_sum=count_normalization_target)
        sc.pp.log1p(adata_manip)
        new_adatas.append(adata_manip)
    return new_adatas

def combine_adatas(path_, adatas):
    """
    Concatenate a list of AnnData objects and write the result to an h5ad file.

    Parameters
    ----------
    path_ : str or Path
        Output file path for the combined h5ad.
    adatas : list[AnnData]

    Returns
    -------
    AnnData
    """
    adata_full = sc.concat(adatas, join='outer', axis=0, merge="first", index_unique='-')
    if 'NegPrb' in adata_full.var.columns:
        adata_full.var['NegPrb'] = (
            adata_full.var['NegPrb'].astype('boolean').fillna(False).astype(bool)
        )
    if 'FalseCode' in adata_full.var.columns:
        adata_full.var['FalseCode'] = (
            adata_full.var['FalseCode'].astype('boolean').fillna(False).astype(bool)
        )
    adata_full.write_h5ad(path_)
    return adata_full


def read_dictionary(dictionary_dir: str | os.PathLike) -> dict:
    """
    Build a nested metadata dictionary from an Excel file.

    The file must contain a 'sample_metadata' sheet with at minimum 'slide_ID'
    and 'sample_ID' columns. An optional 'sub_regions' sheet maps samples to
    anatomical sub-regions via 'sample_id', 'region', and 'coords' columns.

    All columns in 'sample_metadata' beyond the reserved set (slide_ID, sample_ID,
    coords) are captured as 'extra_meta_data' and attached to each AnnData by
    apply_annotation().

    Parameters
    ----------
    dictionary_dir : str or Path
        Path to the Excel metadata file.

    Returns
    -------
    dict
        {slide_id: {sample_id: {
            'coords': range or None,
            'sub_regions': {region_name: [fov_list]},
            'extra_meta_data': {col: value}
        }}}
    """
    master_dictionary = {}

    with pd.ExcelFile(dictionary_dir) as xls:
        meta_df = pd.read_excel(xls, sheet_name="sample_metadata")
        if "sub_regions" in xls.sheet_names:
            sub_df = pd.read_excel(xls, sheet_name="sub_regions", dtype={"sample_id": str})
        else:
            sub_df = pd.DataFrame(columns=["sample_id", "region", "coords"])

    if "slide_ID" in meta_df.columns:
        meta_df["slide_ID"] = meta_df["slide_ID"].astype(str)
    if "sample_ID" in meta_df.columns:
        meta_df["sample_ID"] = meta_df["sample_ID"].astype(str)

    sub_lookup = {}
    if not sub_df.empty and {"sample_id", "region"}.issubset(sub_df.columns):
        for sample_id, group in sub_df.groupby("sample_id"):
            regions = {}
            for _, row in group.iterrows():
                c_str = row.get("coords")
                if isinstance(c_str, str):
                    regions[row["region"]] = [int(x) for x in c_str.split(",") if x.strip().isdigit()]
                else:
                    regions[row["region"]] = []
            sub_lookup[sample_id] = regions

    def parse_range_str(s):
        if isinstance(s, str) and "," in s:
            try:
                start, end = map(int, s.split(",", 1))
                return range(start, end)
            except ValueError:
                pass
        return None

    if "slide_ID" not in meta_df.columns:
        return {}

    for slide_id, slide_group in meta_df.groupby("slide_ID", dropna=False):
        slide_dict = {}
        for _, row in slide_group.iterrows():
            sample_id = row.get("sample_ID")
            if pd.isna(sample_id):
                continue

            coords_val = parse_range_str(row["coords"]) if "coords" in row else None

            reserved_cols = ["slide_ID", "sample_ID", "coords"]
            extra_meta_series = row.drop(labels=reserved_cols, errors='ignore')
            extra_meta = {k: v for k, v in extra_meta_series.items() if not pd.isna(v)}

            slide_dict[sample_id] = {
                "coords":         coords_val,
                "sub_regions":    sub_lookup.get(sample_id, {}),
                "extra_meta_data": extra_meta,
            }
        master_dictionary[slide_id] = slide_dict

    return master_dictionary


def _alloc_by_baseline_with_caps(baseline_counts: pd.Series, caps: pd.Series, total: int) -> pd.Series:
    """
    Allocate `total` across regions in proportion to `baseline_counts`, each region
    capped at `caps`.

    Proportional water-filling: the share is split by baseline weight, and whenever a
    region's share exceeds its available cells the surplus is re-spread over the
    remaining regions in proportion to *their* baseline (again capped), repeating
    until the whole `total` is placed or every region sits at its cap. This preserves
    the relative baseline composition among regions that still have room, and — unlike
    a single remainder pass — always places the full budget whenever the regions
    jointly have the cells, for any set/number of regions. `total` is clamped to the
    total available so it never over-asks.
    """
    out = pd.Series(0, index=baseline_counts.index, dtype=int)
    caps = caps.reindex(out.index).fillna(0).clip(lower=0).astype(int)
    weights = baseline_counts.reindex(out.index).clip(lower=0).astype(float)
    remaining = int(min(int(total), int(caps.sum())))
    if remaining <= 0 or float(weights.sum()) <= 0:
        return out

    # Regions eligible to receive cells: positive baseline weight AND some capacity.
    active = [r for r in out.index if caps[r] > 0 and weights[r] > 0]
    while remaining > 0 and active:
        w = weights[active]
        wsum = float(w.sum())
        if wsum <= 0:
            break
        raw = w / wsum * remaining
        base = np.floor(raw).astype(int)
        # 1) place the proportional floor for each region, bounded by its room
        for r in active:
            give = min(int(base[r]), int(caps[r] - out[r]), remaining)
            if give > 0:
                out[r] += give
                remaining -= give
        # 2) hand any residual to the largest fractional shares that still have room
        for r in (raw - base).sort_values(ascending=False).index:
            if remaining <= 0:
                break
            if out[r] < caps[r]:
                out[r] += 1
                remaining -= 1
        # regions now at capacity drop out; the loop re-spreads what's left over the rest
        active = [r for r in active if out[r] < caps[r]]
    return out


def _alloc_equal_with_caps(caps: pd.Series, total: int) -> pd.Series:
    """
    Split `total` as evenly as possible across the regions in `caps.index`, giving
    each region at most `caps[region]` cells.

    Water-filling: every region with room gets an equal share; regions that hit their
    cap drop out and the remainder is re-spread equally over those that still have
    room. Unlike baseline-proportional allocation, small categories are represented
    up to their availability instead of being rounded away — so no region is dropped
    just because it is rare, as long as `total` >= the number of regions.
    """
    caps = caps.clip(lower=0).astype(int)
    out = pd.Series(0, index=caps.index, dtype=int)
    total = int(min(int(total), int(caps.sum())))
    room = [r for r in caps.index if caps[r] > 0]
    while total > 0 and room:
        share = max(1, total // len(room))
        for r in list(room):
            if total <= 0:
                break
            give = min(share, int(caps[r]) - int(out[r]), total)
            out[r] += give
            total -= give
            if out[r] >= caps[r]:
                room.remove(r)
    return out


def set_region_abundance_by_FMT(
    adata,
    target_by_fmt,
    total_per_fmt=None,
    region_col: str = "napari_region",
    fmt_col: str = "FMT",
    random_state=0,
):
    """
    Composition-engineering experiment: subsample cells per treatment group to hit
    target regional proportions (counts or fractions) while keeping total cell count
    fixed. Unspecified regions retain their baseline relative abundances.

    Parameters
    ----------
    adata : AnnData
    target_by_fmt : dict[str, dict[str, int | float]]
        Maps each FMT group to a {region: target} dict. Targets may be absolute
        counts (int) or fractions (float).
    total_per_fmt : int | dict[str, int] | None
        Per-group cell budget. Defaults to all available cells.
    region_col : str
        obs column with region labels (default: 'napari_region').
    fmt_col : str
        obs column with treatment group labels (default: 'FMT').
    random_state : int | None
        RNG seed for reproducibility.

    Returns
    -------
    AnnData
        Subsampled copy containing only the selected cells.
    """
    rng = np.random.default_rng(random_state)
    obs_rf = adata.obs[[region_col, fmt_col]].dropna().copy()
    avail = obs_rf.groupby([fmt_col, region_col], observed=True).size().unstack(fill_value=0)
    selected = []

    for fmt in avail.index:
        row = avail.loc[fmt]
        total_avail = int(row.sum())

        if isinstance(total_per_fmt, dict):
            t = int(min(total_per_fmt.get(fmt, total_avail), total_avail))
        elif isinstance(total_per_fmt, int):
            t = int(min(total_per_fmt, total_avail))
        else:
            t = total_avail

        baseline = (row / row.sum()).fillna(0)
        desired = pd.Series(0, index=row.index, dtype=int)

        if fmt in target_by_fmt:
            tgt = target_by_fmt[fmt]
            is_count = all(isinstance(v, (int, np.integer)) for v in tgt.values())
            if is_count:
                for r, v in tgt.items():
                    if r in desired.index:
                        desired[r] = int(v)
            else:
                frac_map = {r: float(v) for r, v in tgt.items() if r in desired.index}
                sfrac = sum(frac_map.values())
                if sfrac > 1.0:
                    frac_map = {r: v / sfrac for r, v in frac_map.items()}
                for r, fr in frac_map.items():
                    desired[r] = int(round(fr * t))

            desired = desired.clip(upper=row)
            unspecified = [r for r in desired.index if r not in tgt]
            remain = max(0, t - int(desired.sum()))
            if unspecified and remain > 0:
                caps_unspec = (row - desired).loc[unspecified].clip(lower=0)
                desired.loc[unspecified] += _alloc_by_baseline_with_caps(
                    baseline.loc[unspecified], caps_unspec, remain
                )
        else:
            desired = _alloc_by_baseline_with_caps(row, row.clip(lower=0), t)

        while int(desired.sum()) > t:
            biggest = desired.idxmax()
            if desired[biggest] == 0:
                break
            desired[biggest] -= 1
        desired = desired.clip(upper=row)

        sub = obs_rf[obs_rf[fmt_col] == fmt]
        for region, need in desired.items():
            n = int(need)
            if n <= 0:
                continue
            pool = sub.index[sub[region_col] == region].to_numpy()
            selected.extend(pool.tolist() if n >= len(pool) else
                            rng.choice(pool, size=n, replace=False).tolist())

    sel_index = [idx for idx in obs_rf.index if idx in set(selected)]
    if not sel_index:
        raise ValueError("No cells selected.")
    return adata[sel_index].copy()


def _budget_for(total_per_fmt, key, total_avail: int) -> int:
    """Resolve a group's cell budget from `total_per_fmt` (int, dict, or None)."""
    if isinstance(total_per_fmt, dict):
        return int(min(total_per_fmt.get(key, total_avail), total_avail))
    if isinstance(total_per_fmt, (int, np.integer)):
        return int(min(total_per_fmt, total_avail))
    return int(total_avail)


def _allocate_group(region_avail: pd.Series, target, budget: int, fill: str = "baseline") -> pd.Series:
    """
    Turn availability + a target into per-region desired counts for a `budget`.

    `region_avail` is region -> available count. `target` is a {region: count|fraction}
    dict (or None). Specified regions hit their target; the leftover budget fills the
    unspecified regions, everything capped at availability and trimmed so the total
    never exceeds `budget`.

    `fill` controls how the leftover is spread over the unspecified regions:
      - "baseline": proportional to each region's baseline abundance (rare regions
        can round away to zero).
      - "equal": split as evenly as possible across every unspecified region, capped
        by availability, so each is represented before any is over-filled.
    The two give the same *total* leftover; only its distribution differs.
    """
    baseline = (region_avail / region_avail.sum()).fillna(0)
    desired = pd.Series(0, index=region_avail.index, dtype=int)

    def _fill(caps_unspec, baseline_unspec, remain):
        if fill == "equal":
            return _alloc_equal_with_caps(caps_unspec, remain)
        return _alloc_by_baseline_with_caps(baseline_unspec, caps_unspec, remain)

    if target:
        # Targets are either absolute counts (int) or fractions of the budget (float).
        is_count = all(isinstance(value, (int, np.integer)) for value in target.values())
        if is_count:
            for region, value in target.items():
                if region in desired.index:
                    desired[region] = int(value)
        else:
            # Keep only fractions for regions this group actually has; renormalise if
            # they sum above 1, then convert each fraction to a cell count of the budget.
            region_fracs = {region: float(value) for region, value in target.items()
                            if region in desired.index}
            frac_sum = sum(region_fracs.values())
            if frac_sum > 1.0:
                region_fracs = {region: frac / frac_sum for region, frac in region_fracs.items()}
            for region, frac in region_fracs.items():
                desired[region] = int(round(frac * budget))

        desired = desired.clip(upper=region_avail)
        # Whatever budget is left after the targeted regions goes to the untargeted ones.
        unspecified = [region for region in desired.index if region not in target]
        leftover_budget = max(0, budget - int(desired.sum()))
        if unspecified and leftover_budget > 0:
            unspecified_caps = (region_avail - desired).loc[unspecified].clip(lower=0)
            desired.loc[unspecified] += _fill(unspecified_caps, baseline.loc[unspecified], leftover_budget)
    else:
        # No target at all: fill the whole budget at baseline proportions.
        desired = _fill(region_avail.clip(lower=0), region_avail, budget)

    # Final trim: shave from the largest region until the total fits the budget.
    while int(desired.sum()) > budget:
        biggest_region = desired.idxmax()
        if desired[biggest_region] == 0:
            break
        desired[biggest_region] -= 1
    return desired.clip(upper=region_avail)


def _sample_from_pool(cells, desired: pd.Series, region_col, rng) -> list:
    """Randomly draw `desired[region]` obs-index labels from `cells` per region."""
    chosen = []
    for region, count in desired.items():
        n_needed = int(count)
        if n_needed <= 0:
            continue
        pool = cells.index[cells[region_col] == region].to_numpy()
        # Take the whole pool when we need at least all of it; otherwise a random subset.
        chosen.extend(pool.tolist() if n_needed >= len(pool) else
                      rng.choice(pool, size=n_needed, replace=False).tolist())
    return chosen


def _select_by_fmt_targets(avail, obs_rf, target_by_fmt, total_per_fmt,
                           region_col, fmt_col, rng, fill: str = "baseline") -> set:
    """
    Pooled (FMT-wise) selection: all cells within each FMT group are pooled across
    samples, then subsampled to hit that group's per-region targets on a single
    per-FMT budget. Returns the selected obs-index labels as a set.
    """
    selected = []
    for fmt in avail.index:
        region_avail = avail.loc[fmt]                       # this group's per-region counts
        budget = _budget_for(total_per_fmt, fmt, int(region_avail.sum()))
        desired = _allocate_group(region_avail, target_by_fmt.get(fmt), budget, fill=fill)
        group_cells = obs_rf[obs_rf[fmt_col] == fmt]
        selected += _sample_from_pool(group_cells, desired, region_col, rng)
    return set(selected)


def _max_feasible_n(region_avail: pd.Series, target) -> int:
    """
    Largest total cell count N a single sample can supply while hitting `target`
    exactly, with unspecified regions filled at baseline proportions.

    For fraction targets, each specified region constrains N <= avail_region /
    frac_region (you need N * frac_region cells of it but only avail_region exist),
    and the unspecified regions jointly constrain N <= (sum unspecified avail) /
    (1 - sum specified fracs). Returns the sample's total for a None or count target.
    """
    total = int(region_avail.sum())
    if not target:
        return total
    if all(isinstance(value, (int, np.integer)) for value in target.values()):
        return total  # explicit counts carry no fraction-based ceiling

    # Fraction targets for regions this sample has; renormalise if they sum above 1.
    target_fracs = {region: float(value) for region, value in target.items()
                    if region in region_avail.index}
    frac_sum = sum(target_fracs.values())
    if frac_sum > 1.0:
        target_fracs = {region: frac / frac_sum for region, frac in target_fracs.items()}

    # Each targeted region caps N at avail / fraction; the tightest cap wins.
    n_ceilings = [region_avail[region] / frac for region, frac in target_fracs.items() if frac > 0]
    specified_regions = set(target_fracs)
    leftover_frac = 1.0 - sum(target_fracs.values())
    if leftover_frac > 1e-9:
        # All untargeted regions together must fill the remaining fraction of N.
        unspecified_avail = int(region_avail[[region for region in region_avail.index
                                               if region not in specified_regions]].sum())
        n_ceilings.append(unspecified_avail / leftover_frac)
    return int(np.floor(min(n_ceilings))) if n_ceilings else total


def _feasible_n_by_sample_targets(obs_rsf, target_by_fmt, total_per_fmt,
                                  region_col, fmt_col, sample_col):
    """Largest equal per-sample cell count feasible for `target_by_fmt`.

    Returns ``(N, avail, fmt_groups, binding)`` where ``N`` is the biggest count every
    sample can contribute at its FMT target (min over samples of
    ``_max_feasible_n``, capped by any ``total_per_fmt`` ceiling) and ``binding`` is
    the ``(N_i, fmt, sample)`` tuple that sets it. Factored out so a caller can read
    the feasible N for two scenarios and draw both at a common (matched) size.
    """
    avail = obs_rsf.groupby([fmt_col, sample_col, region_col], observed=True).size().unstack(fill_value=0)
    fmt_groups = list(avail.index.get_level_values(fmt_col).unique())

    feasible_by_sample = []           # list of (N_i, fmt, sample)
    budget_ceilings = []
    for fmt in fmt_groups:
        fmt_block = avail.xs(fmt, level=fmt_col)          # rows = sample, cols = region
        n_samples = max(1, len(fmt_block.index))
        target = target_by_fmt.get(fmt)
        for sample_id in fmt_block.index:
            feasible_by_sample.append((_max_feasible_n(fmt_block.loc[sample_id], target), fmt, sample_id))
        if total_per_fmt is not None:
            budget = _budget_for(total_per_fmt, fmt, int(fmt_block.values.sum()))
            budget_ceilings.append(budget // n_samples)

    # The cell-poorest sample sets the shared per-sample count for everyone.
    binding = min(feasible_by_sample, key=lambda item: item[0])
    N = binding[0]
    if budget_ceilings:
        N = min(N, min(budget_ceilings))
    return N, avail, fmt_groups, binding


def _per_sample_feasible_n(obs_rsf, target_by_fmt, region_col, fmt_col, sample_col) -> dict:
    """Per-sample feasible maxima ``{sample_id: max N}`` for a target config.

    Unlike `_feasible_n_by_sample_targets` (which returns the min across samples),
    this keeps each sample's own `_max_feasible_n`, so two scenarios can be matched
    sample-by-sample rather than collapsed to a single global count.
    """
    avail = obs_rsf.groupby([fmt_col, sample_col, region_col], observed=True).size().unstack(fill_value=0)
    feasible_by_sample = {}
    for fmt in avail.index.get_level_values(fmt_col).unique():
        fmt_block = avail.xs(fmt, level=fmt_col)
        target = target_by_fmt.get(fmt)
        for sample_id in fmt_block.index:
            feasible_by_sample[sample_id] = _max_feasible_n(fmt_block.loc[sample_id], target)
    return feasible_by_sample


def _select_by_sample_targets(obs_rsf, target_by_fmt, total_per_fmt,
                              region_col, fmt_col, sample_col, rng,
                              fill: str = "baseline", force_n=None, per_sample_n=None) -> set:
    """
    Per-sample selection at each sample's FMT target fraction.

    Each sample is subsampled to hit its FMT's target cortex fraction. The per-sample
    cell count is chosen by, in priority order:

    - ``per_sample_n`` : an explicit ``{sample_id: N_i}`` map (used to size-match two
      scenarios sample-by-sample; each sample's N_i must be <= its own feasibility).
      Per-sample counts may differ, so the FMT arm totals need not be equal.
    - ``force_n``      : a single count applied to every sample (a common matched
      size); must not exceed the feasible maximum.
    - otherwise        : the maximisation ``N = min over samples of
      _max_feasible_n(sample, target)`` (all samples get the same N; one cell-poor
      sample caps everyone). ``total_per_fmt`` is an optional ceiling here.

    Returns the selected labels as a set.
    """
    N, avail, fmt_groups, binding = _feasible_n_by_sample_targets(
        obs_rsf, target_by_fmt, total_per_fmt, region_col, fmt_col, sample_col)

    if per_sample_n is None:
        if force_n is not None:
            if force_n > N:
                raise ValueError(
                    f"force_n={force_n} exceeds the feasible per-sample max N={N} "
                    f"(binding sample {binding[2]!r} in FMT {binding[1]!r}).")
            N = int(force_n)
        if N <= 0:
            raise ValueError(
                f"Max equal per-sample contribution is 0: sample {binding[2]!r} "
                f"(FMT {binding[1]!r}) cannot reach its target fraction. "
                f"Lower `dev` or use balance='fmt'."
            )
        print(f"[sample-balanced] N={N} cells/sample x {len(avail.index)} samples "
              f"(feasible max {binding[0]}, limited by {binding[2]!r} in FMT {binding[1]!r})")
    else:
        counts = list(per_sample_n.values())
        print(f"[sample-matched] per-sample N over {len(avail.index)} samples "
              f"(min {min(counts)}, max {max(counts)})")

    selected = []
    for fmt in fmt_groups:
        fmt_block = avail.xs(fmt, level=fmt_col)
        target = target_by_fmt.get(fmt)
        for sample_id in fmt_block.index:
            # Each sample's count: its matched N_i if provided, else the shared feasible N.
            n_for_sample = int(per_sample_n[sample_id]) if per_sample_n is not None else N
            desired = _allocate_group(fmt_block.loc[sample_id], target, n_for_sample, fill=fill)
            sample_cells = obs_rsf[(obs_rsf[fmt_col] == fmt) & (obs_rsf[sample_col] == sample_id)]
            selected += _sample_from_pool(sample_cells, desired, region_col, rng)
    return set(selected)


def tag_region_abundance_by_FMT(
    adata,
    dev,
    random_state=0,
    total_per_fmt=None,
    region_of_interest: str = "Cortex",
    up_fmt: str = "Stroke_FMT",
    down_fmt: str = "Healthy_FMT",
    region_col: str = "napari_region",
    fmt_col: str = "FMT",
    sample_col: str = "sample_ID",
    balance: str = "fmt",
    fill_other: str = "baseline",
    match_scenarios: bool = False,
    match_reference_dev: float | None = None,
    col_prefixes: tuple[str, str] = ("G1", "G2"),
    copy: bool = False,
):
    """
    Composition-engineering (in-place tagging variant of `set_region_abundance_by_FMT`).

    Instead of subsampling into two separate AnnData objects, this builds a matched
    pair of opposite composition scenarios and records each one as a boolean column
    on `adata.obs`, so a single object carries every engineered subset.

    The `region_of_interest` (default 'Cortex') target proportions are derived from
    its cross-sample distribution:

        hi = mean + dev * SD
        lo = max(0, mean - dev * SD)

    where mean/SD are taken over the per-sample region fractions. `dev` replaces the
    previously hard-coded 1-SD shift, so `dev=1` reproduces the original ±1 SD design.
    The two scenarios are:

        G1 : up_fmt   -> hi,   down_fmt -> lo
        G2 : up_fmt   -> lo,   down_fmt -> hi

    `balance` controls how cells are drawn to hit those targets (the targets
    themselves are always the FMT-level fractions above):

    - ``"fmt"`` (default): pool all of a group's cells across samples and subsample
      once to the group target, on a per-FMT budget. Sample identity is ignored, so
      a scenario can draw disproportionately from one sample.
    - ``"sample"``: subsample each sample independently to the same target, and give
      every sample the *same* cell count N. N is maximised subject to feasibility --
      the largest count every sample can hit at its target (one cell-poor sample sets
      it for all). `total_per_fmt` is then an optional ceiling, not required.

    In both modes, non-target regions are filled at baseline proportions and any FMT
    group not named by `up_fmt`/`down_fmt` simply keeps its baseline mix.

    Parameters
    ----------
    adata : AnnData
    dev : float
        Magnitude of the shift in standard deviations of the region-of-interest
        fraction. `dev=1` matches the original ±1 SD experiment.
    random_state : int | None
        RNG seed. Each scenario is drawn from its own `default_rng(random_state)`,
        so results are reproducible and independent of evaluation order (matching the
        prior pattern of calling the subsampling function twice with the same seed).
        Also embedded in the output column names.
    total_per_fmt : int | dict[str, int] | None
        Per-FMT cell budget when balance='fmt' (defaults to all available cells).
        When balance='sample' it is only an optional ceiling on the maximised
        per-sample count (N <= budget // n_samples per FMT); None means no ceiling.
    region_of_interest : str
        Region whose abundance is shifted (default 'Cortex').
    up_fmt, down_fmt : str
        FMT groups enriched/depleted in G1 (reversed in G2). Defaults
        'Stroke_FMT'/'Healthy_FMT' match the manuscript.
    region_col, fmt_col, sample_col : str
        obs columns for region, treatment group, and biological sample.
    balance : {"fmt", "sample"}
        Draw cells pooled per FMT (default) or independently per sample. See above.
    fill_other : {"baseline", "equal"}
        How the non-manipulated regions share the leftover budget after the
        region_of_interest is set. "baseline" (default) keeps each region in
        proportion to its original abundance — deducting/adding relative to that
        baseline; if a region's proportional share exceeds its available cells the
        surplus is redistributed proportionally over the others, so the full leftover
        is always placed and the region_of_interest still hits its target for any
        set/number of non-cortex regions. "equal" instead splits the leftover evenly
        across every other region (capped by availability). Only the
        region_of_interest is ever manipulated; this controls the background
        composition of everything else.
    match_scenarios : bool
        If True (balance='sample' only), match each sample to its own counterpart
        across the mirror pair: sample i contributes N_i = min(feasible_G1(i),
        feasible_G2(i)) to BOTH G1 and G2, at each scenario's target cortex fraction.
        So a given sample (e.g. 120L) has the SAME cell count in G1 and G2, while
        counts still vary across samples (no collapse to the poorest sample). Each
        sample still hits its arm's exact target fraction; because per-sample counts
        differ, the Stroke/Healthy arm totals within a scenario need not be equal.
        Default False keeps the legacy behaviour where each scenario independently
        maximises a single shared per-sample N (so G1 > G2).
    match_reference_dev : float | None
        Only used with match_scenarios. If set (e.g. the largest deviation, 1.5),
        take each sample's matched count from that REFERENCE deviation's feasibility
        instead of the current `dev`, so every deviation ends up with the same
        per-sample cell counts. Cells are still drawn at the CURRENT dev's target
        cortex fractions -- only the counts come from the reference. The larger
        (more extreme) deviation is the more cell-limited, so its counts are always
        feasible at milder deviations. None matches each deviation to itself.
    col_prefixes : (str, str)
        Prefixes for the two output columns (default ('G1', 'G2')).
    copy : bool
        If True, operate on and return a copy; otherwise tag `adata` in place.

    Returns
    -------
    AnnData
        The tagged object with two new boolean obs columns:
        ``{col_prefixes[0]}_{seed}_{dev}`` and ``{col_prefixes[1]}_{seed}_{dev}``
        (e.g. ``G1_0_1`` / ``G2_0_1``). `dev` is formatted compactly, so a
        fractional shift like 0.5 yields ``G1_0_0.5`` (bracket-access the column,
        since the dot blocks attribute access).
    """
    if balance not in ("fmt", "sample"):
        raise ValueError("balance must be 'fmt' or 'sample'.")
    if fill_other not in ("equal", "baseline"):
        raise ValueError("fill_other must be 'equal' or 'baseline'.")

    adata = adata.copy() if copy else adata

    for col in (region_col, fmt_col, sample_col):
        if col not in adata.obs:
            raise KeyError(f"{col!r} not found in adata.obs")

    # Cross-sample mean/SD of the region-of-interest fraction.
    counts_ps = adata.obs.groupby([sample_col, region_col], observed=True).size().unstack(fill_value=0)
    if region_of_interest not in counts_ps.columns:
        raise KeyError(
            f"region_of_interest {region_of_interest!r} not found in {region_col!r} "
            f"categories: {list(counts_ps.columns)}"
        )
    fracs_ps = counts_ps.div(counts_ps.sum(axis=1), axis=0)
    roi_mean = float(fracs_ps[region_of_interest].mean())
    roi_std  = float(fracs_ps[region_of_interest].std())
    hi = roi_mean + dev * roi_std
    lo = max(0.0, roi_mean - dev * roi_std)

    target_g1 = {up_fmt: {region_of_interest: hi}, down_fmt: {region_of_interest: lo}}
    target_g2 = {up_fmt: {region_of_interest: lo}, down_fmt: {region_of_interest: hi}}

    # Optional reference-deviation targets: with match_scenarios, per-sample counts
    # can be taken from a REFERENCE deviation (e.g. the largest) so every deviation
    # shares the same per-sample counts. Cells are still drawn at the CURRENT dev's
    # fractions above; only the counts are read from these reference targets.
    ref_targets = None
    if match_scenarios and match_reference_dev is not None:
        hi_r = roi_mean + match_reference_dev * roi_std
        lo_r = max(0.0, roi_mean - match_reference_dev * roi_std)
        ref_targets = (
            {up_fmt: {region_of_interest: hi_r}, down_fmt: {region_of_interest: lo_r}},
            {up_fmt: {region_of_interest: lo_r}, down_fmt: {region_of_interest: hi_r}},
        )

    if balance == "fmt":
        if match_scenarios:
            raise NotImplementedError(
                "match_scenarios is currently implemented for balance='sample' only.")
        obs_rf = adata.obs[[region_col, fmt_col]].dropna().copy()
        avail = obs_rf.groupby([fmt_col, region_col], observed=True).size().unstack(fill_value=0)
        select = lambda tgt: _select_by_fmt_targets(
            avail, obs_rf, tgt, total_per_fmt, region_col, fmt_col,
            np.random.default_rng(random_state), fill=fill_other)
    else:  # "sample"
        obs_rsf = adata.obs[[region_col, fmt_col, sample_col]].dropna().copy()
        per_sample_n = None
        if match_scenarios:
            # Match each sample to ITS OWN counterpart across the mirror pair: sample i
            # contributes N_i = min(feasible_G1(i), feasible_G2(i)) to BOTH scenarios,
            # at each scenario's target cortex fraction. So a given sample (e.g. 120L)
            # ends with the same cell count in G1 and G2, while counts still vary across
            # samples (no collapse to the poorest sample). The low-cortex target is the
            # more cell-limited half, so this trims each sample's cortex-up count down to
            # its own cortex-down count. Because per-sample counts differ, the
            # Stroke/Healthy arm totals within a scenario need not be equal.
            # Per-sample matched count at the CURRENT dev (always feasible for the draw).
            f1 = _per_sample_feasible_n(obs_rsf, target_g1, region_col, fmt_col, sample_col)
            f2 = _per_sample_feasible_n(obs_rsf, target_g2, region_col, fmt_col, sample_col)
            per_sample_n = {s: int(min(f1[s], f2[s])) for s in f1}
            if ref_targets is not None:
                # Cap each sample to the reference deviation's matched count, so all
                # deviations share the same per-sample counts. min() with the current
                # feasibility above keeps the draw feasible even if the reference were
                # somehow larger for a sample.
                r1 = _per_sample_feasible_n(obs_rsf, ref_targets[0], region_col, fmt_col, sample_col)
                r2 = _per_sample_feasible_n(obs_rsf, ref_targets[1], region_col, fmt_col, sample_col)
                per_sample_n = {s: int(min(per_sample_n[s], r1[s], r2[s])) for s in per_sample_n}
            bad = sorted(s for s, n in per_sample_n.items() if n <= 0)
            if bad:
                raise ValueError(
                    f"Samples cannot reach the target at dev={dev}: {bad}. Lower `dev`.")
            print(f"[match_scenarios] per-sample matched N (min over "
                  f"{col_prefixes[0]}/{col_prefixes[1]}): "
                  + ", ".join(f"{s}={n}" for s, n in per_sample_n.items()))
        select = lambda tgt: _select_by_sample_targets(
            obs_rsf, tgt, total_per_fmt, region_col, fmt_col, sample_col,
            np.random.default_rng(random_state), fill=fill_other, per_sample_n=per_sample_n)

    sel_g1 = select(target_g1)
    sel_g2 = select(target_g2)
    if not sel_g1 or not sel_g2:
        raise ValueError("No cells selected for one or both scenarios.")

    seed_tag = random_state if random_state is not None else "None"
    g1_col = f"{col_prefixes[0]}_{seed_tag}_{dev:g}"
    g2_col = f"{col_prefixes[1]}_{seed_tag}_{dev:g}"
    adata.obs[g1_col] = adata.obs_names.isin(sel_g1)
    adata.obs[g2_col] = adata.obs_names.isin(sel_g2)

    print(f"{region_of_interest} fraction: mean={roi_mean:.4f}, SD={roi_std:.4f} "
          f"-> hi={hi:.4f}, lo={lo:.4f} (dev={dev:g})")
    print(f"Tagged {int(adata.obs[g1_col].sum())} cells -> {g1_col!r}; "
          f"{int(adata.obs[g2_col].sum())} cells -> {g2_col!r}")
    return adata


def impact_metrics(naive_df, anat_df, region_markers: set, alpha=0.05):
    """
    Compare a naive (pooled) DE result against an anatomy-aware result.

    Computes:
    - Jaccard overlap of significant gene sets at FDR `alpha`.
    - Median shift in absolute log2FC between methods (effect calibration).
    - Marker-leak rates: fraction of known region-marker genes among significant
      hits, used as a false-positive proxy for confounding by anatomy.

    Parameters
    ----------
    naive_df : pd.DataFrame
        Output of rank_DE (must have 'gene', 'padj', 'log2fc').
    anat_df : pd.DataFrame
        Output of wilcoxon_stouffer_by_region (must have 'gene', 'padj_stouffer',
        'log2fc_meta').
    region_markers : set
        Gene names known to be region-specific (used for leak calculation).
    alpha : float
        FDR threshold for significance.
    """
    genes = set(naive_df["gene"]).intersection(set(anat_df["gene"]))
    n = naive_df.set_index("gene").loc[genes]
    a = anat_df.set_index("gene").loc[genes]

    S_naive = set(n.index[n["padj"] <= alpha])
    S_anat  = set(a.index[a["padj_stouffer"] <= alpha])
    jaccard = len(S_naive & S_anat) / max(1, len(S_naive | S_anat))

    U = list(S_naive | S_anat)
    delta_abs_log2fc_median = float(
        (n.loc[U, "log2fc"].abs() - a.loc[U, "log2fc_meta"].abs()).median()
    )

    leak_naive = len(S_naive & region_markers) / max(1, len(S_naive))
    leak_anat  = len(S_anat  & region_markers) / max(1, len(S_anat))

    return {
        "jaccard":                  jaccard,
        "delta_abs_log2fc_median":  delta_abs_log2fc_median,
        "leak_naive":               leak_naive,
        "leak_anat":                leak_anat,
    }


def fix_stray_pixels(adata, region_col: str = "quint_region", slide_col: str = "slide_ID",
                     reassign_background: bool = False, coords_key: str | None = None):
    """
    Clean up stray / unassigned QUINT region labels on a spatial AnnData.

    1. "0,0,0" cells (atlas background) are either dropped (default) or, when
       ``reassign_background=True``, kept and treated as strays in step 2.
    2. Every remaining cell whose label is still a raw "R,G,B" triplet — a stray
       pixel the atlas never mapped to a named region — is reassigned to the
       region of its nearest ORIGINALLY-NAMED neighbour, computed per slide.

    A cell is an eligible neighbour only if its *original* label was a genuine
    region name — never another stray, a NaN, or a blank. The neighbour pool and
    the labels copied from it are both taken from a snapshot of the labels made
    before any reassignment, so a fixed stray can never seed another stray and
    every reassignment points at real anatomy.

    The pre-fix labels are preserved in ``obs[f"{region_col}_original"]``.

    Parameters
    ----------
    adata : AnnData
        Must carry per-cell region labels in ``obs[region_col]``, a slide key in
        ``obs[slide_col]``, and 2-D coordinates in ``obsm['spatial_fov']`` /
        ``obsm['spatial']`` (or x/y columns in ``obs``).
    region_col : str
        Column holding the QUINT region labels (default ``'quint_region'``).
    slide_col : str
        Column identifying each slide; KNN is done within a slide (default
        ``'slide_ID'``).
    reassign_background : bool
        If False (default) drop "0,0,0" background cells; if True keep them and
        reassign each to its nearest originally-named neighbour, exactly like any
        other stray.
    coords_key : str or None
        obsm key for the coordinates the KNN runs in. If None (default) prefer
        ``'spatial_fov'`` then ``'spatial'`` — 'spatial_fov' is the frame this
        pipeline builds its region labels in, so the search must match it.

    Returns
    -------
    AnnData
        Cleaned object. A new AnnData is returned; the input is not modified when
        any "0,0,0" cells are dropped, matching the original behaviour.
    """
    rgb_pattern = re.compile(r'^\d+,\d+,\d+$')

    # -- Step 1: handle "0,0,0" atlas background -----------------------------------
    region_str = adata.obs[region_col].astype(str)
    zero_mask = (region_str.values == '0,0,0')
    n_zero = int(zero_mask.sum())
    if n_zero:
        if reassign_background:
            # Keep them: "0,0,0" matches the RGB pattern, so it is picked up as a
            # stray below and reassigned to its nearest originally-named neighbour.
            print(f"Keeping {n_zero} '0,0,0' background cells for reassignment")
        else:
            adata = adata[~zero_mask].copy()
            region_str = adata.obs[region_col].astype(str)
            print(f"Removed {n_zero} cells with {region_col} == '0,0,0'")

    # -- Step 2: classify labels from the ORIGINAL snapshot ------------------------
    # Snapshot the labels up front; every downstream decision (which cells are
    # strays, which cells are valid neighbours, what label a stray inherits) is
    # made against THIS snapshot, so reassigned strays never influence each other.
    original = adata.obs[region_col].copy()

    stray_mask = region_str.str.match(rgb_pattern).fillna(False).values
    blank_mask = region_str.str.strip().isin(['', 'nan', 'None', '<NA>']).values
    named_mask = ~stray_mask & ~blank_mask   # cells ORIGINALLY given a real region name

    print(f"Stray (RGB) entries to reassign: {int(stray_mask.sum())} / {len(stray_mask)}")
    if stray_mask.any():
        print(original[stray_mask].value_counts(dropna=False))

    # -- Step 3: resolve coordinates -----------------------------------------------
    # This pipeline builds its region labels in the 'spatial_fov' frame (napari
    # viz, the polygon region assignment, and the QUINT export all use it), so the
    # KNN must run in that same frame. 'spatial' can be a different (e.g. FOV-local)
    # layout, which would scatter the reassignments. Prefer 'spatial_fov'; allow an
    # explicit override via coords_key.
    if coords_key is not None:
        if coords_key not in adata.obsm:
            raise ValueError(f"coords_key={coords_key!r} not in adata.obsm ({list(adata.obsm)})")
        coords = np.asarray(adata.obsm[coords_key])[:, :2]
        print(f"Using coordinates from obsm[{coords_key!r}]")
    elif 'spatial_fov' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial_fov'])[:, :2]
        print("Using coordinates from obsm['spatial_fov']")
    elif 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'])[:, :2]
        print("Using coordinates from obsm['spatial']")
    else:
        for xc, yc in [('x_centroid', 'y_centroid'), ('CenterX', 'CenterY'), ('x', 'y')]:
            if xc in adata.obs.columns and yc in adata.obs.columns:
                coords = adata.obs[[xc, yc]].values
                print(f"Using coordinate columns: {xc}, {yc}")
                break
        else:
            raise ValueError("No coordinate array found. Check obsm keys or obs columns.")

    # -- Step 4: per-slide KNN onto ORIGINALLY-NAMED cells -------------------------
    fixed = original.copy()
    obs_index = adata.obs.index

    for slide, slide_idx in adata.obs.groupby(slide_col, observed=True).groups.items():
        slide_pos = obs_index.get_indexer(slide_idx)

        slide_stray = stray_mask[slide_pos]
        slide_named = named_mask[slide_pos]

        n_stray = int(slide_stray.sum())
        if n_stray == 0:
            continue
        if slide_named.sum() == 0:
            print(f"WARNING: slide {slide} has no originally-named cells — "
                  f"cannot reassign {n_stray} stray cells")
            continue

        coords_named = coords[slide_pos][slide_named]
        coords_stray = coords[slide_pos][slide_stray]
        labels_named = original.values[slide_pos][slide_named]

        _, nn_idx = cKDTree(coords_named).query(coords_stray, k=1)
        fixed.iloc[slide_pos[slide_stray]] = labels_named[nn_idx]
        print(f"Slide {slide}: reassigned {n_stray} stray cells")

    adata.obs[f'{region_col}_original'] = original
    adata.obs[region_col] = fixed

    # A reassigned stray's old RGB label no longer applies to any cell; if the
    # column is categorical, drop those now-empty categories so they don't linger
    # in .cat.categories / value_counts. The _original column keeps them.
    if isinstance(adata.obs[region_col].dtype, pd.CategoricalDtype):
        adata.obs[region_col] = adata.obs[region_col].cat.remove_unused_categories()

    print(f"\n=== {region_col} after fix ===")
    print(adata.obs[region_col].value_counts(dropna=False))
    # Compare as strings so NaN==NaN doesn't inflate the count (NaN != NaN is True).
    n_changed = int((original.astype(str).values != fixed.astype(str).values).sum())
    print(f"Reassigned cells: {n_changed}")

    return adata
