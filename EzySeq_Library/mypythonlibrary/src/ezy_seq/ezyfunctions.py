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
        sc.pp.filter_cells(adata_manip, min_genes=min_gene_cnt)
        sc.pp.filter_cells(adata_manip, min_counts=min_t_cnt)
        sc.pp.filter_genes(adata_manip, min_cells=min_cell_cnt)
        adata_manip = adata_manip[:, adata_manip.var["NegPrb"].fillna(False) == False]
        adata_manip = adata_manip[:, adata_manip.var["FalseCode"].fillna(False) == False].copy()
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
    out = pd.Series(0, index=baseline_counts.index, dtype=int)
    total = int(total)
    if total <= 0 or baseline_counts.sum() <= 0 or caps.sum() <= 0:
        return out
    weights = baseline_counts.clip(lower=0).astype(float)
    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
    raw = weights * total
    base = np.floor(raw).astype(int)
    take = base.clip(upper=caps.astype(int))
    out += take
    rem = total - int(take.sum())
    if rem <= 0:
        return out
    frac = (raw - base).sort_values(ascending=False)
    for idx in frac.index:
        if rem == 0:
            break
        if out[idx] < int(caps.get(idx, 0)):
            out[idx] += 1
            rem -= 1
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
    avail = obs_rf.groupby([fmt_col, region_col]).size().unstack(fill_value=0)
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


def _allocate_group(row: pd.Series, tgt, t: int) -> pd.Series:
    """
    Turn availability + a target into per-region desired counts for a budget `t`.

    `row` is region -> available count. `tgt` is a {region: count|fraction} dict
    (or None). Specified regions hit their target; the leftover budget fills the
    unspecified regions at their baseline proportions, everything capped at
    availability and trimmed so the total never exceeds `t`.
    """
    baseline = (row / row.sum()).fillna(0)
    desired = pd.Series(0, index=row.index, dtype=int)

    if tgt:
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
    return desired.clip(upper=row)


def _sample_from_pool(sub, desired: pd.Series, region_col, rng) -> list:
    """Randomly draw `desired[region]` obs-index labels from `sub` per region."""
    out = []
    for region, need in desired.items():
        n = int(need)
        if n <= 0:
            continue
        pool = sub.index[sub[region_col] == region].to_numpy()
        out.extend(pool.tolist() if n >= len(pool) else
                   rng.choice(pool, size=n, replace=False).tolist())
    return out


def _select_by_fmt_targets(avail, obs_rf, target_by_fmt, total_per_fmt,
                           region_col, fmt_col, rng) -> set:
    """
    Pooled (FMT-wise) selection: all cells within each FMT group are pooled across
    samples, then subsampled to hit that group's per-region targets on a single
    per-FMT budget. Returns the selected obs-index labels as a set.
    """
    selected = []
    for fmt in avail.index:
        row = avail.loc[fmt]
        t = _budget_for(total_per_fmt, fmt, int(row.sum()))
        desired = _allocate_group(row, target_by_fmt.get(fmt), t)
        sub = obs_rf[obs_rf[fmt_col] == fmt]
        selected += _sample_from_pool(sub, desired, region_col, rng)
    return set(selected)


def _max_feasible_n(row: pd.Series, tgt) -> int:
    """
    Largest total cell count N a single sample can supply while hitting target `tgt`
    exactly, with unspecified regions filled at baseline proportions.

    For fraction targets, each specified region r constrains N <= avail_r / frac_r,
    and the unspecified regions jointly constrain N <= (sum unspecified avail) /
    (1 - sum specified fracs). Returns row.sum() for a None or count-based target.
    """
    total = int(row.sum())
    if not tgt:
        return total
    if all(isinstance(v, (int, np.integer)) for v in tgt.values()):
        return total  # explicit counts carry no fraction-based ceiling

    frac = {r: float(v) for r, v in tgt.items() if r in row.index}
    sf = sum(frac.values())
    if sf > 1.0:
        frac = {r: v / sf for r, v in frac.items()}

    caps = [row[r] / f for r, f in frac.items() if f > 0]
    spec = set(frac)
    rem_frac = 1.0 - sum(frac.values())
    if rem_frac > 1e-9:
        unspec_avail = int(row[[r for r in row.index if r not in spec]].sum())
        caps.append(unspec_avail / rem_frac)
    return int(np.floor(min(caps))) if caps else total


def _select_by_sample_targets(obs_rsf, target_by_fmt, total_per_fmt,
                              region_col, fmt_col, sample_col, rng) -> set:
    """
    Per-sample selection with equal, maximised contribution.

    Every sample is subsampled to its FMT's target fraction, and all samples in the
    scenario contribute the *same* number of cells N. N is chosen as large as
    possible subject to every sample being able to hit its target:

        N = min over samples of `_max_feasible_n(sample, its FMT target)`

    so one cell-poor sample sets the shared per-sample count. If `total_per_fmt` is
    given it acts as an optional ceiling (N also <= budget // n_samples per FMT);
    when None, N is the pure feasibility maximum. Returns the selected labels as a set.
    """
    avail = obs_rsf.groupby([fmt_col, sample_col, region_col]).size().unstack(fill_value=0)
    fmts = list(avail.index.get_level_values(fmt_col).unique())

    feasible = []           # (N_i, fmt, sample)
    ceilings = []
    for fmt in fmts:
        fmt_block = avail.xs(fmt, level=fmt_col)          # rows=sample, cols=region
        n_samp = max(1, len(fmt_block.index))
        tgt = target_by_fmt.get(fmt)
        for samp in fmt_block.index:
            feasible.append((_max_feasible_n(fmt_block.loc[samp], tgt), fmt, samp))
        if total_per_fmt is not None:
            budget = _budget_for(total_per_fmt, fmt, int(fmt_block.values.sum()))
            ceilings.append(budget // n_samp)

    binding = min(feasible, key=lambda x: x[0])
    N = binding[0]
    if ceilings:
        N = min(N, min(ceilings))
    if N <= 0:
        raise ValueError(
            f"Max equal per-sample contribution is 0: sample {binding[2]!r} "
            f"(FMT {binding[1]!r}) cannot reach its target fraction. "
            f"Lower `dev` or use balance='fmt'."
        )
    print(f"[sample-balanced] N={N} cells/sample x {len(feasible)} samples "
          f"(limited by {binding[2]!r} in FMT {binding[1]!r}, max feasible {binding[0]})")

    selected = []
    for fmt in fmts:
        fmt_block = avail.xs(fmt, level=fmt_col)
        tgt = target_by_fmt.get(fmt)
        for samp in fmt_block.index:
            desired = _allocate_group(fmt_block.loc[samp], tgt, N)
            sub = obs_rsf[(obs_rsf[fmt_col] == fmt) & (obs_rsf[sample_col] == samp)]
            selected += _sample_from_pool(sub, desired, region_col, rng)
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

    adata = adata.copy() if copy else adata

    for col in (region_col, fmt_col, sample_col):
        if col not in adata.obs:
            raise KeyError(f"{col!r} not found in adata.obs")

    # Cross-sample mean/SD of the region-of-interest fraction.
    counts_ps = adata.obs.groupby([sample_col, region_col]).size().unstack(fill_value=0)
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

    if balance == "fmt":
        obs_rf = adata.obs[[region_col, fmt_col]].dropna().copy()
        avail = obs_rf.groupby([fmt_col, region_col]).size().unstack(fill_value=0)
        select = lambda tgt: _select_by_fmt_targets(
            avail, obs_rf, tgt, total_per_fmt, region_col, fmt_col,
            np.random.default_rng(random_state))
    else:  # "sample"
        obs_rsf = adata.obs[[region_col, fmt_col, sample_col]].dropna().copy()
        select = lambda tgt: _select_by_sample_targets(
            obs_rsf, tgt, total_per_fmt, region_col, fmt_col, sample_col,
            np.random.default_rng(random_state))

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
