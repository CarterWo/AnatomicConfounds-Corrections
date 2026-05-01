from .ezyfunctions import (
    rank_DE, apply_annotation, find_elbow_n_genes, filter_and_normalize,
    pseudobulk_export, ensure_is_list_of_adatas, combine_adatas, read_dictionary,
    impact_metrics, set_region_abundance_by_FMT,
)
import importlib as _importlib

load = _importlib.import_module("ezy_seq.load")
plot = _importlib.import_module("ezy_seq.plot")
lmm = _importlib.import_module("ezy_seq.lmm")

__all__ = [
    "rank_DE", "apply_annotation", "find_elbow_n_genes", "filter_and_normalize",
    "pseudobulk_export", "ensure_is_list_of_adatas", "combine_adatas", "read_dictionary",
    "impact_metrics", "set_region_abundance_by_FMT", "load", "plot", "lmm",
]
__version__ = "0.1.0"
