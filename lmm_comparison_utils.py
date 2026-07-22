"""
lmm_comparison_utils.py
=======================
Compatibility shim for the post-DE figure scripts.

The ``Post_DE_Processing/visualize_*.py`` scripts import ``lmm_comparison_utils``
— the original, pre-packaging name of the DE-comparison helpers. That code now
lives in the repo's packaged library as ``ezy_seq.lmm``. This module re-exports
that code so the figure scripts keep working unchanged, and bridges the one
function that was renamed along the way:

    parse_seurat_dream_filename()  ->  ezy_seq.lmm.parse_de_filename()

(both return ``{'direction', 'cell_type', 'model', 'annotation'}`` or ``None``).

Why load the submodule file directly instead of ``import ezy_seq.lmm``
---------------------------------------------------------------------
Importing the ``ezy_seq`` *package* runs ``ezy_seq/__init__.py``, which eagerly
pulls in the spatial stack (``squidpy`` -> ``spatialdata`` -> ``dask``). The
figure scripts need none of that, and that stack is easy to have broken/absent
in a lean plotting environment. ``ezy_seq/lmm/__init__.py`` is self-contained
(pandas/numpy/scipy only), so we exec it directly and skip the heavy package
import entirely. Reusing that one implementation — rather than a second copy —
keeps the figures and ``aggregate_seeds.py`` from silently diverging.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

# Locate the repo's bundled ezy_seq/lmm module file (walk up so CWD doesn't matter).
_lmm_init = None
for _d in [_Path(__file__).resolve().parent, *_Path(__file__).resolve().parents]:
    _cand = _d / "EzySeq_Library" / "mypythonlibrary" / "src" / "ezy_seq" / "lmm" / "__init__.py"
    if _cand.is_file():
        _lmm_init = _cand
        break
if _lmm_init is None:
    raise ImportError(
        "lmm_comparison_utils: could not find EzySeq_Library/.../ezy_seq/lmm/__init__.py "
        "under this file's parent directories."
    )

# Load ezy_seq.lmm as a standalone module (no package __init__, no squidpy).
_spec = _ilu.spec_from_file_location("_ezy_seq_lmm_standalone", _lmm_init)
_lmm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lmm)

# Re-export every public name (load_de_file, standardize_columns,
# simplify_celltype_name, aggregate_seed_files, aggregate_seed_directory, …).
for _name in dir(_lmm):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_lmm, _name)

# Legacy alias the figure scripts still use for the DE-filename parser.
parse_seurat_dream_filename = _lmm.parse_de_filename
