from __future__ import annotations

from .load_functions import (cosmx, visium, xenium)

__all__ = ["cosmx", "visium", "xenium"]



# ---------- public API ----------
r'''def load_(path: str | Path, **kw) -> AnnData:
    """
    Load a dataset directory into AnnData.

    Resolution order:
      1) sidecar marker in the directory (.cosmx/.xenium/.visium)
      2) heuristics (folder structure / filenames)

    Use explicit forms if you like: load.cosmx(<dir>), load.visium(<dir>), load.xenium(<dir>)
    """
    p = Path(path)

    platform = _platform_by_hueristic(p)
    if platform == "cosmx":  return load_cosmx(p, **kw)
    if platform == "visium": return load_visium(p, **kw)
    if platform == "xenium": return load_xenium(p, **kw)

    raise ValueError(
        f"Could not determine platform under {p}. "
    )
'''
# “method-like” explicit calls: load.cosmx(<dir>), etc.
r'''def cosmx(path: str | Path, **kw) -> AnnData:  return cosmx(Path(path), **kw)
def visium(path: str | Path, **kw) -> AnnData: return visium(Path(path), **kw)
def xenium(path: str | Path, **kw) -> AnnData: return xenium(Path(path), **kw)'''

__all__ = ["cosmx", "visium", "xenium"]
