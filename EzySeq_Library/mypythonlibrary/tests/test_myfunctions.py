import ezy_seq as ezy
import math
import pytest
import anndata as ad
def test_add():
    assert ezy.add(2, 3) == 5


def test_version_semver_like():
    # basic sanity check the version string exists and looks like x.y.z
    parts = ezy.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)

def test_load_adata():
    adatas=ezy.load_adata(r"E:\Carter-Woods\CosMx export\Test_fldr")
    assert all(isinstance(a, ad.AnnData) for a in adatas)