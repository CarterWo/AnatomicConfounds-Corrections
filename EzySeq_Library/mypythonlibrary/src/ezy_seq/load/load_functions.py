

from __future__ import annotations
from pathlib import Path
import os
import gzip
import shutil
import pandas as pd
import numpy as np
import squidpy as sq
import scanpy as sc
import re
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy import sparse as sp



def ensure_is_list_of_adatas(adatas,on_single_message: str = "You passed a single AnnData. Please pass a list of AnnData objects.") -> list[AnnData]:
    # Single AnnData → raise with your custom message
    if isinstance(adatas, AnnData):
        raise TypeError(on_single_message)
    if adatas is None or not hasattr(adatas, "__iter__"):# Checks if it has the attribute of being iterable
        raise TypeError("Expected an iterable (list/tuple) of AnnData objects.")
    # Convert tuples, etc., to list
    adatas = list(adatas)
    # Validate contents
    if not all(isinstance(a, AnnData) for a in adatas):
        raise TypeError("All items must be AnnData objects.")
    if len(adatas) == 0:
        raise ValueError("Empty list provided.")
    return adatas

def decompress_targz(file_path: str | os.PathLike, dest_dir: str | os.PathLike) -> Path:
    """
    Decompress *.gz into dest_dir. Returns the output path.
    Supports single-file gzip (e.g., myfile.csv.gz). Not for .tar.gz archives.
    """
    src = Path(file_path)
    dest_dir = Path(dest_dir)
    if not src.name.endswith(".gz"):
        raise ValueError(f"Unsupported compression type: {src}")
    out_path = dest_dir / src.name[:-3]  # strip .gz
    if not out_path.exists():
        print(f"Unzipping {src.name} -> {out_path.name}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(src, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_path



def cosmx(export_dir: str | os.PathLike, expect_weird_fov: bool = True) -> list[sc.AnnData]:
    """
    Walk CosMx export directory, handling both standard and inverted folder structures, 
    and load adatas.

    Supported Structures:
    1. Standard: export_dir/SampleName/flatFiles/CsvFolder/*.csv
    2. Inverted: export_dir/flatFiles/CsvFolder/*.csv
    """
    export_dir = Path(export_dir)
    adatas = []
    
    # ---------------------------------------------------------
    # PHASE 1: DISCOVER PATHS
    # ---------------------------------------------------------
    # We want to populate this list with (Path_to_CSV_Folder, Folder_Name)
    paths: list[tuple[Path, str]] = []
    
    flatfiles_at_root = export_dir / "flatFiles"

    # CASE A: 'flatFiles' is directly inside the export_dir (The "Inverted" case)
    if flatfiles_at_root.is_dir():
        print(f"Detected 'flatFiles' folder directly in root: {flatfiles_at_root}")
        for csv_folder in os.listdir(flatfiles_at_root):
            csv_folder_dir = flatfiles_at_root / csv_folder
            if csv_folder_dir.is_dir():
                paths.append((csv_folder_dir, csv_folder))

    # CASE B: 'flatFiles' is inside subfolders (The "Standard" case)
    else:
        print(f"Scanning subdirectories for 'flatFiles'...")
        for sample_name in os.listdir(export_dir):
            sample_dir = export_dir / sample_name
            if not sample_dir.is_dir():
                continue
            
            flatFiles_path = sample_dir / "flatFiles"
            if flatFiles_path.is_dir():
                for csv_folder in os.listdir(flatFiles_path):
                    csv_folder_dir = flatFiles_path / csv_folder
                    if csv_folder_dir.is_dir():
                        paths.append((csv_folder_dir, csv_folder))

    if not paths:
        print("[warn] No valid CSV folders found in either structure.")
        return []

    print(f"Found {len(paths)} dataset(s) to process.")

    # ---------------------------------------------------------
    # PHASE 2: PROCESS DATA
    # ---------------------------------------------------------
    for csv_folder_dir, name in paths:
        print(f"Processing: {name}")
        
        # 1) Decompress any *.gz CSVs in this folder
        for csv_file in os.listdir(csv_folder_dir):
            if csv_file.endswith(".gz"):
                # Assuming decompress_targz is defined elsewhere in your code
                decompress_targz(csv_folder_dir / csv_file, csv_folder_dir)

        # 2) Build expected filenames
        counts = csv_folder_dir / f"{name}_exprMat_file.csv"
        fovs   = csv_folder_dir / f"{name}_fov_positions_file.csv"
        fovs_adj = csv_folder_dir / f"{name}_fov_positions_file_adj.csv"
        meta   = csv_folder_dir / f"{name}_metadata_file.csv"

        # 3) Validate existence
        missing_files = False
        for file_path in [counts, fovs, meta]:
             if not file_path.exists():
                print(f"[error] Missing expected file: {file_path.name}")
                missing_files = True
        
        if missing_files:
            continue

        # 4) Normalize FOV column name
        # We write to a temp file so we don't rely on 'expect_weird_fov' arg logic alone,
        # but actually check the file content.
        try:
            fov_df = pd.read_csv(fovs)
            if "FOV" in fov_df.columns:
                fov_df.rename(columns={'FOV': 'fov'}, inplace=True)
                fov_df.to_csv(fovs_adj, index=False)
                input_fovs = fovs_adj
            else:
                input_fovs = fovs
        except Exception as e:
            print(f"[error] Could not read FOV file for {name}: {e}")
            continue

        # 5) Read with squidpy
        try:
            raw_adata = sq.read.nanostring(
                path=str(csv_folder_dir),
                counts_file=counts.name,
                meta_file=meta.name,
                fov_file=input_fovs.name
            )
        except Exception as e:
            print(f"[error] Squidpy failed to load {name}: {e}")
            continue

        # 6) QC annotations
        # Ensure 'NegPrb' exists to avoid errors during QC calculation
        raw_adata.var["NegPrb"] = raw_adata.var_names.str.startswith("Negative")
        raw_adata.var["FalseCode"] = raw_adata.var_names.str.startswith("SystemControl")
        
        sc.pp.calculate_qc_metrics(
            raw_adata, qc_vars=["NegPrb", "FalseCode"], inplace=True
        )

        # Calculate negative mean
        neg_mask = raw_adata.var["NegPrb"].fillna(False)
        neg_genes = raw_adata.var_names[neg_mask]
        
        if len(neg_genes) > 0:
            Xneg = raw_adata[:, neg_genes].X
            # Handle sparse matrix if necessary (though usually dense here)
            if hasattr(Xneg, "toarray"):
                Xneg = Xneg.toarray()
            raw_adata.obs["neg_mean"] = Xneg.mean(axis=1)
        else:
            raw_adata.obs["neg_mean"] = 0

        raw_adata.obs["slide_ID"] = name
        adatas.append(raw_adata)

    return adatas

def xenium(export_dir: str | os.PathLike, expect_weird_fov: bool = True)->list[AnnData]:
    """
    Walk CosMx export directory, ensure flatFiles CSVs are ready, and load adatas.

    Parameters
    ----------
    export_dir : folder containing per-sample subfolders, each with a 'flatFiles' dir.
    expect_upper_fov : if True, force FOV column to be 'FOV'; if False, 'fov'.
    """
    export_dir = Path(export_dir)
   

    return export_dir
def visium(export_dir: str | os.PathLike, expect_weird_fov: bool = True)->list[AnnData]:
    """
    Walk CosMx export directory, ensure flatFiles CSVs are ready, and load adatas.

    Parameters
    ----------
    export_dir : folder containing per-sample subfolders, each with a 'flatFiles' dir.
    expect_upper_fov : if True, force FOV column to be 'FOV'; if False, 'fov'.
    """
    export_dir = Path(export_dir)
   

    return export_dir