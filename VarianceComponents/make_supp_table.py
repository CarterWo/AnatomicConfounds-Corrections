"""Builds Supplementary Table 3 (variance components) from the per-gene result CSVs.

usage: python3 make_supp_table.py <results_dir> <out.xlsx> <n_cosmx_cells>

The first sheet summarizes each component across genes (median, IQR, 90th percentile,
mean) as a percentage of total variance; the other sheets list the per-gene values.
"""
import csv
import statistics as st
import sys

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

VC_DIR, OUT, N_COSMX = sys.argv[1], sys.argv[2], int(sys.argv[3])

SETS = [
    ("CosMx (this study)", "Napari (4 broad regions + unassigned)", "vc_cosmx_napari.csv", "Per-gene CosMx Napari"),
    ("CosMx (this study)", "QUINT (36 atlas regions)", "vc_cosmx_quint.csv", "Per-gene CosMx QUINT"),
    ("MERFISH (Zhuang-ABCA-1-4)", "Napari-equivalent (CCFv3 division)", "vc_merfish_napari.csv", "Per-gene MERFISH Napari-eq"),
    ("MERFISH (Zhuang-ABCA-1-4)", "QUINT-equivalent (CCFv3 structure)", "vc_merfish_quint.csv", "Per-gene MERFISH QUINT-eq"),
]
COMPS = [("cell_type", "Cell type"), ("region", "Anatomic region"), ("sample_ID", "Biological sample"),
         ("Treatment", "Treatment group"), ("log_depth", "Log unique genes count"), ("Residuals", "Residual")]


def percentile(values, p):
    # linear interpolation, same as R's default quantile()
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


thin = Side(style="thin")
hdr_fill = PatternFill("solid", fgColor="FFD9D9D9")
band_fill = PatternFill("solid", fgColor="FFF7F7F7")
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Table S3"
hdr = ["Dataset", "Region labels", "Genes", "Variance component", "Median (%)",
       "25th percentile (%)", "75th percentile (%)", "90th percentile (%)", "Mean (%)"]
for j, h in enumerate(hdr, 1):
    c = ws.cell(1, j, h)
    c.font = Font(name="Calibri", size=11, bold=True)
    c.fill = hdr_fill
    c.alignment = Alignment(wrap_text=True, vertical="center")
    c.border = Border(bottom=thin)

row, band, per = 2, False, {}
for ds, lab, fn, sheet in SETS:
    rows = list(csv.DictReader(open(f"{VC_DIR}/{fn}")))
    per[sheet] = rows
    band = not band
    for key, name in COMPS:
        if key not in rows[0]:
            continue
        v = [100 * float(r[key]) for r in rows]
        vals = [ds, lab, len(rows), name, round(percentile(v, .5), 2), round(percentile(v, .25), 2),
                round(percentile(v, .75), 2), round(percentile(v, .9), 2), round(st.mean(v), 2)]
        for j, val in enumerate(vals, 1):
            c = ws.cell(row, j, val)
            c.font = Font(name="Calibri", size=11)
            if band:
                c.fill = band_fill
        row += 1
for col, w in zip("ABCDEFGHI", [28, 36, 8, 24, 12, 13, 13, 13, 10]):
    ws.column_dimensions[col].width = w
ws.freeze_panes = "A2"

row += 1
cap = (
    "Supplementary Table 3. Estimated variance components for the two real datasets. For each gene, a "
    "linear mixed model with the covariates of the deployed Dream models (log-scaled unique genes count, "
    "plus treatment group for CosMx, as fixed effects; cell type, anatomic region, and biological sample as "
    "random intercepts) was fit to log-normalized expression with lme4 (REML), and the variance attributable "
    "to each term was expressed as a percentage of the total. Values summarize the per-gene distribution. "
    f"CosMx: {N_COSMX:,} cells from the six Stroke-FMT and Healthy-FMT samples (8,000 sampled per sample, "
    "with olfactory bulb cells then removed as in the main analysis; 32 cell types), 1,000 genes. MERFISH "
    "(Zhuang-ABCA-1-4): the 20,000-cell subsample (5,000 per animal) from four wild-type mice used for the "
    "1.0x external validation (cell types: excitatory neuron, astrocyte, microglia, other), 1,107 genes "
    "detected in at least 1% of cells; no treatment term is included because this dataset has no treatment "
    "contrast. Per-gene estimates are provided in the remaining sheets."
)
c = ws.cell(row, 1, cap)
c.font = Font(name="Calibri", size=9)
c.alignment = Alignment(wrap_text=True, vertical="top")
ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=9)
ws.row_dimensions[row].height = 120

for sheet, rows in per.items():
    s = wb.create_sheet(sheet)
    keys = [k for k, _ in COMPS if k in rows[0]]
    heads = ["Gene"] + [n + " (%)" for k, n in COMPS if k in rows[0]] + ["Singular fit"]
    for j, h in enumerate(heads, 1):
        c = s.cell(1, j, h)
        c.font = Font(bold=True)
        c.fill = hdr_fill
        c.border = Border(bottom=thin)
    for i, r in enumerate(sorted(rows, key=lambda r: -float(r["region"])), 2):
        s.cell(i, 1, r["gene"])
        for j, k in enumerate(keys, 2):
            s.cell(i, j, round(100 * float(r[k]), 3))
        s.cell(i, len(keys) + 2, r["singular"].upper() == "TRUE")
    s.freeze_panes = "B2"
    s.column_dimensions["A"].width = 16
wb.save(OUT)
print("saved", OUT)
