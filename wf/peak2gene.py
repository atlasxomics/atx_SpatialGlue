import logging
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from scipy import io, sparse

from wf.coverage import _extract_barcode, resolve_rscript
import wf.genestats as gs


def _write_skip(out_dir: str, reason: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "peak2gene_skipped.txt"), "w") as f:
        f.write(f"{reason}\n")


def _is_integral_matrix(X, sample_n: int = 100000) -> bool:
    if sparse.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()
    if data.size == 0:
        return True
    if data.size > sample_n:
        rng = np.random.default_rng(1)
        data = data[rng.choice(data.size, size=sample_n, replace=False)]
    return bool(np.allclose(data, np.rint(data), rtol=0, atol=1e-6))


def _as_count_matrix(X, source: str):
    """Return a genes x cells integer sparse matrix for ArchR."""
    X = sparse.csc_matrix(X)
    integral = _is_integral_matrix(X)
    if source == "X" and not integral:
        raise ValueError(
            "RNA .X is fractional and no counts/raw matrix is available; "
            "Peak2Gene requires raw UMI counts."
        )
    if not integral:
        logging.warning(
            "RNA count matrix source '%s' contains non-integer values; rounding "
            "before passing to ArchR.",
            source,
        )
        X.data = np.rint(X.data)
    X.data = X.data.astype(np.int32, copy=False)
    X.eliminate_zeros()
    return X.transpose().tocsc()


def export_peak2gene_inputs(
    out_dir: str,
    rna,
    genes_of_interest: Optional[str] = None,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    counts, source = gs.get_rna_counts_matrix(rna)
    logging.info("Using RNA matrix source '%s' for Peak2Gene.", source)
    counts_genes_by_cells = _as_count_matrix(counts, source)

    cells = pd.DataFrame({
        "cell_id": rna.obs_names.astype(str),
        "barcode": [_extract_barcode(cell) for cell in rna.obs_names.astype(str)],
    })
    if "sg_clusters" in rna.obs:
        cells["sg_clusters"] = rna.obs["sg_clusters"].astype(str).values
    genes = pd.DataFrame({"gene": rna.var_names.astype(str)})

    counts_path = os.path.join(out_dir, "rna_counts_genes_by_cells.mtx")
    cells_path = os.path.join(out_dir, "rna_cells.csv")
    genes_path = os.path.join(out_dir, "rna_genes.csv")
    io.mmwrite(counts_path, counts_genes_by_cells)
    cells.to_csv(cells_path, index=False)
    genes.to_csv(genes_path, index=False)

    goi_path = os.path.join(out_dir, "genes_of_interest.txt")
    goi = []
    if genes_of_interest:
        goi = [
            gene.strip()
            for gene in genes_of_interest.replace("\n", ",").split(",")
            if gene.strip()
        ]
    with open(goi_path, "w") as f:
        f.write("\n".join(goi))
        if goi:
            f.write("\n")

    return {
        "counts": counts_path,
        "cells": cells_path,
        "genes": genes_path,
        "genes_of_interest": goi_path,
    }


def run_archr_peak2gene(
    out_dir: str,
    archr_project_path: str,
    rna,
    genes_of_interest: Optional[str] = None,
) -> None:
    logging.info("Preparing RNA count matrix for ArchR Peak2Gene...")
    with tempfile.TemporaryDirectory(prefix="peak2gene_inputs_") as input_dir:
        input_paths = export_peak2gene_inputs(input_dir, rna, genes_of_interest)

        script_path = os.path.join(os.path.dirname(__file__), "archr_peak2gene.R")
        if not os.path.exists(script_path):
            raise RuntimeError(f"Missing ArchR Peak2Gene helper script: {script_path}")

        rscript = resolve_rscript()
        subprocess.run(
            [
                rscript,
                "-e",
                "library(ArchR); cat('Using ArchR ', as.character(packageVersion('ArchR')), '\\n', sep = '')",
            ],
            check=True,
        )
        subprocess.run(
            [
                rscript,
                script_path,
                archr_project_path,
                input_paths["counts"],
                input_paths["cells"],
                input_paths["genes"],
                out_dir,
                input_paths["genes_of_interest"],
            ],
            check=True,
        )


def write_peak2gene_skip(out_dir: str, reason: str) -> None:
    _write_skip(out_dir, reason)
