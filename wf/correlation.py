import logging
import numpy as np
import pandas as pd
import sys

from anndata import AnnData
from numpy import ndarray
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def get_corr_df(
    array1: ndarray,
    array2: ndarray,
    genes: List[str],
    array1_name: str = "RNA",
    array2_name: str = "GA",
) -> pd.DataFrame:
    """"""

    rhos = np.empty(len(genes), dtype=np.float32)
    pvals = np.empty(len(genes), dtype=np.float64)

    for j in range(len(genes)):
        rho, p = spearmanr(array2[:, j], array1[:, j])
        rhos[j] = np.nan_to_num(rho, nan=0.0)
        pvals[j] = np.nan_to_num(p, nan=1.0)

    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

    res = pd.DataFrame({
        "gene": genes.astype(str).values,
        "spearman_rho": rhos,
        "pval": pvals,
        "qval_bh": qvals,
        f"mean_{array1_name}": array1.mean(axis=0),
        f"mean_{array2_name}": array2.mean(axis=0),
    })

    res["abs_rho"] = res["spearman_rho"].abs()
    res = res.sort_values("abs_rho", ascending=False)

    return res


def log_norm(array: ndarray, scaleto: int) -> ndarray:
    lib = array.sum(axis=1, keepdims=True)
    lib[lib < 1] = 1
    array_norm = np.log1p((array / lib) * scaleto).astype(np.float32)

    return array_norm


def synch_adata(adata1: AnnData, adata2: AnnData) -> Tuple[AnnData, AnnData]:

    # Align cells
    common = adata1.obs_names.intersection(adata2.obs_names)

    if len(common) == 0:
        raise RuntimeError(
            "Could not find common cells between transcriptome and gene accessibility data; please ensure the input files are from the same experiment."
        )

    adata1 = adata1[common, :].copy()
    adata2 = adata2[common, :].copy()
    adata2 = adata2[adata2.obs_names.get_indexer(adata1.obs_names), :].copy()
    assert (adata1.obs_names == adata2.obs_names).all()

    # Align genes
    adata1_feats_up = pd.Index(adata1.var_names.astype(str)).str.upper()
    adata2_feats_up = pd.Index(adata2.var_names.astype(str)).str.upper()
    logging.info(f"Object 1 feats preview: {adata1_feats_up[:5]}")
    logging.info(f"Object 2 feats preview: {adata2_feats_up[:5]}")
    feats_common = adata1_feats_up.intersection(adata2_feats_up)
    if len(feats_common) == 0:
        raise RuntimeError("No features overlap. Check gene naming (symbols vs IDs).")
    if len(feats_common) < 500:
        logging.warning(
            f"Only {len(feats_common)} features overlap. Check gene naming (symbols vs IDs)."
        )

    # indexers
    logging.info("Reindexing...")
    adata1_idx = adata1_feats_up.get_indexer(feats_common)
    adata2_idx = adata2_feats_up.get_indexer(feats_common)

    logging.info("Filtering AnnData by common genes...")
    adata1_sub = adata1[:, adata1_idx].copy()
    adata2_sub = adata2[:,  adata2_idx].copy()

    return adata1_sub, adata2_sub
