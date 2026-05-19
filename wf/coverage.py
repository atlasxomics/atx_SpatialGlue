import glob
import logging
import os
import subprocess
from typing import Optional

import pandas as pd
import snapatac2 as snap

import wf.utils as utils


def candidate_cluster_columns(adata, exclude: Optional[set[str]] = None) -> list[str]:
    exclude = exclude or set()
    candidates = []
    for col in adata.obs.columns:
        if col in exclude:
            continue
        col_lower = str(col).lower()
        if not any(token in col_lower for token in ["cluster", "leiden", "louvain"]):
            continue
        values = adata.obs[col]
        n_unique = values.nunique(dropna=True)
        if n_unique < 2 or n_unique >= adata.n_obs:
            continue
        candidates.append(col)
    return candidates


def export_coverage_group(atac, groupby: str, out_dir: str, suffix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    before = set(glob.glob("*.bw"))
    snap.ex.export_coverage(
        atac,
        groupby=groupby,
        suffix=suffix,
        bin_size=10,
        output_format="bigwig",
    )
    after = set(glob.glob("*.bw"))
    new_bws = sorted(after - before)
    if new_bws:
        subprocess.run(["mv"] + new_bws + [out_dir], check=True)
    else:
        logging.warning(f"No bigWig files were created for {groupby}.")


def export_cluster_coverages(out_dir: str, atac, rna) -> None:
    logging.info("Creating coverage tracks...")

    glue_dir = os.path.join(out_dir, "glue_cluster_coverages")
    export_coverage_group(
        atac,
        groupby="sg_leiden_merged",
        out_dir=glue_dir,
        suffix="_cluster.bw",
    )

    reserved = {"sg_leiden", "sg_leiden_merged"}
    rna_cluster_cols = candidate_cluster_columns(rna, exclude=reserved)
    if rna_cluster_cols:
        rna_dir = os.path.join(out_dir, "rna_cluster_coverages")
        for col in rna_cluster_cols:
            safe_col = utils.safe_name(col)
            groupby = f"rna_{safe_col}"
            atac.obs[groupby] = pd.Categorical(
                rna.obs.loc[atac.obs_names, col].astype(str).values
            )
            logging.info(f"Exporting RNA cluster coverage tracks for '{col}'.")
            export_coverage_group(
                atac,
                groupby=groupby,
                out_dir=os.path.join(rna_dir, safe_col),
                suffix=f"_{groupby}.bw",
            )
    else:
        logging.info("No pre-existing RNA cluster columns found for coverage export.")

    atac_cluster_cols = candidate_cluster_columns(atac, exclude=reserved)
    atac_cluster_cols = [
        col for col in atac_cluster_cols if not str(col).startswith("rna_")
    ]
    if atac_cluster_cols:
        atac_dir = os.path.join(out_dir, "atac_cluster_coverages")
        for col in atac_cluster_cols:
            safe_col = utils.safe_name(col)
            logging.info(f"Exporting ATAC cluster coverage tracks for '{col}'.")
            export_coverage_group(
                atac,
                groupby=col,
                out_dir=os.path.join(atac_dir, safe_col),
                suffix=f"_{safe_col}.bw",
            )
    else:
        logging.info("No pre-existing ATAC cluster columns found for coverage export.")

    logging.info("Finished coverage track export.")
