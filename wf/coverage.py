import glob
import inspect
import logging
import os
import re
import shutil
import subprocess
from typing import Optional

import pandas as pd

import wf.utils as utils

_BARCODE_REGEX = re.compile(r"([ATCG]{16})", re.IGNORECASE)


def _extract_barcode(value: str) -> Optional[str]:
    match = _BARCODE_REGEX.search(str(value).upper())
    if match is None:
        return None
    return match.group(1)


def resolve_rscript() -> str:
    rscript = os.environ.get("ARCHR_RSCRIPT")
    if not rscript:
        if os.path.exists("/usr/local/bin/Rscript"):
            rscript = "/usr/local/bin/Rscript"
        elif os.path.exists("/usr/bin/Rscript"):
            rscript = "/usr/bin/Rscript"
        else:
            rscript = "Rscript"
    if os.path.sep not in rscript and shutil.which(rscript) is None:
        raise RuntimeError(f"Could not find Rscript executable: {rscript}")
    return rscript


def coverage_threads() -> int:
    raw = os.environ.get("COVERAGE_THREADS") or os.environ.get("ARCHR_THREADS")
    try:
        threads = int(raw) if raw is not None else 32
    except ValueError:
        threads = 32
    return max(1, threads)


def configure_thread_env(threads: int) -> None:
    for key in [
        "ARCHR_THREADS",
        "COVERAGE_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(key, str(threads))


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


def valid_group_column(adata, col: str) -> bool:
    if col not in adata.obs.columns:
        return False
    n_unique = adata.obs[col].nunique(dropna=True)
    return 1 < n_unique < adata.n_obs


def export_coverage_group(atac, groupby: str, out_dir: str, suffix: str) -> None:
    import snapatac2 as snap

    os.makedirs(out_dir, exist_ok=True)
    before = set(glob.glob("*.bw"))
    threads = coverage_threads()
    configure_thread_env(threads)
    kwargs = {}
    try:
        params = inspect.signature(snap.ex.export_coverage).parameters
        for name in ["n_jobs", "n_threads", "num_threads", "threads"]:
            if name in params:
                kwargs[name] = threads
                break
    except (TypeError, ValueError):
        logging.info("Could not inspect SnapATAC2 export_coverage thread kwargs.")
    snap.ex.export_coverage(
        atac,
        groupby=groupby,
        suffix=suffix,
        bin_size=10,
        output_format="bigwig",
        **kwargs,
    )
    after = set(glob.glob("*.bw"))
    new_bws = sorted(after - before)
    if new_bws:
        subprocess.run(["mv"] + new_bws + [out_dir], check=True)
    else:
        logging.warning(f"No bigWig files were created for {groupby}.")


def copy_rna_obs_to_atac(atac, rna, source_col: str, target_col: str) -> bool:
    if not valid_group_column(rna, source_col):
        logging.info(
            "Skipping coverage group '%s': missing or not a useful grouping.",
            source_col,
        )
        return False
    atac.obs[target_col] = pd.Categorical(
        rna.obs.loc[atac.obs_names, source_col].astype(str).values
    )
    return True


def export_cluster_coverages(out_dir: str, atac, rna) -> None:
    logging.info("Creating coverage tracks...")

    for col in ["sample", "condition"]:
        if not copy_rna_obs_to_atac(atac, rna, col, col):
            continue
        safe_col = utils.safe_name(col)
        logging.info(f"Exporting coverage tracks for '{col}'.")
        export_coverage_group(
            atac,
            groupby=col,
            out_dir=os.path.join(out_dir, f"{safe_col}_coverages"),
            suffix=f"_{safe_col}.bw",
        )

    glue_dir = os.path.join(out_dir, "CoPro_cluster_coverages")
    export_coverage_group(
        atac,
        groupby="sg_clusters",
        out_dir=glue_dir,
        suffix="_cluster.bw",
    )

    reserved = {"sg_clusters", "sg_leiden", "sg_leiden_merged"}
    rna_cluster_cols = candidate_cluster_columns(rna, exclude=reserved)
    if rna_cluster_cols:
        rna_dir = os.path.join(out_dir, "RNA_cluster_coverages")
        for col in rna_cluster_cols:
            safe_col = utils.safe_name(col)
            groupby = f"rna_{safe_col}"
            copy_rna_obs_to_atac(atac, rna, col, groupby)
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
        atac_dir = os.path.join(out_dir, "ATAC_cluster_coverages")
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


def write_archr_cluster_metadata(rna, path: str) -> None:
    if "sg_clusters" not in rna.obs:
        raise RuntimeError(
            "Cannot export ArchR coverages because RNA AnnData is missing "
            "obs['sg_clusters']."
        )

    metadata = pd.DataFrame({
        "cell_id": rna.obs_names.astype(str),
        "barcode": [_extract_barcode(cell) for cell in rna.obs_names.astype(str)],
        "sg_clusters": rna.obs["sg_clusters"].astype(str).values,
    })
    for col in ["sample", "condition"]:
        if valid_group_column(rna, col):
            metadata[col] = rna.obs[col].astype(str).values

    reserved = {"sg_clusters", "sg_leiden", "sg_leiden_merged"}
    for col in candidate_cluster_columns(rna, exclude=reserved):
        metadata[f"rna_{utils.safe_name(col)}"] = rna.obs[col].astype(str).values

    missing_barcodes = metadata["barcode"].isna().sum()
    if missing_barcodes:
        logging.warning(
            "ArchR coverage metadata has %s cells without 16 bp barcodes; "
            "these cells can only match by exact cell ID.",
            missing_barcodes,
        )

    metadata.to_csv(path, index=False)


def export_archr_cluster_coverages(
    out_dir: str,
    archr_project_path: str,
    rna,
) -> None:
    logging.info("Creating ArchR coverage tracks...")

    os.makedirs(out_dir, exist_ok=True)
    cluster_csv = os.path.join(out_dir, "archr_sg_clusters.csv")
    write_archr_cluster_metadata(rna, cluster_csv)

    script_path = os.path.join(os.path.dirname(__file__), "archr_coverages.R")
    if not os.path.exists(script_path):
        raise RuntimeError(f"Missing ArchR coverage helper script: {script_path}")

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
            cluster_csv,
            out_dir,
        ],
        check=True,
    )

    logging.info("Finished ArchR coverage track export.")
