import glob
import logging
import os
import pandas as pd
import pickle
import random
import subprocess
import sys
from typing import Optional

import numpy as np

from scipy import sparse

from latch.functions.messages import message
from latch.resources.tasks import custom_task
from latch.types import LatchDir, LatchFile

import wf.correlation as corr
from wf.coverage import export_archr_cluster_coverages, export_cluster_coverages
import wf.genestats as gs
from wf.markers import write_cluster_marker_outputs
from wf.peak2gene import run_archr_peak2gene, write_peak2gene_skip
import wf.plotting as pl
import wf.utils as utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


@custom_task(cpu=4, memory=576, storage_gib=1000)
def glue_preprocess_task(
    project_name: str,
    wt_anndata: LatchFile,
    ge_anndata: LatchFile,
    atac_anndata: Optional[LatchFile] = None,
) -> LatchDir:
    import scanpy as sc
    import torch

    # ------------------ Initialize ---------------------
    logging.info("Starting glue preprocessing task...")
    out_dir = f"/root/{project_name}_preprocess"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(utils.SEED)
    np.random.seed(utils.SEED)
    torch.manual_seed(utils.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(utils.SEED)

    logging.info("Reading WT AnnData...")
    rna = sc.read_h5ad(wt_anndata.local_path)

    atac = None
    if atac_anndata is not None:
        logging.info("Reading ATAC tile AnnData...")
        atac = sc.read_h5ad(atac_anndata.local_path)
    else:
        logging.info(
            "No ATAC tile AnnData provided; coverage export will require an "
            "ArchRProject."
        )

    logging.info("Reading gene accessibility AnnData...")
    ge = sc.read_h5ad(ge_anndata.local_path)

    atac_n_obs = atac.n_obs if atac is not None else "not provided"
    logging.info(f"n_obs RNA: {rna.n_obs} n_obs ATAC tiles: {atac_n_obs} n_obs GE: {ge.n_obs}")
    logging.info(f"\nRNA obs_names examples: {list(map(str, rna.obs_names[:5]))}")
    if atac is not None:
        logging.info(f"ATAC obs_names examples: {list(map(str, atac.obs_names[:5]))}")
    logging.info(f"GE obs_names examples: {list(map(str, ge.obs_names[:5]))}")

    rna.obs_names = utils.ensure_obs_run_barcodes(rna, "RNA")
    if atac is not None:
        atac.obs_names = utils.ensure_obs_run_barcodes(atac, "ATAC")
    ge.obs_names = utils.ensure_obs_run_barcodes(ge, "Gene accessibility")

    rna.var_names = utils.ensure_var_gene_symbols(rna, "RNA", min_fraction=0.5)
    ge.var_names = utils.ensure_var_gene_symbols(ge, "GE", min_fraction=0.5)

    rna.var_names_make_unique()
    ge.var_names_make_unique()

    rna_matched, ge_matched, atac_tiles_matched = utils.align_modalities(rna, ge, atac)
    if atac_tiles_matched is None:
        logging.info(f"Matched barcodes across RNA/GE: {rna_matched.n_obs} spots")
    else:
        logging.info(
            "Matched barcodes across RNA/GE/ATAC tiles: "
            f"{rna_matched.n_obs} spots"
        )

    # -------------------- GE cleanup + LSI --------------------
    logging.info("Cleaning gene accessibility AnnData...")
    if hasattr(ge_matched.X, "tocsr"):
        ge_matched.X = ge_matched.X.tocsr()

    ge_sums = np.array(ge_matched.X.sum(axis=0)).ravel()

    keep_var = ge_sums > 0
    if (~keep_var).sum():
        print(f"Dropping {(~keep_var).sum()} zero-signal GE features")
        ge_matched = ge_matched[:, keep_var].copy()

    logging.info("LSI on gene accessibility AnnData...")
    ge_matched.obsm["X_lsi"] = utils.compute_lsi(ge_matched.X)
    ge_matched.obsm["feat"] = ge_matched.obsm["X_lsi"].astype("float32")
    logging.info(f"GE SpatialGlue features: {ge_matched.obsm['feat'].shape}")

    # -------------------- RNA features with HVG guard ------------------
    logging.info("Adding feat to WT AnnData...")
    utils.add_rna_features(rna_matched)

    logging.info("Writing prepared SpatialGlue inputs...")
    rna_matched.write(f"{out_dir}/rna_prepared.h5ad")
    ge_matched.write(f"{out_dir}/ge_prepared.h5ad")
    if atac_tiles_matched is not None:
        atac_tiles_matched.write(f"{out_dir}/atac_tiles_prepared.h5ad")
    pd.DataFrame([{
        "has_atac_tiles": bool(atac_tiles_matched is not None),
    }]).to_csv(f"{out_dir}/prepared_manifest.csv", index=False)

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}/preprocess")


@custom_task(cpu=64, memory=576, storage_gib=1000)
def glue_train_task(
    project_name: str,
    prepared_dir: LatchDir,
    n_neighbors: int = 15,
    min_cluster_size: int = 200,
    resolutions: str = utils.DEFAULT_RESOLUTIONS,
    chosen_resolution: float = 0.0,
    spatialglue_model_pickle: Optional[LatchFile] = None,
) -> LatchDir:
    import scanpy as sc
    import torch
    from SpatialGlue.preprocess import construct_neighbor_graph
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

    # ------------------ Initialize ---------------------
    logging.info("Starting glue training task...")
    out_dir = f"/root/{project_name}_coverages"
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = utils.figures_dir(out_dir)

    random.seed(utils.SEED)
    np.random.seed(utils.SEED)
    torch.manual_seed(utils.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(utils.SEED)

    logging.info("Reading prepared SpatialGlue inputs...")
    rna_prepared_path = LatchFile(
        f"{prepared_dir.remote_path}/rna_prepared.h5ad"
    ).local_path
    ge_prepared_path = LatchFile(
        f"{prepared_dir.remote_path}/ge_prepared.h5ad"
    ).local_path
    prepared_manifest_path = LatchFile(
        f"{prepared_dir.remote_path}/prepared_manifest.csv"
    ).local_path
    prepared_manifest = pd.read_csv(prepared_manifest_path)
    has_atac_tiles = utils.as_bool(prepared_manifest.loc[0, "has_atac_tiles"])
    if has_atac_tiles:
        atac_tiles_prepared_path = LatchFile(
            f"{prepared_dir.remote_path}/atac_tiles_prepared.h5ad"
        ).local_path
    else:
        atac_tiles_prepared_path = None

    rna_matched = sc.read_h5ad(rna_prepared_path)
    ge_matched = sc.read_h5ad(ge_prepared_path)
    atac_tiles_matched = (
        sc.read_h5ad(atac_tiles_prepared_path)
        if atac_tiles_prepared_path is not None
        else None
    )
    atac_shape = atac_tiles_matched.shape if atac_tiles_matched is not None else "not provided"
    logging.info(
        f"Prepared inputs loaded: RNA {rna_matched.shape}, GE {ge_matched.shape}, "
        f"ATAC tiles {atac_shape}"
    )

    # -------------------- Train or load SpatialGlue --------------------
    if spatialglue_model_pickle is not None:
        logging.info(
            "Loading existing SpatialGlue_model.pickle; skipping SpatialGlue "
            "neighbor-graph construction and training."
        )
        with open(spatialglue_model_pickle.local_path, "rb") as f:
            out = pickle.load(f)
        if not isinstance(out, dict) or "SpatialGlue" not in out:
            raise ValueError(
                "spatialglue_model_pickle must contain a dict with a "
                "'SpatialGlue' embedding, as written by this workflow."
            )
        rna_result = rna_matched.copy()
        ge_result = ge_matched.copy()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logging.info("construct_neighbor_graph...")
        data = construct_neighbor_graph(
            rna_matched, ge_matched, datatype="Spatial-epigenome-transcriptome"
        )

        logging.info("Train_SpatialGlue...")
        model = Train_SpatialGlue(
            data,
            datatype="Spatial-epigenome-transcriptome",
            random_seed=utils.SEED,
            device=device,
        )
        logging.info("training...")
        out = model.train()

        rna_result = data["adata_omics1"]
        ge_result = data["adata_omics2"]

    out["SpatialGlue"] = utils.to_numpy_array(out["SpatialGlue"])
    for key in ["alpha", "alpha_omics1", "alpha_omics2"]:
        if key in out and out.get(key) is not None:
            out[key] = utils.to_numpy_array(out[key])
    if out["SpatialGlue"].shape[0] != rna_result.n_obs:
        raise ValueError(
            "Loaded SpatialGlue embedding has "
            f"{out['SpatialGlue'].shape[0]} rows, but prepared RNA has "
            f"{rna_result.n_obs} observations. Use a pickle generated from "
            "the same prepared inputs."
        )

    # Collect embeddings/weights on an AnnData for downstream analysis
    logging.info("copy data...")
    adata = rna_result.copy()
    adata.obsm["SpatialGlue"] = out["SpatialGlue"]
    adata.obsm["alpha"] = out.get("alpha")
    adata.obsm["alpha_omics1"] = out.get("alpha_omics1")
    adata.obsm["alpha_omics2"] = out.get("alpha_omics2")

    # -------------------- Neighbors/UMAP/Leiden --------------------
    logging.info("clustering on spatialglue dims...")
    sc.pp.neighbors(
        adata, use_rep="SpatialGlue", n_neighbors=n_neighbors, random_state=utils.SEED
    )
    sc.tl.umap(adata, random_state=utils.SEED)

    sweep_rows = []
    final_raw_key = None
    final_merged_key = None
    best_moran = None
    best_raw_key = None
    best_merged_key = None
    best_resolution = None
    connectivities = utils.spatial_connectivities(adata, n_neighbors=n_neighbors)
    for resolution in utils.parse_resolutions(resolutions):
        key = f"sg_leiden_{utils.resolution_suffix(resolution)}"
        merged_key = f"{key}_merged"
        sc.tl.leiden(
            adata, resolution=resolution, random_state=utils.SEED, key_added=key
        )
        adata.obs[key] = adata.obs[key].astype("category")

        utils.merge_small_clusters(
            adata,
            cluster_key=key,
            embed_key="SpatialGlue",
            min_cells=min_cluster_size,
            new_key=merged_key,
            verbose=True,
        )

        cluster_morans_i = utils.morans_i(
            connectivities,
            adata.obs[merged_key].cat.codes.values,
        )
        sweep_rows.append({
            "resolution": resolution,
            "raw_key": key,
            "merged_key": merged_key,
            "n_clusters_raw": int(adata.obs[key].nunique()),
            "n_clusters_merged": int(adata.obs[merged_key].nunique()),
            "morans_i": cluster_morans_i,
            "min_cluster_size": int(min_cluster_size),
            "n_neighbors": int(n_neighbors),
        })
        logging.info(
            f"Resolution {resolution:g}: {adata.obs[key].nunique()} raw "
            f"clusters, {adata.obs[merged_key].nunique()} merged clusters, "
            f"Moran's I={cluster_morans_i:.4f}"
        )

        if best_moran is None or cluster_morans_i > best_moran:
            best_moran = cluster_morans_i
            best_raw_key = key
            best_merged_key = merged_key
            best_resolution = resolution

        if chosen_resolution != 0 and np.isclose(resolution, chosen_resolution):
            final_raw_key = key
            final_merged_key = merged_key

    if chosen_resolution == 0:
        final_raw_key = best_raw_key
        final_merged_key = best_merged_key
        chosen_resolution = best_resolution
        logging.info(
            f"Auto-selected resolution {chosen_resolution:g} with "
            f"Moran's I={best_moran:.4f}"
        )

    if final_raw_key is None or final_merged_key is None:
        requested_resolution = chosen_resolution
        final_raw_key = best_raw_key
        final_merged_key = best_merged_key
        chosen_resolution = best_resolution
        warning = (
            f"chosen_resolution={requested_resolution:g} was not found in "
            f"resolutions={resolutions}. Falling back to resolution "
            f"{chosen_resolution:g}, which had the best Moran's I="
            f"{best_moran:.4f}."
        )
        logging.warning(warning)
        message(typ="warning", data={"title": "Resolution override not found", "body": warning})

    adata.obs["sg_clusters"] = adata.obs[final_merged_key].astype("category")

    sweep_path = utils.table_path(out_dir, "spatialglue_cluster_sweep.csv")
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    logging.info(f"Saved SpatialGlue cluster sweep: {sweep_path}")

    # -------------------- Plots --------------------
    logging.info("Plotting figures...")
    sc.settings.figdir = figures_dir
    cluster_result_keys = []
    for row in sweep_rows:
        cluster_result_keys.extend([row["raw_key"], row["merged_key"]])
    cluster_result_keys.append("sg_clusters")
    pl.write_cluster_resolution_plots(out_dir, adata, cluster_result_keys)
    sc.pl.umap(
        adata,
        color=["sg_clusters"],
        size=utils.SCANPY_CLUSTER_POINT_SIZE,
        save=".png",
    )

    # -------------------- Save data --------------------
    # Copy new clustering results
    result_objects = [rna_result, ge_result]
    if atac_tiles_matched is not None:
        result_objects.append(atac_tiles_matched)
    for obj in result_objects:
        for old_key in ["sg_leiden", "sg_leiden_merged", "sg_merged_leiden"]:
            if old_key in obj.obs.columns:
                del obj.obs[old_key]
        for key in cluster_result_keys:
            if key in adata.obs.columns:
                obj.obs[key] = adata.obs.loc[obj.obs_names, key].astype(str).values
                obj.obs[key] = obj.obs[key].astype("category")

    for obj in [rna_result, ge_result]:
        obj.obsm["SpatialGlue"] = adata.obsm["SpatialGlue"]
        if "X_umap" in adata.obsm:
            obj.obsm["X_umap"] = adata.obsm["X_umap"]
        obj.uns["spatialglue_umap_params"] = {
            "source_representation": "SpatialGlue",
            "n_neighbors": int(n_neighbors),
            "random_state": int(utils.SEED),
        }
    for key in ["alpha", "alpha_omics1", "alpha_omics2"]:
        if key in adata.obsm and adata.obsm[key] is not None:
            rna_result.obsm[key] = adata.obsm[key]

    marker_jobs = [
        (rna_result, "RNA", "rna_"),
        (ge_result, "GE", "ge_"),
    ]
    for marker_adata, modality_name, output_prefix in marker_jobs:
        try:
            write_cluster_marker_outputs(
                marker_adata,
                out_dir,
                cluster_key="sg_clusters",
                marker_top_n=50,
                modality_name=modality_name,
                output_prefix=output_prefix,
            )
        except Exception as e:
            warning = (
                f"Skipping {modality_name} cluster marker genes after error: {e}"
            )
            logging.exception(warning)
            message(typ="warning", data={"title": warning, "body": warning})

    logging.info("Writing data...")
    for obj in result_objects:
        utils.strip_plotting_embeddings(obj)

    if atac_tiles_matched is not None:
        atac_tiles_matched.write(f"{out_dir}/atac_glue.h5ad")
    ge_plotting = utils.make_plotting_anndata(
        ge_result,
        matrix_dtype=np.float16,
        force_dense=True,
        obs_rename={
            "sg_clusters": "CoPro clusters",
            "cluster": "ATAC_cluster",
        },
    )
    rna_plotting = utils.make_plotting_anndata(
        rna_result,
        matrix_dtype=np.float16,
        categorical_obs_keep={
            "sample",
            "condition",
            "cluster",
            "sg_cluster",
            "sg_clusters",
        },
        obs_drop={"on_off", "row", "col", "xcor", "ycor"},
        obs_rename={
            "sg_clusters": "CoPro clusters",
            "cluster": "WT_cluster",
        },
    )
    if "ATAC_cluster" in ge_plotting.obs.columns:
        rna_plotting.obs["ATAC_cluster"] = ge_plotting.obs.loc[
            rna_plotting.obs_names, "ATAC_cluster"
        ].values
    if "WT_cluster" in rna_plotting.obs.columns:
        ge_plotting.obs["WT_cluster"] = rna_plotting.obs.loc[
            ge_plotting.obs_names, "WT_cluster"
        ].values
    utils.order_plotting_obs_columns(ge_plotting)
    utils.order_plotting_obs_columns(rna_plotting)

    ge_plotting.write(f"{out_dir}/ge_glue_sm.h5ad")
    rna_plotting.write(f"{out_dir}/rna_glue_sm.h5ad")
    ge_result.write(f"{out_dir}/ge_glue.h5ad")
    rna_result.write(f"{out_dir}/rna_glue.h5ad")
    pd.DataFrame([{
        "has_atac_tiles": bool(atac_tiles_matched is not None),
        "rna_full_h5ad": "rna_glue.h5ad",
        "rna_plotting_h5ad": "rna_glue_sm.h5ad",
        "ge_full_h5ad": "ge_glue.h5ad",
        "ge_plotting_h5ad": "ge_glue_sm.h5ad",
    }]).to_csv(f"{out_dir}/coverage_manifest.csv", index=False)

    with open(f"{out_dir}/SpatialGlue_model.pickle", "wb") as f:
        pickle.dump(out, f)

    legacy_figures = glob.glob("/root/figures/*")
    if legacy_figures:
        subprocess.run(["mv"] + legacy_figures + [figures_dir], check=False)

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")


@custom_task(cpu=8, memory=384, storage_gib=1000)
def coverage_task(
    project_name: str,
    results_dir: LatchDir,
    archr_project: Optional[LatchDir] = None,
) -> LatchDir:
    import scanpy as sc

    out_dir = f"/root/{project_name}_coverages"
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = LatchFile(f"{results_dir.remote_path}/coverage_manifest.csv").local_path
    manifest = pd.read_csv(manifest_path)
    has_atac_tiles = utils.as_bool(manifest.loc[0, "has_atac_tiles"])
    if not has_atac_tiles and archr_project is None:
        msg = (
            "No ATAC tile matrix or ArchRProject was provided, so coverage "
            "track generation was skipped."
        )
        logging.warning(msg)
        message(typ="warning", data={"title": "Coverage skipped", "body": msg})
        with open(f"{out_dir}/coverage_skipped.txt", "w") as f:
            f.write(f"{msg}\n")
        return LatchDir(out_dir, f"{results_dir.remote_path}/coverages")

    rna_path = LatchFile(f"{results_dir.remote_path}/rna_glue.h5ad").local_path
    logging.info("Reading clustered RNA AnnData for coverage export...")
    rna = sc.read_h5ad(rna_path)

    if has_atac_tiles:
        if archr_project is not None:
            msg = (
                "Both ATAC tile AnnData and ArchRProject were provided. Using "
                "the ATAC tile AnnData coverage path."
            )
            logging.warning(msg)
            message(typ="warning", data={"title": "Coverage source", "body": msg})

        logging.info("Downloading clustered ATAC AnnData for coverage export...")
        atac_path = LatchFile(f"{results_dir.remote_path}/atac_glue.h5ad").local_path
        logging.info("Reading clustered ATAC AnnData object...")
        atac = sc.read_h5ad(atac_path)
        export_cluster_coverages(out_dir, atac, rna)
    else:
        logging.info("Using ArchRProject for coverage export...")
        export_archr_cluster_coverages(out_dir, archr_project.local_path, rna)

    return LatchDir(out_dir, f"{results_dir.remote_path}/coverages")


@custom_task(cpu=1, memory=4, storage_gib=10)
def finalize_task(
    results_dir: LatchDir,
    coverage_dir: LatchDir,
    peak2gene_dir: LatchDir,
    preprocess_dir: LatchDir,
) -> LatchDir:
    logging.info(f"Coverage outputs written to: {coverage_dir.remote_path}")
    logging.info(f"Peak2Gene outputs written to: {peak2gene_dir.remote_path}")
    logging.info(f"Deleting preprocess checkpoint: {preprocess_dir.remote_path}")
    try:
        subprocess.run(
            ["latch", "rmr", "-y", "--no-glob", preprocess_dir.remote_path],
            check=True,
        )
    except Exception as e:
        warning = (
            "Unable to delete preprocess checkpoint after workflow completion: "
            f"{e}"
        )
        logging.exception(warning)
        message(typ="warning", data={"title": "Preprocess cleanup failed", "body": warning})
    return results_dir


@custom_task(cpu=8, memory=384, storage_gib=1000)
def peak2gene_task(
    project_name: str,
    results_dir: LatchDir,
    peak2gene_archr_project: Optional[LatchDir] = None,
    genes_of_interest: Optional[str] = None,
) -> LatchDir:
    import scanpy as sc

    out_dir = f"/root/{project_name}_peak2gene"
    os.makedirs(out_dir, exist_ok=True)
    remote_path = f"{results_dir.remote_path}/peak2gene"

    if peak2gene_archr_project is None:
        msg = (
            "No Peak2Gene ArchRProject was provided, so Peak2Gene link "
            "generation was skipped."
        )
        logging.info(msg)
        write_peak2gene_skip(out_dir, msg)
        return LatchDir(out_dir, remote_path)

    try:
        rna_path = LatchFile(f"{results_dir.remote_path}/rna_glue.h5ad").local_path
        logging.info("Reading clustered RNA AnnData for Peak2Gene export...")
        rna = sc.read_h5ad(rna_path)
        run_archr_peak2gene(
            out_dir=out_dir,
            archr_project_path=peak2gene_archr_project.local_path,
            rna=rna,
            genes_of_interest=genes_of_interest,
        )
    except Exception as e:
        msg = (
            "Peak2Gene link generation failed and was skipped. "
            f"Reason: {e}"
        )
        logging.exception(msg)
        message(typ="warning", data={"title": "Peak2Gene skipped", "body": msg})
        write_peak2gene_skip(out_dir, msg)

    return LatchDir(out_dir, remote_path)


@custom_task(cpu=8, memory=576, storage_gib=1000)
def corr_task(
    project_name: str,
    results_dir: LatchDir,
    ge_anndata: LatchFile,
    min_frac_expressing: float = 0.05,
    genes_of_interest: Optional[str] = None,
) -> LatchDir:
    import scanpy as sc

    out_dir = f"/root/{project_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Download and read data -----------------------------------------------
    logging.info("Downloading RNA data...")
    # Add check if exists on latch
    rna_path = LatchFile(f"{results_dir.remote_path}/rna_glue.h5ad").local_path

    logging.info("Downloading Gene Accessibility data...")
    ge_path = ge_anndata.local_path

    logging.info("Reading RNA data...")
    rna = sc.read_h5ad(rna_path)
    logging.info("Reading Gene Accessibility data...")
    ge = sc.read_h5ad(ge_path)

    logging.info("Preparing data for correlation...")
    rna.obs_names = utils.ensure_obs_run_barcodes(rna, "RNA")
    ge.obs_names = utils.ensure_obs_run_barcodes(ge, "Gene accessibility")

    # Make sure at least half are gene symbols
    rna.var_names = utils.ensure_var_gene_symbols(rna, "RNA", min_fraction=0.5)
    ge.var_names = utils.ensure_var_gene_symbols(ge, "GE", min_fraction=0.5)

    # Get Spearman correlation table -----------------------------------------
    rna.var_names_make_unique()
    ge.var_names_make_unique()

    # Reduce both to common genes/cells and align
    rna_sub, ge_sub = corr.synch_adata(rna, ge)

    genes = rna_sub.var_names

    logging.info("Ensuring dense matrix...")
    # Prefer monotonic transform (normalized/log1p/counts layers)
    preferred_layers = ["normalized", "log1p", "counts"]
    X_rna = None
    for layer in preferred_layers:
        if layer in rna_sub.layers:
            logging.info(f"Using RNA layer '{layer}' for correlation")
            message(
                typ="info",
                data={
                    "title": "RNA Layers", "body":
                    f"Using RNA layer '{layer}' for correlation"
                }
            )
            X_rna = utils.to_dense(rna_sub.layers[layer]).astype(np.float32)
            break
    if X_rna is None:  # Fall back to .X with warning
        logging.warning(
            "RNA layers normalized/log1p/counts not found; using rna_sub.X"
        )
        message(
            typ="warning",
            data={
                "title": "RNA Layers",
                "body": """Using RNA .X for correlation analysis.  If
                Transcriptome AnnData is from an RNAQC or optimize_wt Workflow,
                this represents scaled data which is suboptimal for
                correlations."""
            }
        )
        X_rna = utils.to_dense(rna_sub.X).astype(np.float32)
    X_ge = utils.to_dense(ge_sub.X).astype(np.float32)

    # Gene stats ------------------------------------------------------------
    X_rna_counts, rna_source = gs.get_rna_counts_matrix(rna_sub)
    X_rna_counts_dense = utils.to_dense(X_rna_counts).astype(np.float32)
    mean_umi = X_rna_counts_dense.mean(axis=0)
    frac_expressing = (X_rna_counts_dense > 0).mean(axis=0)
    keep = frac_expressing >= float(min_frac_expressing)
    n_keep = int(keep.sum())
    rna_stats = gs.compute_gene_stats_matrix(
        X_rna_counts, genes, prefix="rna_umi", include_minmax_nonzero=True
    )
    filter_stats = pd.DataFrame({
        "gene": genes.astype(str).values,
        "corr_mean_umi": mean_umi,
        "corr_frac_expressing": frac_expressing,
        "passes_corr_filter": keep,
        "corr_min_frac_expressing": float(min_frac_expressing),
    })

    ge_stats = gs.compute_gene_stats_matrix(
        sparse.csr_matrix(X_ge),
        genes,
        prefix="ge_norm",
        include_minmax_nonzero=False
    )

    # Merge all stats before correlation so the workflow still produces a
    # useful QC table if all genes fail the correlation filters.
    stats = (
        rna_stats.merge(ge_stats, on="gene", how="inner")
        .merge(filter_stats, on="gene", how="inner")
    )

    res_path = utils.table_path(out_dir, "atac-ge_vs_rna_spearman.csv")
    if n_keep == 0:
        msg = (
            "No genes passed correlation filters: "
            f"fraction expressing >= {min_frac_expressing}. "
            "Skipping correlation table generation and correlation plots."
        )
        logging.warning(msg)
        message(
            typ="warning",
            data={
                "title": "No correlation genes",
                "body": msg,
            },
        )
        pd.DataFrame(columns=[
            "gene",
            "spearman_rho",
            "pval",
            "qval_bh",
            "mean_RNA",
            "mean_GA",
            "abs_rho",
        ]).to_csv(res_path, index=False)
        pd.DataFrame(columns=[
            "gene",
            "spearman_r",
            "mean_umi",
            "frac_expressing",
            "pval",
            "qval_bh",
        ]).to_csv(
            utils.table_path(out_dir, "atac_rna_spearman_all_genes.csv"),
            index=False,
        )
        logging.info(f"Saved empty Spearman results: {res_path}")

        stats_path = utils.table_path(out_dir, "gene_stats.csv")
        stats.to_csv(stats_path, index=False)
        logging.info(f"Saved: {stats_path} (RNA counts source: {rna_source})")
        report_genes = pl.selected_report_genes(genes_of_interest, genes)
        pl.write_spatial_expression_reports(
            out_dir=out_dir,
            rna=rna_sub,
            ge=ge_sub,
            genes=genes,
            X_rna=X_rna,
            X_ge=X_ge,
            report_genes=report_genes,
        )
        pl.write_spatial_cluster_reports(out_dir, rna_sub, ge_sub)
        pl.write_umi_reports(
            out_dir=out_dir,
            rna=rna_sub,
            genes=genes,
            X_counts=X_rna_counts_dense,
            X_expr=X_rna,
            report_genes=report_genes,
        )
        return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")

    logging.info(
        f"Correlation filters retained {n_keep}/{len(genes)} genes "
        f"(fraction expressing >= {min_frac_expressing})."
    )

    logging.info("Computing correlations...")
    res = corr.get_corr_df(X_rna[:, keep], X_ge[:, keep], genes[keep])
    res.to_csv(res_path, index=False)
    logging.info(f"Saved Spearman results: {res_path}")

    stats = stats.merge(res, on="gene", how="left")
    corr_with_filter = res.merge(filter_stats, on="gene", how="left")
    notebook_corr = corr_with_filter.rename(columns={
        "spearman_rho": "spearman_r",
        "corr_mean_umi": "mean_umi",
        "corr_frac_expressing": "frac_expressing",
    })
    notebook_corr_path = utils.table_path(out_dir, "atac_rna_spearman_all_genes.csv")
    notebook_corr[[
        "gene",
        "spearman_r",
        "mean_umi",
        "frac_expressing",
        "pval",
        "qval_bh",
        "abs_rho",
    ]].to_csv(notebook_corr_path, index=False)
    logging.info(f"Saved notebook-style Spearman table: {notebook_corr_path}")

    stats_path = utils.table_path(out_dir, "gene_stats.csv")
    stats.to_csv(stats_path, index=False)
    logging.info(f"Saved: {stats_path} (RNA counts source: {rna_source})")

    overview_path = utils.fig_path(out_dir, "atac_rna_correlation_overview.png")
    logging.info(f"Saving correlation overview to {overview_path}")
    pl.plot_correlation_overview(corr_with_filter, overview_path)

    report_genes = pl.selected_report_genes(genes_of_interest, genes, res)
    pl.write_spatial_expression_reports(
        out_dir=out_dir,
        rna=rna_sub,
        ge=ge_sub,
        genes=genes,
        X_rna=X_rna,
        X_ge=X_ge,
        report_genes=report_genes,
    )
    pl.write_spatial_cluster_reports(out_dir, rna_sub, ge_sub)
    pl.write_umi_reports(
        out_dir=out_dir,
        rna=rna_sub,
        genes=genes,
        X_counts=X_rna_counts_dense,
        X_expr=X_rna,
        report_genes=report_genes,
    )

    pl.write_cluster_correlation_outputs(
        out_dir=out_dir,
        rna=rna_sub,
        genes=genes,
        X_rna=X_rna,
        X_ge=X_ge,
        corr_df=res,
        genes_of_interest=genes_of_interest,
    )

    bar_path = utils.fig_path(out_dir, "top_genes_bar.png")
    logging.info(f"Saving top correlated genes figure to {bar_path}")
    pl.plot_top_genes_bar(res, n=20, fdr_thresh=0.05, outpath=bar_path)

    volcano_path = utils.fig_path(out_dir, "corr_volcano.png")
    logging.info(f"Saving volcano to {volcano_path}")
    pl.plot_corr_volcano_broken(
        res,
        outpath=volcano_path,
        q_thresh=0.05,
        rho_thresh=0.1,
        y_low=(0, 300),       # bottom panel exactly 0–300
        y_high=(300, 311),   # top panel starts at 308; upper bound auto
        jitter_y=0.3,         # keep points from crossing the break
        top_pos_labels=10,
        top_neg_labels=10,
        title="Correlation volcano"
    )

    return LatchDir(out_dir, f"latch:///glue_outs/{project_name}")
