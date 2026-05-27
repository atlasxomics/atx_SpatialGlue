""" ATX Workflow for running SpatialGlue of whole transcriptome + ATAC
"""
import logging
import sys
from typing import Optional

from latch.resources.workflow import workflow
from latch.types import LatchDir, LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter
from latch.types.metadata import (
    Fork, ForkBranch, Params, Section, Spoiler, Text,
)

from wf.utils import DEFAULT_RESOLUTIONS
from wf.task import (
    coverage_task,
    corr_task,
    finalize_task,
    glue_preprocess_task,
    glue_train_task,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

flow = [
    Section(
        "Input Data",
        Params("project_name"),
        Params("wt_anndata"),
        Params("ge_anndata"),
        Spoiler(
            "ATAC Data",
            Text(
                "Optionally, add data for epigenomic coverage tracks. Input can"
                "be either an AnnData h5ad file containing a tile matrix or an"
                "ArchRProject directory."
            ),
            Fork(
                "epigenomic_input_type",
                "Choose epigenomic input type",
                atac_anndata=ForkBranch(
                    "atac_anndata", Params("atac_anndata")
                ),
                archr_project=ForkBranch(
                    "archr_project", Params("archr_project")
                ),
            ),
        ),
    ),
    Section(
        "Clustering",
        Text(
            "Parameters for SpatialGlue clustering"
        ),
        Params("n_neighbors"),
        Params("min_cluster_size"),
        Params("resolutions"),
        Params("chosen_resolution"),
    ),
    Section(
        "Correlation Statistics",
        Params("min_frac_expressing"),
        Params("genes_of_interest"),
    ),
    Spoiler(
        "Advanced Options",
        Params("spatialglue_model_pickle")
    )
]

metadata = LatchMetadata(
    display_name="atx_glue",
    author=LatchAuthor(
        name="James McGann",
    ),
    parameters={
        "project_name": LatchParameter(
            display_name="Project Name",
            description="Name of output directory in glue_outs/",
            batch_table_column=True,
        ),
        "atac_anndata": LatchParameter(
            display_name="Epigenomic tile AnnData",
            description="Optional H5AD file containing an AnnData object with \
                an ATAC tile matrix as .X. If omitted, SpatialGlue runs with \
                RNA and gene accessibility inputs and coverage tracks are \
                exported from an ArchRProject if one is supplied.",
            batch_table_column=True,
        ),
        "archr_project": LatchParameter(
            display_name="ArchRProject",
            description="Optional ArchRProject directory used to export \
                coverage tracks when an epigenomic tile AnnData is not \
                supplied. If both are supplied, the tile AnnData coverage path \
                is used.",
            batch_table_column=True,
        ),
        "wt_anndata": LatchParameter(
            display_name="Transcriptome AnnData",
            description="H5AD file containing an AnnData object with gene \
                expression data as .X.",
            batch_table_column=True,
        ),
        "ge_anndata": LatchParameter(
            display_name="Gene Accessibility AnnData",
            description="H5AD file containing an AnnData object with gene \
                accessibility data as .X; in AtlasXomics Workflows, this is \
                typically named 'combined_ge.h5ad'.",
            batch_table_column=True,
        ),
        "spatialglue_model_pickle": LatchParameter(
            display_name="Existing SpatialGlue model pickle",
            description="Optional SpatialGlue_model.pickle from a previous \
                run. If provided, the workflow skips SpatialGlue training and \
                reuses the saved embedding/attention output.",
            batch_table_column=True,
        ),
        "n_neighbors": LatchParameter(
            display_name="SpatialGlue clustering neighbors",
            description="Number of neighbors used when clustering the \
                SpatialGlue embedding.",
            batch_table_column=True,
        ),
        "min_cluster_size": LatchParameter(
            display_name="Minimum cluster size",
            description="Clusters smaller than this are merged into the \
                nearest larger cluster in SpatialGlue embedding space.",
            batch_table_column=True,
        ),
        "resolutions": LatchParameter(
            display_name="Leiden resolution sweep",
            description="Comma-separated Leiden resolutions to run on the \
                SpatialGlue embedding.",
            batch_table_column=True,
        ),
        "chosen_resolution": LatchParameter(
            display_name="Chosen Leiden resolution (override)",
            description="Optional resolution override from the sweep. Set to \
                0 to automatically use the resolution with the best Moran's I \
                score.",
            batch_table_column=True,
        ),
        "min_frac_expressing": LatchParameter(
            display_name="Correlation min fraction expressing",
            description="Minimum fraction of spots with nonzero raw RNA UMI \
                for a gene before computing RNA vs gene accessibility \
                correlation.",
            batch_table_column=True,
        ),
        "genes_of_interest": LatchParameter(
            display_name="Genes of interest",
            description="Optional comma-separated gene symbols used for \
                per-cluster RNA/ATAC GE summaries. If empty, the workflow uses \
                the top correlated genes.",
            batch_table_column=True,
        ),
    },
    tags=[],
    flow=flow
)


@workflow(metadata)
def glue_wf(
    project_name: str,
    wt_anndata: LatchFile,
    ge_anndata: LatchFile,
    atac_anndata: Optional[LatchFile] = None,
    archr_project: Optional[LatchDir] = None,
    spatialglue_model_pickle: Optional[LatchFile] = None,
    n_neighbors: int = 15,
    min_cluster_size: int = 200,
    resolutions: str = DEFAULT_RESOLUTIONS,
    chosen_resolution: float = 0.0,
    min_frac_expressing: float = 0.05,
    genes_of_interest: Optional[str] = None,
) -> LatchDir:

    prepared = glue_preprocess_task(
        project_name=project_name,
        wt_anndata=wt_anndata,
        ge_anndata=ge_anndata,
        atac_anndata=atac_anndata,
    )

    results = glue_train_task(
        project_name=project_name,
        prepared_dir=prepared,
        n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size,
        resolutions=resolutions,
        chosen_resolution=chosen_resolution,
        spatialglue_model_pickle=spatialglue_model_pickle,
    )

    coverage_results = coverage_task(
        project_name=project_name,
        results_dir=results,
        archr_project=archr_project,
    )

    corr_results = corr_task(
        project_name=project_name,
        results_dir=results,
        ge_anndata=ge_anndata,
        min_frac_expressing=min_frac_expressing,
        genes_of_interest=genes_of_interest,
    )

    return finalize_task(
        results_dir=corr_results,
        coverage_dir=coverage_results,
    )


if __name__ == "__main__":
    from latch.types import LatchDir
    corr_task(
        project_name="D01887_develop",
        results_dir=LatchDir("latch://13502.account/glue_outs/D01887_00000802"),
        ge_anndata=LatchFile("latch://13502.account/snap_outs/Co_pro_D01887_ATAC/combined_ge.h5ad")
    )
