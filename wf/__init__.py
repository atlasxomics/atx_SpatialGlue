""" ATX Workflow for running SpatialGlue of whole transcriptome + ATAC
"""
import logging
import sys
from typing import Optional

from latch.resources.workflow import workflow
from latch.types import LatchDir, LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter

from wf.task import DEFAULT_RESOLUTIONS, corr_task, glue_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

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
            display_name="Epigenomic AnnData",
            description="H5AD file containing an AnnData object with a tile \
                matrix as .X; in AtlasXomics Workflows, this is typically \
                named 'combined.h5ad'.",
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
            display_name="Chosen Leiden resolution",
            description="Optional resolution override from the sweep. Set to \
                0 to automatically use the resolution with the best Moran's I \
                score.",
            batch_table_column=True,
        ),
        "min_umi_threshold": LatchParameter(
            display_name="Correlation min mean UMI",
            description="Minimum mean raw RNA UMI per gene required before \
                computing RNA vs gene accessibility correlation.",
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
)


@workflow(metadata)
def glue_wf(
    project_name: str,
    atac_anndata: LatchFile,
    wt_anndata: LatchFile,
    ge_anndata: LatchFile,
    n_neighbors: int = 15,
    min_cluster_size: int = 200,
    resolutions: str = DEFAULT_RESOLUTIONS,
    chosen_resolution: float = 0.0,
    min_umi_threshold: float = 0.5,
    min_frac_expressing: float = 0.05,
    genes_of_interest: Optional[str] = None,
) -> LatchDir:

    results = glue_task(
        project_name=project_name,
        atac_anndata=atac_anndata,
        wt_anndata=wt_anndata,
        ge_anndata=ge_anndata,
        n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size,
        resolutions=resolutions,
        chosen_resolution=chosen_resolution,
    )

    results = corr_task(
        project_name=project_name,
        results_dir=results,
        ge_anndata=ge_anndata,
        min_umi_threshold=min_umi_threshold,
        min_frac_expressing=min_frac_expressing,
        genes_of_interest=genes_of_interest,
    )

    return results


if __name__ == "__main__":
    from latch.types import LatchDir
    corr_task(
        project_name="D01887_develop",
        results_dir=LatchDir("latch://13502.account/glue_outs/D01887_00000802"),
        ge_anndata=LatchFile("latch://13502.account/snap_outs/Co_pro_D01887_ATAC/combined_ge.h5ad")
    )
