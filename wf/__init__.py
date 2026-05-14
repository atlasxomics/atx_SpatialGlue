""" ATX Workflow for running SpatialGlue of whole transcriptome + ATAC
"""
import logging
import sys

from latch.resources.workflow import workflow
from latch.types import LatchDir, LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter

from wf.task import (
    DEFAULT_RESOLUTIONS,
    glue_train_task,
)

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
        "rna_prepared_h5ad": LatchParameter(
            display_name="Prepared RNA AnnData",
            description="Prepared RNA h5ad from the preprocessing step, \
                typically named 'rna_prepared.h5ad'.",
            batch_table_column=True,
        ),
        "ge_prepared_h5ad": LatchParameter(
            display_name="Prepared gene accessibility AnnData",
            description="Prepared gene accessibility h5ad from the \
                preprocessing step, typically named 'ge_prepared.h5ad'.",
            batch_table_column=True,
        ),
        "atac_tiles_prepared_h5ad": LatchParameter(
            display_name="Prepared ATAC tile AnnData",
            description="Prepared ATAC tile h5ad from the preprocessing step, \
                typically named 'atac_tiles_prepared.h5ad'.",
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
    },
    tags=[],
)


@workflow(metadata)
def glue_wf(
    project_name: str,
    rna_prepared_h5ad: LatchFile,
    ge_prepared_h5ad: LatchFile,
    atac_tiles_prepared_h5ad: LatchFile,
    n_neighbors: int = 15,
    min_cluster_size: int = 200,
    resolutions: str = DEFAULT_RESOLUTIONS,
    chosen_resolution: float = 0.0,
) -> LatchDir:

    return glue_train_task(
        project_name=project_name,
        rna_prepared_h5ad=rna_prepared_h5ad,
        ge_prepared_h5ad=ge_prepared_h5ad,
        atac_tiles_prepared_h5ad=atac_tiles_prepared_h5ad,
        n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size,
        resolutions=resolutions,
        chosen_resolution=chosen_resolution,
    )
