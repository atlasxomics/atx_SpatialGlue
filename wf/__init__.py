""" ATX Workflow for running SpatialGlue of whole transcriptome + ATAC
"""
import logging
import sys

from latch.resources.workflow import workflow
from latch.types import LatchDir, LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter

from wf.task import corr_task, glue_task

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
        "epigenomic_anndata": LatchParameter(
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
            display_name="Gene Accessiblity AnnData",
            description="H5AD file containing an AnnData object with gene \
                accesibility data as .X; in AtlasXomics Workflows, this is \
                typically named 'combined_ge.h5ad'.",
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
    ge_anndata: LatchFile
) -> LatchDir:

    results = glue_task(
        project_name=project_name,
        atac_anndata=atac_anndata,
        wt_anndata=wt_anndata
    )

    results = corr_task(
        project_name=project_name,
        results_dir=results,
        ge_anndata=ge_anndata
    )

    return results


if __name__ == "__main__":
    from latch.types import LatchDir
    corr_task(
        project_name="D01887_develop",
        results_dir=LatchDir("latch://13502.account/glue_outs/D01887_00000802"),
        ge_anndata=LatchFile("latch://13502.account/snap_outs/Co_pro_D01887_ATAC/combined_ge.h5ad")
    )
