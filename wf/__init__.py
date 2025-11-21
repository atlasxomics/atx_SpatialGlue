""" ATX Workflow for running SpatialGlue of whole transcriptome + ATAC
"""
import logging
import sys

from latch.resources.workflow import workflow
from latch.types import LatchDir, LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter

from wf.task import glue_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info("This will go into output.log from __init__.py")

metadata = LatchMetadata(
    display_name="atx_glue",
    author=LatchAuthor(
        name="James McGann",
    ),
    parameters={
        "project_name": LatchParameter(
            display_name="Project Name",
            batch_table_column=True,
        ),
        "atac_anndata": LatchParameter(
            display_name="ATAC AnnData",
            batch_table_column=True,
        ),
        "wt_anndata": LatchParameter(
            display_name="Transcriptome AnnData",
            batch_table_column=True,
        ),
    },
    tags=[],
)


@workflow(metadata)
def glue_wf(
    project_name: str,
    atac_anndata: LatchFile,
    wt_anndata: LatchFile
) -> LatchDir:

    return glue_task(
        project_name=project_name,
        atac_anndata=atac_anndata,
        wt_anndata=wt_anndata
    )


if __name__ == "__main__":
    glue_task(
        project_name="D01887_develop",
        atac_anndata=LatchFile("latch://13502.account/snap_outs/Co_pro_D01887_ATAC/combined.h5ad"),
        wt_anndata=LatchFile("latch://13502.account/rnaSeqQC_output/D01887_NG05558_000729_star/optimize_outs/set1_cr1-0-nc30-nn15-md0-5-sp1-0/combined.h5ad")
    )
