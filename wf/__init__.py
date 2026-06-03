"""Peak2Gene-only entry point for an existing atx_glue output directory."""
import logging
import sys
from typing import Optional

from latch.resources.workflow import workflow
from latch.types import LatchDir
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter
from latch.types.metadata import Params, Section

from wf.task import peak2gene_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

flow = [
    Section(
        "Input Data",
        Params("existing_output_dir"),
        Params("peak2gene_archr_project"),
        Params("genes_of_interest"),
    ),
]

metadata = LatchMetadata(
    display_name="atx_glue_peak2gene_only",
    author=LatchAuthor(
        name="James McGann",
    ),
    parameters={
        "existing_output_dir": LatchParameter(
            display_name="Existing SpatialGlue Output Directory",
            description="Existing atx_glue output directory containing \
                rna_glue.h5ad. Peak2Gene outputs will be written into a \
                peak2gene/ subdirectory under this output directory.",
            batch_table_column=True,
        ),
        "peak2gene_archr_project": LatchParameter(
            display_name="Peak2Gene ArchRProject",
            description="ArchRProject directory containing peaks used to \
                compute ArchR Peak2Gene links. RNA expression is taken from \
                rna_glue.h5ad in the existing output directory.",
            batch_table_column=True,
        ),
        "genes_of_interest": LatchParameter(
            display_name="Genes of interest",
            description="Optional comma-separated gene symbols used to write \
                per-gene Peak2Gene link CSV and BEDPE outputs.",
            batch_table_column=True,
        ),
    },
    tags=[],
    flow=flow,
)


@workflow(metadata)
def glue_wf(
    existing_output_dir: LatchDir,
    peak2gene_archr_project: LatchDir,
    genes_of_interest: Optional[str] = None,
) -> LatchDir:
    return peak2gene_task(
        results_dir=existing_output_dir,
        project_name="",
        peak2gene_archr_project=peak2gene_archr_project,
        genes_of_interest=genes_of_interest,
    )
