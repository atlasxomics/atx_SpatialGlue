import importlib.util
from pathlib import Path
import tempfile
import unittest

import anndata
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


_SPEC = importlib.util.spec_from_file_location(
    "wf_utils_chunked", Path(__file__).parents[1] / "wf" / "utils.py"
)
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)


class GeneChunkedH5adTests(unittest.TestCase):
    def test_backed_gene_reads_and_metadata_survive_stream_boundary(self):
        values = np.arange(3 * 513).reshape(3, 513).astype(np.float16)
        adata = anndata.AnnData(
            X=values,
            dtype=values.dtype,
            obs=pd.DataFrame(
                {"CoPro_cluster": pd.Categorical(["a", "b", "a"])},
                index=["spot1", "spot2", "spot3"],
            ),
            var=pd.DataFrame(index=[f"gene{i}" for i in range(513)]),
        )
        adata.obsm["spatial_offset"] = np.arange(6).reshape(3, 2)
        adata.uns["CoPro_cluster_nhood_enrichment"] = {"zscore": np.eye(2)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copro_sm.h5ad"
            utils.write_gene_chunked_h5ad(adata, path)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with h5py.File(path) as handle:
                self.assertEqual(handle["X"].chunks, (3, 1))
                self.assertEqual(handle["X"].dtype, np.dtype("float16"))
            backed = anndata.read_h5ad(path, backed="r")
            try:
                for gene in (0, 511, 512):
                    np.testing.assert_array_equal(backed.X[:, gene], values[:, gene])
                pd.testing.assert_frame_equal(backed.obs, adata.obs)
                pd.testing.assert_frame_equal(backed.var, adata.var)
                np.testing.assert_array_equal(
                    backed.obsm["spatial_offset"], adata.obsm["spatial_offset"]
                )
                np.testing.assert_array_equal(
                    backed.uns["CoPro_cluster_nhood_enrichment"]["zscore"], np.eye(2)
                )
            finally:
                backed.file.close()

    def test_failed_rechunk_preserves_existing_output_and_cleans_temporary_files(self):
        adata = anndata.AnnData(X=sparse.eye(3, format="csr"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copro_sm.h5ad"
            path.write_bytes(b"existing output")
            with self.assertRaisesRegex(ValueError, "requires dense X"):
                utils.write_gene_chunked_h5ad(adata, path)
            self.assertEqual(path.read_bytes(), b"existing output")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
