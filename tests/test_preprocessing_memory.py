import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from anndata import AnnData
from scipy import sparse


_UTILS_PATH = Path(__file__).parents[1] / "wf" / "utils.py"
_UTILS_SPEC = importlib.util.spec_from_file_location("wf_utils", _UTILS_PATH)
utils = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(utils)


def _adata(X, obs_names, var_names):
    return AnnData(
        X=sparse.csr_matrix(X, dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )


def test_align_modalities_inplace_preserves_rna_order_and_objects():
    rna = _adata(
        [[1], [2], [3]],
        ["b", "a", "c"],
        ["rna_gene"],
    )
    ge = _adata(
        [[30], [10], [99]],
        ["c", "b", "x"],
        ["ge_gene"],
    )
    atac = _adata(
        [[100], [300], [999]],
        ["b", "c", "y"],
        ["tile"],
    )

    rna_out, ge_out, atac_out = utils.align_modalities(
        rna, ge, atac, inplace=True
    )

    assert rna_out is rna
    assert ge_out is ge
    assert atac_out is atac
    assert rna_out.obs_names.tolist() == ["b", "c"]
    assert ge_out.obs_names.tolist() == ["b", "c"]
    assert atac_out.obs_names.tolist() == ["b", "c"]
    np.testing.assert_array_equal(ge_out.X.toarray().ravel(), [10, 30])
    np.testing.assert_array_equal(atac_out.X.toarray().ravel(), [100, 300])


def test_compute_lsi_is_float32_finite_and_does_not_modify_input():
    X = sparse.csr_matrix(
        [
            [2, 0, 1, 0, 3],
            [0, 4, 0, 1, 0],
            [1, 1, 0, 0, 2],
            [0, 2, 3, 0, 0],
            [4, 0, 0, 1, 1],
            [0, 1, 2, 2, 0],
        ],
        dtype=np.float64,
    )
    before = X.copy()

    result = utils.compute_lsi(X, n_components=2, seed=7)

    assert result.shape == (6, 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(X.toarray(), before.toarray())


def test_add_rna_features_keeps_sparse_counts_and_input_unchanged():
    counts = sparse.csr_matrix(
        [
            [2, 0, 1, 0],
            [0, 4, 0, 1],
            [1, 1, 0, 2],
            [0, 2, 3, 0],
            [4, 0, 0, 1],
            [0, 1, 2, 2],
        ],
        dtype=np.float32,
    )
    rna = _adata(
        counts,
        [f"cell_{i}" for i in range(counts.shape[0])],
        [f"gene_{i}" for i in range(counts.shape[1])],
    )
    rna.layers["counts"] = counts.copy()
    rna.var["highly_variable"] = [True, True, True, False]
    before = rna.layers["counts"].copy()

    utils.add_rna_features(rna, n_components=2)

    assert sparse.issparse(rna.layers["counts"])
    np.testing.assert_array_equal(rna.layers["counts"].toarray(), before.toarray())
    assert rna.obsm["feat"].shape == (6, 2)
    assert rna.obsm["feat"].dtype == np.float32
    assert np.isfinite(rna.obsm["feat"]).all()


def test_spatialglue_sparse_graph_matches_upstream_dense_normalization():
    directed = np.asarray(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    rows, cols = np.nonzero(directed)
    edge_table = pd.DataFrame(
        {"x": rows, "y": cols, "value": directed[rows, cols]}
    )

    actual = utils.spatialglue_normalize_sparse_adjacency(
        edge_table, directed.shape[0]
    )

    symmetric = directed + directed.T
    symmetric = np.where(symmetric > 1, 1, symmetric)
    with_self = symmetric + np.eye(directed.shape[0], dtype=np.float32)
    inv_sqrt_degree = np.power(with_self.sum(axis=1), -0.5)
    expected = (
        inv_sqrt_degree[:, np.newaxis]
        * with_self
        * inv_sqrt_degree[np.newaxis, :]
    )
    assert sparse.isspmatrix_csr(actual)
    np.testing.assert_allclose(actual.toarray(), expected, rtol=1e-6, atol=1e-6)
