import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def _load_module(name, relative_path):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corr = _load_module("wf_correlation", "wf/correlation.py")
genestats = _load_module("wf_genestats", "wf/genestats.py")


def _adata(X, obs_names, var_names):
    return AnnData(
        X=sparse.csr_matrix(X, dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )


def test_index_alignment_preserves_rna_order_without_matrix_copy():
    rna = _adata(
        [[30, 31, 32, 33], [10, 11, 12, 13], [20, 21, 22, 23], [90, 91, 92, 93]],
        ["c3", "c1", "c2", "rna_only"],
        ["A", "B", "C", "D"],
    )
    ge = _adata(
        [[220, 200, 299, 210], [320, 300, 399, 310], [920, 900, 999, 910], [120, 100, 199, 110]],
        ["c2", "c3", "ge_only", "c1"],
        ["c", "a", "x", "b"],
    )

    alignment = corr.alignment_indexers(rna, ge)
    rna_matrix = corr.IndexedMatrix(rna.X, alignment.rna_obs, alignment.rna_var)
    ge_matrix = corr.IndexedMatrix(ge.X, alignment.ge_obs, alignment.ge_var)
    rna_meta = corr.aligned_metadata(rna, alignment.rna_obs, alignment.rna_var)

    assert alignment.obs_names.tolist() == ["c3", "c1", "c2"]
    assert alignment.genes.tolist() == ["A", "B", "C"]
    np.testing.assert_array_equal(
        rna_matrix[:, :].toarray(),
        [[30, 31, 32], [10, 11, 12], [20, 21, 22]],
    )
    np.testing.assert_array_equal(
        ge_matrix[:, :].toarray(),
        [[300, 310, 320], [100, 110, 120], [200, 210, 220]],
    )
    assert rna_meta.shape == (3, 3)
    assert rna_meta.X is None


def test_indexed_chunked_correlations_match_materialized_matrices():
    x_base = sparse.csr_matrix(
        [
            [9, 1, 5, 2, 0],
            [8, 2, 4, 2, 1],
            [7, 3, 3, 2, 0],
            [6, 4, 2, 2, 1],
            [5, 5, 1, 2, 0],
            [4, 6, 0, 2, 1],
        ],
        dtype=np.float32,
    )
    y_base = sparse.csr_matrix(
        [
            [0, 6, 3, 9, 1],
            [1, 5, 3, 8, 2],
            [0, 4, 3, 7, 3],
            [1, 3, 3, 6, 4],
            [0, 2, 3, 5, 5],
            [1, 1, 3, 4, 6],
        ],
        dtype=np.float32,
    )
    rows = np.asarray([5, 3, 1, 4, 2, 0])
    x_cols = np.asarray([1, 3, 4])
    y_cols = np.asarray([4, 2, 0])
    x = corr.IndexedMatrix(x_base, rows, x_cols)
    y = corr.IndexedMatrix(y_base, rows, y_cols)
    genes = pd.Index(["g1", "g2", "g3"])

    indexed = corr.get_corr_df(x, y, genes, chunk_size=2).sort_values("gene")
    materialized = corr.get_corr_df(
        x_base[rows, :][:, x_cols],
        y_base[rows, :][:, y_cols],
        genes,
        chunk_size=2,
    ).sort_values("gene")

    pd.testing.assert_frame_equal(indexed, materialized)


def test_chunked_gene_stats_match_materialized_input():
    base = sparse.csr_matrix(
        [[0, 1, 2, 0], [3, 0, 4, 1], [0, 5, 0, 2], [6, 0, 7, 0]],
        dtype=np.float32,
    )
    rows = np.asarray([3, 1, 2])
    cols = np.asarray([2, 0, 3])
    indexed = corr.IndexedMatrix(base, rows, cols)
    genes = pd.Index(["g2", "g0", "g3"])

    actual = genestats.compute_gene_stats_matrix(
        indexed, genes, "test", chunk_size=1
    )
    expected = genestats.compute_gene_stats_matrix(
        base[rows, :][:, cols], genes, "test", chunk_size=3
    )

    pd.testing.assert_frame_equal(actual, expected)
