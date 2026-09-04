import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
from scipy import sparse


_UTILS_PATH = Path(__file__).parents[1] / "wf" / "utils.py"
_UTILS_SPEC = importlib.util.spec_from_file_location("wf_utils_nhood", _UTILS_PATH)
utils = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(utils)


def _fake_squidpy():
    def spatial_neighbors(adata, **kwargs):
        graph = sparse.eye(adata.n_obs, format="csr")
        adata.obsp["spatial_connectivities"] = graph
        adata.obsp["spatial_distances"] = graph
        adata.uns["spatial_neighbors"] = {"params": kwargs}

    def nhood_enrichment(adata, cluster_key, **kwargs):
        n_clusters = len(adata.obs[cluster_key].cat.categories)
        values = np.arange(n_clusters * n_clusters, dtype=float).reshape(
            n_clusters, n_clusters
        )
        adata.uns[f"{cluster_key}_nhood_enrichment"] = {
            "zscore": values,
            "count": values + adata.n_obs,
        }

    return SimpleNamespace(
        gr=SimpleNamespace(
            spatial_neighbors=spatial_neighbors,
            nhood_enrichment=nhood_enrichment,
        )
    )


def test_precomputed_neighborhood_results_survive_h5ad_without_graph(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "squidpy", _fake_squidpy())
    obs = pd.DataFrame(
        {
            "CoPro_cluster": pd.Categorical(["0", "0", "1", "1"]),
            "RNA_cluster": pd.Categorical(["A", "B", "A", "B"]),
            "ATAC_cluster": pd.Categorical(["x", "x", "y", "y"]),
            "sample": pd.Categorical(["s1", "s1", "s2", "s2"]),
            "condition": pd.Categorical(["control", "control", "case", "case"]),
        },
        index=["spot1", "spot2", "spot3", "spot4"],
    )
    adata = anndata.AnnData(obs=obs)
    adata.obsm["spatial_offset"] = np.asarray(
        [[0, 0], [0, 1], [10, 0], [10, 1]], dtype=float
    )

    cluster_keys = ["CoPro_cluster", "RNA_cluster", "ATAC_cluster"]
    written = utils.precompute_neighborhood_enrichment(
        adata,
        cluster_keys=cluster_keys,
        group_keys=("sample", "condition"),
        sample_key="sample",
    )

    assert len(written) == 6
    assert not adata.obsp
    assert "spatial_neighbors" not in adata.uns
    for cluster_key in cluster_keys:
        result = adata.uns[f"{cluster_key}_nhood_enrichment"]
        assert result["zscore"].shape == (2, 2)

        grouped = adata.uns[f"{cluster_key}_nhood_enrichment_by_group"]
        assert grouped["schema_version"] == 1
        assert grouped["cluster_key"] == cluster_key
        assert {
            entry["group_key"] for entry in grouped["groups"].values()
        } == {"sample", "condition"}

    output = tmp_path / "copro_sm.h5ad"
    adata.write_h5ad(output)
    restored = anndata.read_h5ad(output)
    for key in written:
        assert key in restored.uns
    assert not restored.obsp

