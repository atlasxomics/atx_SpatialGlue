import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import gridspec


def plot_corr_volcano_broken(
    corr,
    outpath="corr_volcano.pdf",
    gene_col="gene",
    r_col="spearman_rho",
    q_col="qval_bh",
    q_thresh=0.05,
    rho_thresh=0.30,
    top_pos_labels=8,
    top_neg_labels=8,
    highlight_genes=None,
    # y-axis ranges in -log10 units (set any to None for auto)
    y_low=(0, None),          # bottom panel
    y_high=(None, None),      # top panel
    background_size=10,
    sig_size=22,
    hl_size=30,
    jitter_y=0.0,
    title="Correlation volcano"
):

    df = corr.copy()

    for c in (gene_col, r_col, q_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")

    # ---- y = -log10(q) with safe handling for q==0 ----
    q = df[q_col].astype(float).to_numpy()
    nz = q > 0
    if nz.any():
        cap_y = -np.log10(q[nz].min()) + 1.0
    else:
        cap_y = 10.0
    y = np.empty_like(q, dtype=float)
    y[nz] = -np.log10(q[nz])
    y[~nz] = cap_y

    # optional jitter to separate identical FDRs
    if jitter_y and jitter_y > 0:
        rng = np.random.default_rng(42)
        y += rng.normal(0, jitter_y, size=y.shape)

    df["_y"] = y
    df["_abs_rho"] = df[r_col].abs()

    # ---- masks ----
    sig = (df[q_col] <= q_thresh) & (df["_abs_rho"] >= rho_thresh)
    sig_pos = sig & (df[r_col] >= 0)
    sig_neg = sig & (df[r_col] < 0)

    # ---- x range (symmetric) ----
    xpad = 0.05
    xlim = max(abs(df[r_col].min()), abs(df[r_col].max())) + xpad

    # ---- choose y ranges (auto if None) ----
    y_low_min, y_low_max = y_low
    y_high_min, y_high_max = y_high

    # sensible defaults if any are None
    if y_low_min is None:
        y_low_min = 0
    if y_low_max is None:
        y_low_max = max(6.0, -math.log10(q_thresh) + 2.0)  # detailed band
    if y_high_min is None:
        y_high_min = max(y_low_max + 5.0, 200.0)           # gap between bands
    if y_high_max is None:
        y_high_max = max(np.quantile(df["_y"], 0.995), cap_y)

    # if user-specified ranges overlap, fix ordering
    if y_low_max >= y_high_min:
        y_high_min = y_low_max + 5.0

    # ---- figure (avoid tight_layout with broken axes) ----
    fig = plt.figure(figsize=(7.2, 7.4))

    # with this (60% top / 40% bottom):
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.60, 0.40], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    def _scatter(ax):
        ax.scatter(
            df[r_col],
            df["_y"],
            s=background_size,
            c="#C7C7C7",
            alpha=0.5,
            linewidths=0,
            label="All"
        )
        if sig_neg.any():
            neg = df[sig_neg]
            ax.scatter(
                neg[r_col],
                neg["_y"],
                s=sig_size,
                c="#3B82F6",
                alpha=0.9,
                linewidths=0,
                label=f"Sig (ρ ≤ -{rho_thresh:g})"
            )
        if sig_pos.any():
            pos = df[sig_pos]
            ax.scatter(
                pos[r_col],
                pos["_y"],
                s=sig_size,
                c="#EF4444",
                alpha=0.9,
                linewidths=0,
                label=f"Sig (ρ ≥ {rho_thresh:g})"
            )
        ax.axhline(
            -math.log10(q_thresh), color="k", linestyle="--", linewidth=1
        )
        ax.axvline(+rho_thresh, color="k", linestyle="--", linewidth=1)
        ax.axvline(-rho_thresh, color="k", linestyle="--", linewidth=1)

    for ax in (ax_top, ax_bot):
        _scatter(ax)
        ax.set_xlim(-xlim, xlim)

    ax_bot.set_ylim(y_low_min,  y_low_max)
    ax_top.set_ylim(y_high_min, y_high_max)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ---- break marks ----
    d = .008
    kw = dict(color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1-d, 1+d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1-d, 1+d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)

    # ---- labels ----
    def _label(ax, rows, ha):
        for _, r in rows.iterrows():
            ax.text(
                r[r_col],
                r["_y"],
                str(r[gene_col]),
                fontsize=8,
                ha=ha,
                va="bottom"
            )

    if top_pos_labels and sig_pos.any():
        lab = df[sig_pos].nlargest(top_pos_labels, "_abs_rho")
        _label(ax_top, lab[lab["_y"] >= y_high_min], "left")
        _label(ax_bot, lab[lab["_y"] < y_high_min], "left")

    if top_neg_labels and sig_neg.any():
        lab = df[sig_neg].nlargest(top_neg_labels, "_abs_rho")
        _label(ax_top, lab[lab["_y"] >= y_high_min], "right")
        _label(ax_bot, lab[lab["_y"] < y_high_min], "right")

    if highlight_genes:
        sel = df[df[gene_col].astype(str).isin(set(highlight_genes))]
        for ax in (ax_top, ax_bot):
            if not sel.empty:
                ax.scatter(sel[r_col], sel["_y"], s=hl_size, facecolors="none", edgecolors="black", linewidths=1.2, label="Highlighted")
                for _, r in sel.iterrows():
                    ax.text(r[r_col], r["_y"], str(r[gene_col]), fontsize=9, weight="bold", ha="center", va="bottom")

    # ---- cosmetics ----
    ax_top.set_title(title)
    ax_bot.set_xlabel("Spearman correlation (ρ)")
    ax_bot.set_ylabel(r"$-\,\log_{10}(\mathrm{FDR})$")
    h, l = ax_top.get_legend_handles_labels()
    if h:
        ax_top.legend(h[:3], l[:3], frameon=False, loc="upper left")

    # use subplots_adjust instead of tight_layout for broken axes
    fig.subplots_adjust(
        hspace=0.05, left=0.12, right=0.98, top=0.92, bottom=0.10
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_top_genes_bar(
    corr: pd.DataFrame,
    n: int = 20,
    fdr_thresh: float = 0.05,
    outpath="top_genes_bar.pdf",
    r_col="spearman_rho",
    gene_col="gene",
    fdr_col="qval_bh",
    use_abs=False
):
    """
    use_abs: if True, rank by |r| (uses 'abs_rho' if present, else abs of
        r_col).
    """
    df = corr.copy()

    # Basic checks
    if gene_col not in df.columns:
        raise ValueError(f"Missing gene column '{gene_col}' in CSV.")
    if r_col not in df.columns:
        # fallbacks if someone renamed columns
        for c in ["spearman_rho", "r_spearman", "spearman_r", "rho", "r", "correlation"]:
            if c in df.columns:
                r_col = c
                break
        else:
            raise ValueError("Could not find a correlation column.")
    if fdr_col and fdr_col not in df.columns:
        # allow common FDR names
        for c in ["qval_bh", "fdr", "qval", "q_value", "padj", "p_adj", "FDR", "q"]:
            if c in df.columns:
                fdr_col = c
                break
        else:
            fdr_col = None  # proceed without FDR filter

    # Filter by FDR if available
    if fdr_col:
        df = df[df[fdr_col] <= fdr_thresh]

    # Decide ranking column
    if use_abs:
        rank_col = "abs_rho" if "abs_rho" in df.columns else None
        if rank_col is None:
            df["_abs_tmp"] = df[r_col].abs()
            rank_col = "_abs_tmp"
    else:
        rank_col = r_col

    # Rank & select
    df = df[[gene_col, r_col] + ([rank_col] if rank_col not in (r_col, gene_col) else [])].dropna()
    df = df.sort_values(rank_col, ascending=False).head(n).iloc[::-1]  # reverse so largest at top

    # Plot
    plt.figure(figsize=(6, 8))
    plt.barh(df[gene_col].astype(str), df[r_col].astype(float))
    plt.xlabel("Correlation (Spearman ρ)")
    plt.title(f"Top {len(df)} genes by {'|ρ|' if use_abs else 'ρ'}")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
