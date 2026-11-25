# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", app_title="02 · Unsupervised Learning")


@app.cell
def _():
    import functools

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.datasets import (
        load_breast_cancer,
        load_digits,
        make_blobs,
        make_circles,
        make_moons,
    )
    from sklearn.decomposition import NMF, PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    PALETTE = [
        "#2563EB",
        "#EA580C",
        "#059669",
        "#7C3AED",
        "#DC2626",
        "#0891B2",
        "#CA8A04",
        "#DB2777",
        "#4F46E5",
        "#16A34A",
    ]
    return (
        AgglomerativeClustering,
        DBSCAN,
        KMeans,
        NMF,
        PALETTE,
        PCA,
        StandardScaler,
        TSNE,
        adjusted_rand_score,
        dendrogram,
        functools,
        go,
        linkage,
        load_breast_cancer,
        load_digits,
        make_blobs,
        make_circles,
        make_moons,
        make_subplots,
        mo,
        np,
        pd,
        silhouette_score,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # 02 · Unsupervised Learning: Transformations & Clustering

    Unsupervised learning discovers hidden structure in **unlabeled data** $X$—either by learning compact representations (**Transformations / Dimensionality Reduction**) or grouping similar points (**Clustering**).

    ### What You'll Explore
    1. **Lab 1 — PCA Deep Dive (Breast Cancer):** Interactive 2D/3D projections, explained variance scree plot, feature loadings heatmap, and why scaling transforms PCA.
    2. **Lab 2 — NMF & t-SNE Showdown (Handwritten Digits):** Parts-based non-negative features (`NMF`) vs. global linear components (`PCA`) vs. nonlinear local manifolds (`t-SNE`).
    3. **Lab 3 — DBSCAN Density Clustering Playground:** Interactive `eps` and `min_samples` controls distinguishing **Core**, **Border**, and **Noise** points.
    4. **Lab 4 — Clustering Showdown (`k-Means` vs. `Agglomerative` vs. `DBSCAN`):** Elbow/silhouette curves, hierarchical dendrograms, and non-convex failure modes.
    5. **Reference — Unsupervised Methods Cheat Sheet:** Assumptions, scaling rules, and key trade-offs.
    """)


@app.cell
def _(make_blobs, make_circles, make_moons, np):
    def get_cluster_dataset(name: str, noise: float = 0.06, n_samples: int = 240):
        """Return (X, y_true_for_reference) for clustering labs."""
        if name == "Moons":
            return make_moons(n_samples=n_samples, noise=noise, random_state=0)
        if name == "Circles":
            return make_circles(
                n_samples=n_samples, noise=noise * 0.75, factor=0.45, random_state=0
            )
        if name == "Anisotropic Blobs":
            _X, _y = make_blobs(n_samples=n_samples, centers=3, random_state=170)
            _transform = np.array([[0.6, -0.6], [-0.4, 0.8]])
            return _X @ _transform, _y
        if name == "Varied Density Blobs":
            return make_blobs(
                n_samples=n_samples,
                centers=[[-4, -2], [0, 3], [4, -1]],
                cluster_std=[0.45, 1.35, 0.65],
                random_state=42,
            )
        return make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=0.65 + noise * 3.0,
            random_state=42,
        )

    def style_fig(fig, title: str, height: int = 440):
        fig.update_layout(
            template="plotly_white",
            title=dict(text=title, font=dict(size=15)),
            height=height,
            margin=dict(l=55, r=25, t=60, b=50),
            paper_bgcolor="white",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A"),
        )
        return fig

    return get_cluster_dataset, style_fig


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 1 · Principal Component Analysis (PCA) Deep Dive (`Breast Cancer`)

    PCA rotates high-dimensional data onto orthogonal axes (**Principal Components**) ordered by how much variance they capture. Because variance depends on squared units, **unscaled features with large ranges hijack PC1**!

    Toggle **Apply StandardScaler** OFF below to watch PC1 suddenly absorb ~98% of the variance (just `mean area` & `worst area`) while losing separation between malignant and benign tumors.
    """)


@app.cell
def _(mo):
    pca_scale = mo.ui.switch(value=True, label="Apply StandardScaler (Recommended)")
    pca_dim = mo.ui.radio(
        options=["2D (PC1 vs PC2)", "3D (PC1 vs PC2 vs PC3)"],
        value="2D (PC1 vs PC2)",
        inline=True,
        label="Projection View",
    )
    pca_n_scree = mo.ui.slider(
        start=4, stop=15, step=1, value=10, show_value=True, label="Components in Scree Plot"
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**PCA Controls** — Inspect variance concentration, class separation, and feature loadings"
                ),
                mo.hstack([pca_scale, pca_dim, pca_n_scree], justify="start", wrap=True, gap=1.2),
            ]
        ),
        kind="info",
    )
    return pca_dim, pca_n_scree, pca_scale


@app.cell
def _(
    PALETTE,
    PCA,
    StandardScaler,
    go,
    load_breast_cancer,
    make_subplots,
    mo,
    np,
    pca_dim,
    pca_n_scree,
    pca_scale,
    style_fig,
):
    _cancer = load_breast_cancer()
    _X_in = StandardScaler().fit_transform(_cancer.data) if pca_scale.value else _cancer.data
    _n_scree = int(pca_n_scree.value)

    _pca_full = PCA(n_components=_n_scree, random_state=42)
    _X_proj = _pca_full.fit_transform(_X_in)
    _ratios = _pca_full.explained_variance_ratio_
    _cum = np.cumsum(_ratios)

    # Scatter plot (2D or 3D)
    _scatter_fig = go.Figure()
    for _cls_idx, _cls_name in enumerate(_cancer.target_names):
        _mask = _cancer.target == _cls_idx
        if pca_dim.value.startswith("3D"):
            _scatter_fig.add_trace(
                go.Scatter3d(
                    x=_X_proj[_mask, 0],
                    y=_X_proj[_mask, 1],
                    z=_X_proj[_mask, 2],
                    mode="markers",
                    name=str(_cls_name),
                    marker=dict(color=PALETTE[_cls_idx], size=4.5, opacity=0.82),
                )
            )
        else:
            _scatter_fig.add_trace(
                go.Scatter(
                    x=_X_proj[_mask, 0],
                    y=_X_proj[_mask, 1],
                    mode="markers",
                    name=str(_cls_name),
                    marker=dict(
                        color=PALETTE[_cls_idx],
                        size=7,
                        opacity=0.82,
                        line=dict(width=0.6, color="white"),
                    ),
                )
            )

    if pca_dim.value.startswith("3D"):
        _scatter_fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({_ratios[0]:.1%})",
                yaxis_title=f"PC2 ({_ratios[1]:.1%})",
                zaxis_title=f"PC3 ({_ratios[2]:.1%})",
            )
        )
    else:
        _scatter_fig.update_xaxes(title_text=f"First Principal Component ({_ratios[0]:.1%} var)")
        _scatter_fig.update_yaxes(title_text=f"Second Principal Component ({_ratios[1]:.1%} var)")

    style_fig(
        _scatter_fig,
        f"Breast Cancer PCA Projection ({'Standardized' if pca_scale.value else 'Unscaled Raw Features'})",
        height=440,
    )

    # Scree + Top Loadings Heatmap
    _diag_fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.46, 0.54],
        subplot_titles=[
            f"Explained Variance (Top {_n_scree} PCs)",
            "Top-10 Feature Loadings on PC1–PC3",
        ],
        horizontal_spacing=0.18,
    )
    _pc_labels = [f"PC{i}" for i in range(1, _n_scree + 1)]
    _diag_fig.add_trace(
        go.Bar(
            x=_pc_labels,
            y=_ratios * 100,
            name="Individual %",
            marker_color="#2563EB",
        ),
        row=1,
        col=1,
    )
    _diag_fig.add_trace(
        go.Scatter(
            x=_pc_labels,
            y=_cum * 100,
            mode="lines+markers",
            name="Cumulative %",
            line=dict(color="#EA580C", width=2.5),
        ),
        row=1,
        col=1,
    )
    _diag_fig.add_hline(y=80, line_dash="dot", line_color="#059669", row=1, col=1)

    _loadings = _pca_full.components_[:3]
    _top_feat_idx = np.argsort(np.max(np.abs(_loadings[:2]), axis=0))[-10:]
    _diag_fig.add_trace(
        go.Heatmap(
            z=_loadings[:, _top_feat_idx].T,
            x=["PC1", "PC2", "PC3"],
            y=[_cancer.feature_names[i] for i in _top_feat_idx],
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Loading", x=1.02),
        ),
        row=1,
        col=2,
    )
    _diag_fig.update_yaxes(title_text="Variance Explained (%)", range=[0, 105], row=1, col=1)
    style_fig(_diag_fig, "Scree Plot & Feature Loadings Interpretation", height=420)

    _stats = mo.hstack(
        [
            mo.stat(
                f"{_cancer.data.shape} → ({len(_X_in)}, 2)", label="Shape Reduction", bordered=True
            ),
            mo.stat(f"{_ratios[0]:.1%}", label="PC1 Variance", bordered=True),
            mo.stat(f"{_ratios[1]:.1%}", label="PC2 Variance", bordered=True),
            mo.stat(f"{_cum[1]:.1%}", label="Top-2 Total Variance", bordered=True),
        ],
        widths="equal",
        gap=0.8,
    )

    if not pca_scale.value:
        _note = mo.callout(
            mo.md(
                "**Unscaled Variance Trap:** Without `StandardScaler`, PC1 captures nearly all variance solely because `worst area` and `mean area` span thousands while smoothness/compactness sit below `1.0`. Look at the Loadings heatmap on the right—only area features are bright!"
            ),
            kind="warn",
        )
    else:
        _note = mo.callout(
            mo.md(
                "**Standardized PCA:** Every feature contributes equally on z-score scale. Notice PC1 combines overall tumor size/concavity features while PC2 contrasts texture/fractal dimensions—cleanly separating malignant and benign clusters in just 2 dimensions."
            ),
            kind="success",
        )

    mo.vstack([_stats, mo.ui.plotly(_scatter_fig), mo.ui.plotly(_diag_fig), _note], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 2 · NMF & t-SNE Showdown (`Handwritten Digits`)

    - **PCA** finds orthogonal global directions with both positive and negative weights (parts cancel each other out).
    - **NMF (Non-negative Matrix Factorization)** restricts components and weights to $\\ge 0$, extracting intuitive **additive parts** (pen strokes and loops).
    - **t-SNE** learns a nonlinear 2D embedding optimized to preserve *local neighborhoods* for visualization.
    """)


@app.cell
def _(
    NMF,
    PCA,
    StandardScaler,
    TSNE,
    functools,
    load_digits,
    mo,
    np,
    train_test_split,
):
    @functools.lru_cache(maxsize=1)
    def get_digits_subset():
        _digits = load_digits()
        # Stratified 400-sample subset keeps NMF and t-SNE crisp & responsive in browser WASM
        _X_sub, _, _y_sub, _ = train_test_split(
            _digits.data,
            _digits.target,
            train_size=400,
            stratify=_digits.target,
            random_state=42,
        )
        return _X_sub / 16.0, _y_sub

    @functools.lru_cache(maxsize=8)
    def fit_pca_nmf(n_comp: int):
        _X, _ = get_digits_subset()
        _pca = PCA(n_components=n_comp, random_state=42).fit(_X)
        _nmf = NMF(n_components=n_comp, init="nndsvda", max_iter=800, random_state=42).fit(_X)
        return _pca, _nmf

    @functools.lru_cache(maxsize=8)
    def fit_tsne_and_pca(perplexity: int):
        _X, _y = get_digits_subset()
        _Xs = StandardScaler().fit_transform(_X)
        _X_pca = PCA(n_components=2, random_state=42).fit_transform(_Xs)
        _X_tsne = TSNE(
            n_components=2, perplexity=float(perplexity), init="pca", random_state=42
        ).fit_transform(_Xs)
        return _X_pca, _X_tsne, _y

    def tile_components(components: np.ndarray) -> np.ndarray:
        """Tile K flattened 8x8 components into a square matrix with NaN borders."""
        _k = len(components)
        _side = int(np.ceil(np.sqrt(_k)))
        _canvas = np.full((_side * 9 - 1, _side * 9 - 1), np.nan)
        for _idx, _comp in enumerate(components):
            _r = _idx // _side
            _c = _idx % _side
            _canvas[_r * 9 : _r * 9 + 8, _c * 9 : _c * 9 + 8] = _comp.reshape(8, 8)
        return _canvas

    nmf_n_comp = mo.ui.radio(
        options={"4 (2×2)": 4, "9 (3×3)": 9, "16 (4×4)": 16, "25 (5×5)": 25},
        value="16 (4×4)",
        inline=True,
        label="Basis Components (k)",
    )
    nmf_sample_idx = mo.ui.slider(
        start=0, stop=99, step=1, value=8, show_value=True, label="Sample Digit Index"
    )
    tsne_perplexity = mo.ui.slider(
        start=5, stop=50, step=5, value=25, show_value=True, label="t-SNE Perplexity"
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**NMF & t-SNE Controls** — Explore additive digit strokes vs. nonlinear manifold embeddings"
                ),
                mo.hstack(
                    [nmf_n_comp, nmf_sample_idx, tsne_perplexity],
                    justify="start",
                    wrap=True,
                    gap=1.2,
                ),
            ]
        ),
        kind="info",
    )
    return (
        fit_pca_nmf,
        fit_tsne_and_pca,
        get_digits_subset,
        nmf_n_comp,
        nmf_sample_idx,
        tile_components,
        tsne_perplexity,
    )


@app.cell
def _(
    PALETTE,
    fit_pca_nmf,
    fit_tsne_and_pca,
    get_digits_subset,
    go,
    make_subplots,
    mo,
    nmf_n_comp,
    nmf_sample_idx,
    np,
    style_fig,
    tile_components,
    tsne_perplexity,
):
    _k = int(nmf_n_comp.value)
    _idx = int(nmf_sample_idx.value)
    _X_sub, _y_sub = get_digits_subset()
    _pca, _nmf = fit_pca_nmf(_k)

    _pca_rec = _pca.inverse_transform(_pca.transform(_X_sub[_idx : _idx + 1]))[0]
    _nmf_rec = _nmf.inverse_transform(_nmf.transform(_X_sub[_idx : _idx + 1]))[0]
    _pca_rmse = float(np.sqrt(np.mean((_X_sub[_idx] - _pca_rec) ** 2)))
    _nmf_rmse = float(np.sqrt(np.mean((_X_sub[_idx] - _nmf_rec) ** 2)))

    _fig_parts = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"PCA Components (k={_k}, Signed Global Patterns)",
            f"NMF Components (k={_k}, Non-Negative Additive Parts)",
        ],
        horizontal_spacing=0.08,
    )
    _fig_parts.add_trace(
        go.Heatmap(
            z=tile_components(_pca.components_),
            colorscale="RdBu_r",
            zmid=0,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    _fig_parts.add_trace(
        go.Heatmap(
            z=tile_components(_nmf.components_),
            colorscale="Blues",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    for _c in (1, 2):
        _fig_parts.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=_c)
        _fig_parts.update_xaxes(showticklabels=False, row=1, col=_c)
    style_fig(
        _fig_parts,
        "Learned Basis Components: Signed Orthogonal (PCA) vs. Parts-Based (NMF)",
        height=360,
    )

    _fig_rec = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"Original Digit '{_y_sub[_idx]}'",
            f"PCA Reconstruction (RMSE {_pca_rmse:.3f})",
            f"NMF Reconstruction (RMSE {_nmf_rmse:.3f})",
        ],
    )
    for _col, _img in enumerate([_X_sub[_idx], _pca_rec, _nmf_rec], start=1):
        _fig_rec.add_trace(
            go.Heatmap(
                z=np.clip(_img, 0, 1).reshape(8, 8),
                colorscale="Greys",
                zmin=0,
                zmax=1,
                showscale=False,
            ),
            row=1,
            col=_col,
        )
        _fig_rec.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=_col)
        _fig_rec.update_xaxes(showticklabels=False, row=1, col=_col)
    style_fig(_fig_rec, f"Digit Reconstruction Quality with {_k} Components", height=270)

    # PCA vs t-SNE 2D Manifold comparison
    _perp = int(tsne_perplexity.value)
    _X_pca2, _X_tsne2, _y_m = fit_tsne_and_pca(_perp)
    _fig_man = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "PCA 2D Linear Projection (Global Variance)",
            f"t-SNE 2D Manifold Embedding (perplexity={_perp})",
        ],
        horizontal_spacing=0.08,
    )
    for _d in range(10):
        _mask = _y_m == _d
        _fig_man.add_trace(
            go.Scatter(
                x=_X_pca2[_mask, 0],
                y=_X_pca2[_mask, 1],
                mode="markers",
                name=f"Digit {_d}",
                legendgroup=str(_d),
                marker=dict(color=PALETTE[_d], size=6, opacity=0.85),
            ),
            row=1,
            col=1,
        )
        _fig_man.add_trace(
            go.Scatter(
                x=_X_tsne2[_mask, 0],
                y=_X_tsne2[_mask, 1],
                mode="markers",
                name=f"Digit {_d}",
                legendgroup=str(_d),
                showlegend=False,
                marker=dict(color=PALETTE[_d], size=6, opacity=0.85),
            ),
            row=1,
            col=2,
        )
    style_fig(
        _fig_man,
        "Linear Projection (PCA) vs. Local Manifold Embedding (t-SNE) on Digits",
        height=440,
    )

    _caveat = mo.accordion(
        {
            "Check Your Intuition: Should you use t-SNE embeddings as input to a classifier or interpret cluster sizes literally?": mo.md(
                "**No!** `t-SNE` has two crucial caveats:\n"
                "1. **No `.transform()` for new data:** Standard `t-SNE` is non-parametric—it only embeds the fixed dataset you pass to `.fit_transform()`.\n"
                "2. **Distorted global geometry:** `t-SNE` expands dense clusters and compresses sparse ones to match local neighborhood perplexity, so relative cluster sizes and distances *between* separated islands are not faithful Euclidean distances."
            )
        }
    )

    mo.vstack(
        [mo.ui.plotly(_fig_parts), mo.ui.plotly(_fig_rec), mo.ui.plotly(_fig_man), _caveat],
        gap=0.75,
    )


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 3 · DBSCAN Density Clustering Playground (`make_moons` & More)

    Unlike centroid algorithms, **DBSCAN** groups dense regions and explicitly rejects outliers as **Noise (`-1`)**. Every point falls into one of three roles:
    - **Core Point:** Has at least `min_samples` points within radius `eps`.
    - **Border Point:** Within `eps` of a core point, but has fewer than `min_samples` neighbors itself.
    - **Noise (`-1`):** Outside `eps` of every core point.
    """)


@app.cell
def _(mo):
    db_dataset = mo.ui.dropdown(
        options=["Moons", "Circles", "Anisotropic Blobs", "Varied Density Blobs"],
        value="Moons",
        label="Dataset",
    )
    db_eps = mo.ui.slider(
        start=0.10, stop=1.10, step=0.05, value=0.35, show_value=True, label="eps (Radius)"
    )
    db_min_samples = mo.ui.slider(
        start=2, stop=15, step=1, value=5, show_value=True, label="min_samples"
    )
    db_noise = mo.ui.slider(
        start=0.02, stop=0.18, step=0.02, value=0.06, show_value=True, label="Noise σ"
    )
    db_scale = mo.ui.switch(value=True, label="StandardScaler")
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**DBSCAN Controls** — Tune neighborhood radius `eps` and density threshold `min_samples`"
                ),
                mo.hstack(
                    [db_dataset, db_eps, db_min_samples, db_noise, db_scale],
                    justify="start",
                    wrap=True,
                    gap=1.0,
                ),
            ]
        ),
        kind="info",
    )
    return db_dataset, db_eps, db_min_samples, db_noise, db_scale


@app.cell
def _(
    DBSCAN,
    PALETTE,
    StandardScaler,
    db_dataset,
    db_eps,
    db_min_samples,
    db_noise,
    db_scale,
    get_cluster_dataset,
    go,
    mo,
    np,
    silhouette_score,
    style_fig,
):
    _X, _ = get_cluster_dataset(db_dataset.value, noise=float(db_noise.value), n_samples=220)
    _X_in = StandardScaler().fit_transform(_X) if db_scale.value else _X

    _eps = float(db_eps.value)
    _ms = int(db_min_samples.value)
    _db = DBSCAN(eps=_eps, min_samples=_ms).fit(_X_in)
    _labels = _db.labels_

    _core_mask = np.zeros(len(_X_in), dtype=bool)
    _core_mask[_db.core_sample_indices_] = True
    _noise_mask = _labels == -1
    _border_mask = (~_core_mask) & (~_noise_mask)

    _unique_clusters = sorted(c for c in set(_labels) if c != -1)
    _n_clusters = len(_unique_clusters)
    _noise_pct = float(np.mean(_noise_mask))

    _non_noise = ~_noise_mask
    if len(set(_labels[_non_noise])) >= 2:
        _sil = f"{silhouette_score(_X_in[_non_noise], _labels[_non_noise]):.3f}"
    else:
        _sil = "N/A (<2 clusters)"

    _fig = go.Figure()
    for _ci in _unique_clusters:
        _color = PALETTE[_ci % len(PALETTE)]
        _cm = (_labels == _ci) & _core_mask
        _bm = (_labels == _ci) & _border_mask
        if np.any(_cm):
            _fig.add_trace(
                go.Scatter(
                    x=_X_in[_cm, 0],
                    y=_X_in[_cm, 1],
                    mode="markers",
                    name=f"Cluster {_ci} (Core)",
                    marker=dict(color=_color, size=10, line=dict(width=1, color="white")),
                )
            )
        if np.any(_bm):
            _fig.add_trace(
                go.Scatter(
                    x=_X_in[_bm, 0],
                    y=_X_in[_bm, 1],
                    mode="markers",
                    name=f"Cluster {_ci} (Border)",
                    marker=dict(
                        color=_color,
                        size=7,
                        symbol="diamond-open",
                        line=dict(width=1.8, color=_color),
                    ),
                )
            )

    if np.any(_noise_mask):
        _fig.add_trace(
            go.Scatter(
                x=_X_in[_noise_mask, 0],
                y=_X_in[_noise_mask, 1],
                mode="markers",
                name="Noise (-1)",
                marker=dict(color="#475569", size=9, symbol="x"),
            )
        )

    _fig.update_xaxes(title_text="Feature 0", scaleanchor="y", scaleratio=1)
    _fig.update_yaxes(title_text="Feature 1")
    style_fig(
        _fig,
        f"DBSCAN on {db_dataset.value} (eps={_eps:.2f}, min_samples={_ms}) — Core, Border & Noise Points",
        height=450,
    )

    _stats = mo.hstack(
        [
            mo.stat(str(_n_clusters), label="Clusters Found", bordered=True),
            mo.stat(str(int(_core_mask.sum())), label="Core Points", bordered=True),
            mo.stat(str(int(_border_mask.sum())), label="Border Points", bordered=True),
            mo.stat(f"{_noise_pct:.1%}", label="Noise Ratio (-1)", bordered=True),
            mo.stat(_sil, label="Silhouette (non-noise)", bordered=True),
        ],
        widths="equal",
        gap=0.7,
    )

    if _noise_pct > 0.45:
        _msg = mo.callout(
            mo.md(
                "**Too Much Noise (`eps` too small or `min_samples` too high):** Most points lack enough neighbors inside radius `eps`. Increase `eps` or reduce `min_samples`."
            ),
            kind="warn",
        )
    elif _n_clusters == 1 and db_dataset.value in ("Moons", "Circles"):
        _msg = mo.callout(
            mo.md(
                "**Clusters Merged (`eps` too large):** The neighborhood radius bridges the gap between the two shapes. Lower `eps` (try `0.25–0.45`) to separate them cleanly."
            ),
            kind="warn",
        )
    else:
        _msg = mo.callout(
            mo.md(
                "Notice how DBSCAN finds arbitrary curved shapes automatically **without specifying `k` in advance**—and cleanly flags stray outliers as `Noise (-1)`."
            ),
            kind="success",
        )

    mo.vstack([_stats, mo.ui.plotly(_fig), _msg], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 4 · Clustering Showdown: `k-Means` vs. `Agglomerative` vs. `DBSCAN`

    Every clustering algorithm encodes an implicit geometric assumption:
    - **k-Means** partitions space into **convex Voronoi cells** around centroids (struggles with moons, rings, and stretched blobs).
    - **Agglomerative Clustering** merges nearest points bottom-up using a **linkage rule** (`ward`, `complete`, `average`, `single`) visualized by a **Dendrogram**.
    - **DBSCAN** follows density chains of arbitrary shape, though struggles when clusters have **widely varying densities**.
    """)


@app.cell
def _(mo):
    cmp_dataset = mo.ui.dropdown(
        options=[
            "Moons",
            "Circles",
            "Anisotropic Blobs",
            "Varied Density Blobs",
            "Well-Separated Blobs",
        ],
        value="Moons",
        label="Dataset Geometry",
    )
    cmp_k = mo.ui.slider(
        start=2, stop=7, step=1, value=2, show_value=True, label="k (k-Means & Agglomerative)"
    )
    cmp_linkage = mo.ui.radio(
        options=["ward", "complete", "average", "single"],
        value="ward",
        inline=True,
        label="Agglomerative Linkage",
    )
    cmp_eps = mo.ui.slider(
        start=0.15, stop=1.00, step=0.05, value=0.35, show_value=True, label="DBSCAN eps"
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Clustering Showdown Controls** — Try `Moons` with `single` linkage or `Varied Density Blobs` to see strengths & blind spots"
                ),
                mo.hstack(
                    [cmp_dataset, cmp_k, cmp_linkage, cmp_eps], justify="start", wrap=True, gap=1.0
                ),
            ]
        ),
        kind="info",
    )
    return cmp_dataset, cmp_eps, cmp_k, cmp_linkage


@app.cell
def _(
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    PALETTE,
    StandardScaler,
    adjusted_rand_score,
    cmp_dataset,
    cmp_eps,
    cmp_k,
    cmp_linkage,
    dendrogram,
    get_cluster_dataset,
    go,
    linkage,
    make_subplots,
    mo,
    np,
    silhouette_score,
    style_fig,
):
    _X, _y_ref = get_cluster_dataset(cmp_dataset.value, noise=0.06, n_samples=210)
    _Xs = StandardScaler().fit_transform(_X)
    _k = int(cmp_k.value)
    _link = cmp_linkage.value
    _eps = float(cmp_eps.value)

    _km = KMeans(n_clusters=_k, n_init="auto", random_state=42)
    _y_km = _km.fit_predict(_Xs)
    _y_agg = AgglomerativeClustering(n_clusters=_k, linkage=_link).fit_predict(_Xs)
    _y_db = DBSCAN(eps=_eps, min_samples=5).fit_predict(_Xs)

    def _score_str(labels: np.ndarray) -> str:
        _ari = adjusted_rand_score(_y_ref, labels)
        _valid = labels != -1
        if len(set(labels[_valid])) >= 2:
            _sil = f"{silhouette_score(_Xs[_valid], labels[_valid]):.2f}"
        else:
            _sil = "N/A"
        return f"ARI vs true: {_ari:.2f} · Sil: {_sil}"

    _titles = [
        f"k-Means (k={_k})<br><sup>{_score_str(_y_km)}</sup>",
        f"Agglomerative ({_link}, k={_k})<br><sup>{_score_str(_y_agg)}</sup>",
        f"DBSCAN (eps={_eps:.2f})<br><sup>{_score_str(_y_db)}</sup>",
    ]
    _fig = make_subplots(rows=1, cols=3, subplot_titles=_titles, horizontal_spacing=0.06)

    for _col, (_lbls, _centroids) in enumerate(
        [(_y_km, _km.cluster_centers_), (_y_agg, None), (_y_db, None)], start=1
    ):
        for _cid in sorted(set(_lbls)):
            _mask = _lbls == _cid
            _fig.add_trace(
                go.Scatter(
                    x=_Xs[_mask, 0],
                    y=_Xs[_mask, 1],
                    mode="markers",
                    showlegend=False,
                    name=f"Cluster {_cid}",
                    marker=dict(
                        color="#64748B" if _cid == -1 else PALETTE[_cid % len(PALETTE)],
                        size=7,
                        symbol="x" if _cid == -1 else "circle",
                    ),
                ),
                row=1,
                col=_col,
            )
        if _centroids is not None:
            _fig.add_trace(
                go.Scatter(
                    x=_centroids[:, 0],
                    y=_centroids[:, 1],
                    mode="markers",
                    showlegend=False,
                    name="Centroids",
                    marker=dict(
                        color="#0F172A",
                        size=13,
                        symbol="star",
                        line=dict(width=1.5, color="white"),
                    ),
                ),
                row=1,
                col=_col,
            )
        _fig.update_xaxes(title_text="Scaled F0", row=1, col=_col)
    _fig.update_yaxes(title_text="Scaled F1", row=1, col=1)
    style_fig(_fig, f"Clustering Comparison on {cmp_dataset.value}", height=410)

    # Diagnostics: k-Means Elbow + Silhouette (left) & Agglomerative Dendrogram (right)
    _ks = list(range(2, 10))
    _inertias, _sils = [], []
    for _kv in _ks:
        _km_tmp = KMeans(n_clusters=_kv, n_init="auto", random_state=42).fit(_Xs)
        _inertias.append(float(_km_tmp.inertia_))
        _sils.append(float(silhouette_score(_Xs, _km_tmp.labels_)))

    _Z = linkage(_Xs, method=_link)
    _dendro = dendrogram(_Z, truncate_mode="lastp", p=20, no_plot=True)
    # Approximate horizontal cut height halfway between (k-1)th and kth last merge
    _cut_y = (
        0.5 * (_Z[-_k, 2] + _Z[-(_k - 1), 2]) if _k >= 2 and _k <= len(_Z) else float(_Z[-1, 2])
    )

    _diag = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        subplot_titles=[
            "k-Means Diagnostics: Elbow (Inertia) & Silhouette",
            f"Agglomerative Dendrogram (linkage='{_link}', cut at k={_k})",
        ],
        horizontal_spacing=0.12,
    )
    _diag.add_trace(
        go.Scatter(
            x=_ks,
            y=_inertias,
            mode="lines+markers",
            name="Inertia (Elbow)",
            line=dict(color="#2563EB", width=2.5),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    _diag.add_trace(
        go.Scatter(
            x=_ks,
            y=_sils,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#EA580C", width=2.5, dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    _diag.add_vline(x=_k, line_dash="dot", line_color="#0F172A", row=1, col=1)

    for _xs, _ys in zip(_dendro["icoord"], _dendro["dcoord"], strict=True):
        _diag.add_trace(
            go.Scatter(
                x=_xs,
                y=_ys,
                mode="lines",
                line=dict(color="#2563EB", width=1.8),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    _diag.add_hline(
        y=_cut_y,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"k={_k} cut",
        row=1,
        col=2,
    )
    _diag.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    _diag.update_yaxes(title_text="Inertia", row=1, col=1, secondary_y=False)
    _diag.update_yaxes(title_text="Silhouette Score", row=1, col=1, secondary_y=True)
    _diag.update_xaxes(
        title_text="Merged Leaf Clusters (truncated)", showticklabels=False, row=1, col=2
    )
    _diag.update_yaxes(title_text="Merge Distance", row=1, col=2)
    style_fig(_diag, "Choosing k and Inspecting Hierarchical Merges", height=390)

    mo.vstack([mo.ui.plotly(_fig), mo.ui.plotly(_diag)], gap=0.75)


@app.cell
def _(mo, pd):
    mo.md(
        """
        ---
        ## Reference · Unsupervised Learning Cheat Sheet
        """
    )
    _df = pd.DataFrame(
        [
            {
                "Algorithm": "Principal Component Analysis (PCA)",
                "Type": "Transformation",
                "Scaling?": "Yes (Critical)",
                "Strengths": "Fast linear compression, orthogonal components, exact variance accounting",
                "Weaknesses": "Linear only; components can mix positive/negative features awkwardly",
                "Key Hyperparameters": "n_components",
            },
            {
                "Algorithm": "Non-Negative Matrix Factorization (NMF)",
                "Type": "Transformation",
                "Scaling?": "Non-negative input >= 0",
                "Strengths": "Extracts additive, interpretable parts (strokes, topics, spectra)",
                "Weaknesses": "Requires non-negative data, non-unique solution depending on init",
                "Key Hyperparameters": "n_components, init, alpha_W",
            },
            {
                "Algorithm": "t-SNE",
                "Type": "Manifold Viz",
                "Scaling?": "Yes",
                "Strengths": "Reveals nonlinear local clusters dramatically well in 2D",
                "Weaknesses": "Slow on large N, no .transform(), global distances distorted",
                "Key Hyperparameters": "perplexity, learning_rate",
            },
            {
                "Algorithm": "k-Means Clustering",
                "Type": "Clustering",
                "Scaling?": "Yes",
                "Strengths": "Fast, scales to huge datasets, exposes clean cluster centroids",
                "Weaknesses": "Assumes convex equal-variance blobs; requires choosing k",
                "Key Hyperparameters": "n_clusters",
            },
            {
                "Algorithm": "Agglomerative Clustering",
                "Type": "Clustering",
                "Scaling?": "Yes",
                "Strengths": "Builds full hierarchy (dendrogram), flexible linkage shapes",
                "Weaknesses": "O(N²) memory/compute on large N; no .predict() for unseen points",
                "Key Hyperparameters": "n_clusters, linkage (ward/complete/average/single)",
            },
            {
                "Algorithm": "DBSCAN",
                "Type": "Clustering",
                "Scaling?": "Yes",
                "Strengths": "Finds arbitrary non-convex shapes, auto-detects k, flags noise (-1)",
                "Weaknesses": "Struggles with varying cluster densities; sensitive to eps",
                "Key Hyperparameters": "eps, min_samples",
            },
        ]
    )
    mo.ui.table(_df, selection=None, show_column_summaries=False)


if __name__ == "__main__":
    app.run()
