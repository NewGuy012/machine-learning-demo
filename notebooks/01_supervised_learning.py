# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", app_title="01 · Supervised Learning")

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.datasets import (
        load_breast_cancer,
        load_iris,
        make_blobs,
        make_circles,
        make_moons,
    )
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    PALETTE = ["#2563EB", "#EA580C", "#059669", "#7C3AED", "#DC2626"]


@app.function
def get_2d_dataset(name: str, noise: float = 0.22):
    """Return (X_2d, y, feature_names, class_names) for interactive 2D labs."""
    if name == "Iris (Petal Length vs Width)":
        iris = load_iris()
        return (
            iris.data[:, [2, 3]],
            iris.target,
            ["Petal length (cm)", "Petal width (cm)"],
            list(iris.target_names),
        )
    if name == "Moons (Nonlinear)":
        X, y = make_moons(n_samples=260, noise=noise, random_state=42)
        return X, y, ["Feature 0", "Feature 1"], ["Class 0", "Class 1"]
    if name == "Circles (Concentric)":
        X, y = make_circles(n_samples=260, noise=noise * 0.75, factor=0.48, random_state=42)
        return X, y, ["Feature 0", "Feature 1"], ["Outer", "Inner"]
    if name == "Breast Cancer (Area vs Smoothness)":
        cancer = load_breast_cancer()
        return (
            cancer.data[:, [3, 4]],
            cancer.target,
            ["mean area", "mean smoothness"],
            list(cancer.target_names),
        )
    if name == "XOR (Checkerboard)":
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.5, 1.5, size=(280, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        X = X + rng.normal(0, noise * 0.45, size=X.shape)
        return X, y, ["Feature 0", "Feature 1"], ["Quadrant II/IV", "Quadrant I/III"]
    X, y = make_blobs(
        n_samples=260,
        centers=[[-1.2, -1.0], [1.2, 1.0], [1.4, -1.2]],
        cluster_std=0.85 + noise,
        random_state=42,
    )
    return X, y, ["Feature 0", "Feature 1"], ["Blob 0", "Blob 1", "Blob 2"]


@app.function
def make_mesh(X: np.ndarray, grid_size: int = 110):
    """Build a padded 2D evaluation mesh around X."""
    pad_x = max((X[:, 0].max() - X[:, 0].min()) * 0.12, 1e-6)
    pad_y = max((X[:, 1].max() - X[:, 1].min()) * 0.12, 1e-6)
    xx = np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, grid_size)
    yy = np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, grid_size)
    gx, gy = np.meshgrid(xx, yy)
    return xx, yy, np.c_[gx.ravel(), gy.ravel()]


@app.function
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


@app.function
def style_fig(fig, title: str, height: int = 440):
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 15}},
        height=height,
        margin={"l": 55, "r": 25, "t": 60, "b": 50},
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font={"color": "#0F172A"},
    )
    return fig


@app.function
def add_boundary_heatmap(
    fig,
    xx: np.ndarray,
    yy: np.ndarray,
    z: np.ndarray,
    n_cls: int,
    *,
    alpha: float = 0.28,
    row: int | None = None,
    col: int | None = None,
):
    pos = {"row": row, "col": col} if row is not None else {}
    cscale = [[i / max(n_cls - 1, 1), hex_to_rgba(PALETTE[i], alpha)] for i in range(n_cls)]
    fig.add_trace(
        go.Heatmap(
            x=xx,
            y=yy,
            z=z,
            colorscale=cscale,
            zmin=0,
            zmax=max(n_cls - 1, 1),
            showscale=False,
            hoverinfo="skip",
        ),
        **pos,
    )


@app.function
def add_train_test_points(
    fig,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    class_names: list[str],
    *,
    row: int | None = None,
    col: int | None = None,
):
    pos = {"row": row, "col": col} if row is not None else {}
    for c_idx, c_name in enumerate(class_names):
        m_tr = y_tr == c_idx
        m_te = y_te == c_idx
        fig.add_trace(
            go.Scatter(
                x=X_tr[m_tr, 0],
                y=X_tr[m_tr, 1],
                mode="markers",
                name=f"{c_name} (train)",
                marker={"color": PALETTE[c_idx], "size": 7, "opacity": 0.74},
            ),
            **pos,
        )
        fig.add_trace(
            go.Scatter(
                x=X_te[m_te, 0],
                y=X_te[m_te, 1],
                mode="markers",
                name=f"{c_name} (test)",
                marker={
                    "color": PALETTE[c_idx],
                    "size": 9,
                    "symbol": "diamond",
                    "line": {"width": 1.2, "color": "#0F172A"},
                },
            ),
            **pos,
        )


@app.function
def add_boundary_panel(
    fig,
    model,
    xx: np.ndarray,
    yy: np.ndarray,
    grid: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    class_names: list[str],
    *,
    row: int,
    col: int,
    showlegend: bool,
    marker_size: int = 7,
    line_width: float = 1.0,
):
    n_cls = len(class_names)
    z = model.predict(grid).reshape(len(yy), len(xx))
    add_boundary_heatmap(fig, xx, yy, z, n_cls, row=row, col=col)
    for ci, cname in enumerate(class_names):
        mask = y_te == ci
        fig.add_trace(
            go.Scatter(
                x=X_te[mask, 0],
                y=X_te[mask, 1],
                mode="markers",
                name=cname,
                showlegend=showlegend,
                marker={
                    "color": PALETTE[ci],
                    "size": marker_size,
                    "line": {"width": line_width, "color": "white"},
                },
            ),
            row=row,
            col=col,
        )


@app.cell
def _():
    mo.md("""
    # 01 · Supervised Learning: Interactive Visual Labs

    Supervised learning trains models on **labeled examples** $(X, y)$ to predict categories (**Classification**) or continuous values (**Regression**) on unseen data.

    > **How to use this notebook:** Move any slider, dropdown, or toggle below. Marimo reactively re-runs only the affected cells so you can *see* decision boundaries, bias–variance curves, feature scaling effects, and regularization in real time.

    ### What You'll Explore
    1. **Lab 1 — Iris Explorer & k-Nearest Neighbors:** Interactive scatter matrix, decision boundaries, and the $k$-NN complexity curve.
    2. **Lab 2 — Kernelized SVMs & Why Scaling Matters:** Breast Cancer scaling experiment with log-scale feature visuals and ringed support vectors.
    3. **Lab 3 — Trees & Ensembles:** Side-by-side Decision Trees, Random Forests, and Gradient Boosted Decision Trees.
    4. **Lab 4 — Classifier Showdown:** All 8 supervised algorithm families (including Linear Models, Naive Bayes, and Neural Networks) on the exact same geometry.
    5. **Lab 5 — Regression & Regularization:** Polynomial degree vs. `OLS`, `Ridge` ($L_2$), and `Lasso` ($L_1$) weight shrinkage.
    6. **Reference — Supervised Algorithm Cheat Sheet:** Strengths, weaknesses, scaling sensitivity, and key hyperparameters.
    """)


@app.cell
def _():
    mo.md("""
    ---
    ## Lab 1 · Iris Feature Explorer & k-Nearest Neighbors ($k$-NN)

    Before fitting any classifier, inspect how features separate the classes. On the classic **Iris** dataset, *petal* dimensions nearly separate all three species, whereas *sepal* dimensions overlap heavily.

    Then use the **$k$-NN playground** to watch how neighborhood size $k$ controls the **bias–variance tradeoff**: small $k$ memorizes noise (high variance), while large $k$ over-smooths boundaries (high bias).
    """)


@app.cell
def _():
    iris_raw = load_iris()
    _features = list(iris_raw.feature_names)

    iris_x_col = mo.ui.dropdown(options=_features, value=_features[2], label="X Feature")
    iris_y_col = mo.ui.dropdown(options=_features, value=_features[3], label="Y Feature")
    iris_view_mode = mo.ui.radio(
        options=["2D Pair Explorer", "Full 4×4 Scatter Matrix (SPLOM)"],
        value="2D Pair Explorer",
        inline=True,
        label="Iris Exploration View",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md("**Iris Dataset Explorer** — Interactive feature pairs & 4×4 scatter matrix"),
                mo.hstack(
                    [iris_view_mode, iris_x_col, iris_y_col], justify="start", wrap=True, gap=1.0
                ),
            ]
        ),
        kind="neutral",
    )
    return iris_raw, iris_view_mode, iris_x_col, iris_y_col


@app.cell
def _(iris_raw, iris_view_mode, iris_x_col, iris_y_col):
    _features = list(iris_raw.feature_names)
    _fig = go.Figure()
    if iris_view_mode.value == "Full 4×4 Scatter Matrix (SPLOM)":
        for _ci, _species in enumerate(iris_raw.target_names):
            _mask = iris_raw.target == _ci
            _fig.add_trace(
                go.Splom(
                    dimensions=[
                        {"label": _f, "values": iris_raw.data[_mask, _fi]}
                        for _fi, _f in enumerate(_features)
                    ],
                    name=str(_species),
                    diagonal_visible=False,
                    marker={"color": PALETTE[_ci], "size": 5, "opacity": 0.82},
                )
            )
        style_fig(_fig, "Iris 4×4 Interactive Scatter Matrix (hover or zoom any panel)", height=560)
    else:
        _xi = _features.index(iris_x_col.value)
        _yi = _features.index(iris_y_col.value)
        for _ci, _species in enumerate(iris_raw.target_names):
            _mask = iris_raw.target == _ci
            _fig.add_trace(
                go.Scatter(
                    x=iris_raw.data[_mask, _xi],
                    y=iris_raw.data[_mask, _yi],
                    mode="markers",
                    name=str(_species),
                    marker={
                        "color": PALETTE[_ci],
                        "size": 8,
                        "opacity": 0.88,
                        "line": {"width": 0.8, "color": "white"},
                    },
                )
            )
        _fig.update_xaxes(title_text=iris_x_col.value)
        _fig.update_yaxes(title_text=iris_y_col.value)
        style_fig(
            _fig,
            f"Iris Feature Pair: {iris_x_col.value} vs {iris_y_col.value}",
            height=450,
        )
    mo.ui.plotly(_fig)


@app.cell
def _():
    knn_dataset = mo.ui.dropdown(
        options=[
            "Iris (Petal Length vs Width)",
            "Moons (Nonlinear)",
            "Circles (Concentric)",
            "Overlapping Blobs",
        ],
        value="Moons (Nonlinear)",
        label="Dataset",
    )
    knn_k = mo.ui.slider(
        start=1, stop=25, step=1, value=5, show_value=True, label="k (n_neighbors)"
    )
    knn_weights = mo.ui.radio(
        options=["uniform", "distance"],
        value="uniform",
        inline=True,
        label="Neighbor Weighting",
    )
    knn_metric = mo.ui.radio(
        options=["euclidean", "manhattan"],
        value="euclidean",
        inline=True,
        label="Distance Metric",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md("**k-NN Controls** — Adjust neighborhood size and distance geometry"),
                mo.hstack(
                    [knn_dataset, knn_k, knn_weights, knn_metric],
                    justify="start",
                    wrap=True,
                    gap=1.2,
                ),
            ]
        ),
        kind="info",
    )
    return knn_dataset, knn_k, knn_metric, knn_weights


@app.cell
def _(knn_dataset, knn_k, knn_metric, knn_weights):
    _X, _y, _feat_names, _cls_names = get_2d_dataset(knn_dataset.value, noise=0.24)
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, test_size=0.30, stratify=_y, random_state=42
    )
    _scaler = StandardScaler()
    _X_tr_s = _scaler.fit_transform(_X_tr)
    _X_te_s = _scaler.transform(_X_te)

    _k = int(knn_k.value)
    _knn = KNeighborsClassifier(n_neighbors=_k, weights=knn_weights.value, metric=knn_metric.value)
    _knn.fit(_X_tr_s, _y_tr)
    _tr_acc = float(_knn.score(_X_tr_s, _y_tr))
    _te_acc = float(_knn.score(_X_te_s, _y_te))
    _wrong_mask = _knn.predict(_X_te_s) != _y_te
    _n_wrong = int(_wrong_mask.sum())

    # Evaluate mesh in standardized space, then map grid back to original units
    _xx_s, _yy_s, _grid_s = make_mesh(_scaler.transform(_X), grid_size=105)
    _xx = _xx_s * _scaler.scale_[0] + _scaler.mean_[0]
    _yy = _yy_s * _scaler.scale_[1] + _scaler.mean_[1]
    _grid_pred = _knn.predict(_grid_s).reshape(len(_yy), len(_xx))

    _k_vals = list(range(1, 26))
    _sweep_tr, _sweep_te = [], []
    for _kv in _k_vals:
        _m = KNeighborsClassifier(
            n_neighbors=_kv, weights=knn_weights.value, metric=knn_metric.value
        ).fit(_X_tr_s, _y_tr)
        _sweep_tr.append(float(_m.score(_X_tr_s, _y_tr)))
        _sweep_te.append(float(_m.score(_X_te_s, _y_te)))

    _fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=[
            f"Decision Boundary (k={_k}, {knn_weights.value})",
            "Model Complexity Curve (Bias–Variance)",
        ],
        horizontal_spacing=0.10,
    )
    add_boundary_heatmap(_fig, _xx, _yy, _grid_pred, len(_cls_names), alpha=0.25, row=1, col=1)
    add_train_test_points(_fig, _X_tr, _y_tr, _X_te, _y_te, _cls_names, row=1, col=1)

    if _n_wrong > 0:
        _fig.add_trace(
            go.Scatter(
                x=_X_te[_wrong_mask, 0],
                y=_X_te[_wrong_mask, 1],
                mode="markers",
                name="Misclassified (test)",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 15,
                    "symbol": "circle",
                    "line": {"width": 2.5, "color": "#DC2626"},
                },
            ),
            row=1,
            col=1,
        )

    for _name, _vals, _color in [
        ("Train Accuracy", _sweep_tr, "#2563EB"),
        ("Test Accuracy", _sweep_te, "#EA580C"),
    ]:
        _fig.add_trace(
            go.Scatter(
                x=_k_vals,
                y=_vals,
                mode="lines+markers",
                name=_name,
                line={"color": _color, "width": 2.5},
                marker={"size": 5},
            ),
            row=1,
            col=2,
        )
    _fig.add_vline(
        x=_k,
        line_dash="dash",
        line_color="#0F172A",
        line_width=2,
        annotation_text=f"k={_k}",
        annotation_position="top right",
        row=1,
        col=2,
    )
    _fig.update_xaxes(title_text=_feat_names[0], row=1, col=1)
    _fig.update_yaxes(title_text=_feat_names[1], row=1, col=1)
    _fig.update_xaxes(title_text="Number of Neighbors (k)", row=1, col=2)
    _fig.update_yaxes(title_text="Accuracy", range=[0.5, 1.03], row=1, col=2)
    style_fig(_fig, f"k-NN Explorer on {knn_dataset.value}", height=460)

    _stats = mo.hstack(
        [
            mo.stat(f"{_tr_acc:.1%}", label="Train Accuracy", bordered=True),
            mo.stat(f"{_te_acc:.1%}", label="Test Accuracy", bordered=True),
            mo.stat(
                f"{(_tr_acc - _te_acc):+.1%}",
                label="Generalization Gap",
                caption="Train − Test",
                bordered=True,
            ),
            mo.stat(f"{_n_wrong} / {len(_y_te)}", label="Test Errors", bordered=True),
        ],
        widths="equal",
        gap=0.8,
    )

    if _k <= 2 and knn_weights.value == "uniform":
        _advice = mo.callout(
            mo.md(
                "**High Variance (Overfitting zone):** At `k=1` or `2`, the boundary carves tiny jagged islands around individual training points (notice ~100% train accuracy with a wider generalization gap)."
            ),
            kind="warn",
        )
    elif _k >= 17:
        _advice = mo.callout(
            mo.md(
                "**Higher Bias (Underfitting risk):** Very large `k` averages across distant points, smoothing out fine structure—especially on curved moons or concentric circles."
            ),
            kind="warn",
        )
    else:
        _advice = mo.callout(
            mo.md(
                f"**Balanced Complexity (`k={_k}`):** Smooth boundary capturing the underlying shape while ignoring isolated training noise. Try switching to `distance` weighting to give closer neighbors stronger votes."
            ),
            kind="success",
        )

    _quiz = mo.accordion(
        {
            "Check Your Intuition: Why does k=1 with uniform weights always achieve 100% training accuracy?": mo.md(
                "Because each training point's **single closest neighbor in the training set is itself** (distance = 0), so a 1-NN classifier essentially acts as a lookup table on the training data—guaranteeing zero training error even on pure random noise!"
            )
        }
    )

    mo.vstack([_stats, mo.ui.plotly(_fig), _advice, _quiz], gap=0.75)


@app.cell
def _():
    mo.md("""
    ---
    ## Lab 2 · Kernelized SVMs & Why Feature Scaling Matters

    Support Vector Machines with an **RBF kernel** classify points using pairwise Euclidean distances $K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$. When features live on wildly different scales—like Breast Cancer's `mean area` (hundreds to thousands) vs. `mean smoothness` (`0.05`–`0.16`)—one feature dominates every distance calculation.

    Below, explore **both** the full 30-feature Breast Cancer scaling effect and an interactive 2D **Support Vector Geometry Lab**.
    """)


@app.cell
def _():
    _cancer = load_breast_cancer()
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(_cancer.data, _cancer.target, random_state=0)

    _configs = [
        ("Unscaled (Raw)", None),
        ("MinMaxScaler [0, 1]", MinMaxScaler()),
        ("StandardScaler (z-score)", StandardScaler()),
    ]
    _names, _tr_scores, _te_scores = [], [], []
    for _label, _scaler in _configs:
        if _scaler is None:
            _X1, _X2 = _X_tr, _X_te
        else:
            _X1 = _scaler.fit_transform(_X_tr)
            _X2 = _scaler.transform(_X_te)
        _m = SVC().fit(_X1, _y_tr)
        _names.append(_label)
        _tr_scores.append(float(_m.score(_X1, _y_tr)))
        _te_scores.append(float(_m.score(_X2, _y_te)))

    _order = np.argsort(_X_tr.max(axis=0) - _X_tr.min(axis=0))
    _sample_idx = np.concatenate([_order[:5], _order[-5:]])

    _fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=[
            "Breast Cancer Feature Scales (Log10 Min–Max Spread)",
            "SVC(RBF) Accuracy Before vs. After Scaling (30 Features)",
        ],
        horizontal_spacing=0.12,
    )
    for _idx in _sample_idx:
        _col_min = float(_X_tr[:, _idx].min())
        _col_max = float(_X_tr[:, _idx].max())
        _fname = _cancer.feature_names[_idx]
        _fig.add_trace(
            go.Scatter(
                x=[max(_col_min, 1e-4), max(_col_max, 1e-4)],
                y=[_fname, _fname],
                mode="lines+markers",
                line={"color": "#2563EB", "width": 4},
                marker={"size": 7, "color": "#EA580C"},
                showlegend=False,
                hovertemplate=f"{_fname}: [{_col_min:.4g}, {_col_max:.4g}]<extra></extra>",
            ),
            row=1,
            col=1,
        )
    _fig.update_xaxes(type="log", title_text="Feature Value Range (log scale)", row=1, col=1)

    for _bar_name, _scores, _color in [
        ("Train Accuracy", _tr_scores, "#2563EB"),
        ("Test Accuracy", _te_scores, "#059669"),
    ]:
        _fig.add_trace(
            go.Bar(
                name=_bar_name,
                x=_names,
                y=_scores,
                marker_color=_color,
                text=[f"{s:.1%}" for s in _scores],
                textposition="outside",
            ),
            row=1,
            col=2,
        )
    _fig.update_yaxes(title_text="Accuracy", range=[0.5, 1.09], row=1, col=2)
    _fig.update_layout(barmode="group")
    style_fig(_fig, "Why RBF SVMs Require Feature Scaling (Breast Cancer, 30 Features)", height=430)

    mo.ui.plotly(_fig)


@app.cell
def _():
    svm_dataset = mo.ui.dropdown(
        options=[
            "Breast Cancer (Area vs Smoothness)",
            "Moons (Nonlinear)",
            "Circles (Concentric)",
        ],
        value="Breast Cancer (Area vs Smoothness)",
        label="2D Dataset",
    )
    svm_scaler = mo.ui.radio(
        options=["None (Unscaled)", "MinMaxScaler", "StandardScaler"],
        value="None (Unscaled)",
        inline=True,
        label="Feature Scaler",
    )
    svm_kernel = mo.ui.radio(
        options=["rbf", "linear", "poly"],
        value="rbf",
        inline=True,
        label="Kernel",
    )
    svm_c = mo.ui.dropdown(
        options={"0.01": 0.01, "0.1": 0.1, "1": 1.0, "10": 10.0, "100": 100.0},
        value="1",
        label="Regularization C",
    )
    svm_gamma = mo.ui.dropdown(
        options={"scale": "scale", "0.05": 0.05, "0.2": 0.2, "1.0": 1.0, "5.0": 5.0, "20.0": 20.0},
        value="scale",
        label="Gamma (RBF/Poly)",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Interactive SVM Geometry Lab** — Toggle `Feature Scaler` on *Breast Cancer (Area vs Smoothness)* to watch the RBF boundary rescue itself!"
                ),
                mo.hstack(
                    [svm_dataset, svm_scaler, svm_kernel, svm_c, svm_gamma],
                    justify="start",
                    wrap=True,
                    gap=1.0,
                ),
            ]
        ),
        kind="info",
    )
    return svm_c, svm_dataset, svm_gamma, svm_kernel, svm_scaler


@app.cell
def _(svm_c, svm_dataset, svm_gamma, svm_kernel, svm_scaler):
    _X, _y, _fnames, _cnames = get_2d_dataset(svm_dataset.value, noise=0.20)
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, test_size=0.28, stratify=_y, random_state=0
    )

    _scaler_cls = {"MinMaxScaler": MinMaxScaler, "StandardScaler": StandardScaler}.get(
        svm_scaler.value
    )
    if _scaler_cls is not None:
        _sc = _scaler_cls()
        _X_tr_m = _sc.fit_transform(_X_tr)
        _X_te_m = _sc.transform(_X_te)
    else:
        _sc = None
        _X_tr_m, _X_te_m = _X_tr, _X_te

    _svc = SVC(
        kernel=svm_kernel.value,
        C=float(svm_c.value),
        gamma=svm_gamma.value if svm_kernel.value != "linear" else "scale",
    ).fit(_X_tr_m, _y_tr)
    _tr_acc = float(_svc.score(_X_tr_m, _y_tr))
    _te_acc = float(_svc.score(_X_te_m, _y_te))
    _sv_idx = _svc.support_
    _n_sv = len(_sv_idx)

    _xx_raw, _yy_raw, _grid_raw = make_mesh(_X, grid_size=105)
    _grid_m = _sc.transform(_grid_raw) if _sc is not None else _grid_raw
    _dec = _svc.decision_function(_grid_m).reshape(len(_yy_raw), len(_xx_raw))

    _fig = go.Figure()
    _fig.add_trace(
        go.Contour(
            x=_xx_raw,
            y=_yy_raw,
            z=np.clip(_dec, -2.5, 2.5),
            colorscale=[[0, "#DBEAFE"], [0.5, "#FFFFFF"], [1, "#FFEDD5"]],
            zmid=0,
            contours={"start": -2.0, "end": 2.0, "size": 0.5, "showlabels": False},
            opacity=0.88,
            colorbar={"title": "Decision<br>Score"},
            hoverinfo="skip",
        )
    )
    _fig.add_trace(
        go.Contour(
            x=_xx_raw,
            y=_yy_raw,
            z=_dec,
            contours={"coloring": "none", "start": -1, "end": 1, "size": 1, "showlabels": True},
            line={"color": "#0F172A", "width": 2},
            showscale=False,
            hoverinfo="skip",
        )
    )
    add_train_test_points(_fig, _X_tr, _y_tr, _X_te, _y_te, _cnames)
    _fig.add_trace(
        go.Scatter(
            x=_X_tr[_sv_idx, 0],
            y=_X_tr[_sv_idx, 1],
            mode="markers",
            name=f"Support Vectors ({_n_sv})",
            marker={
                "color": "rgba(0,0,0,0)",
                "size": 13,
                "line": {"width": 2.0, "color": "#0F172A"},
            },
        )
    )
    _fig.update_xaxes(title_text=_fnames[0])
    _fig.update_yaxes(title_text=_fnames[1])
    style_fig(
        _fig,
        f"SVM ({svm_kernel.value.upper()}, C={svm_c.value}, scaler={svm_scaler.value}) — Margins [-1, 0, +1] & Support Vectors",
        height=470,
    )

    _stats = mo.hstack(
        [
            mo.stat(f"{_tr_acc:.1%}", label="Train Accuracy", bordered=True),
            mo.stat(f"{_te_acc:.1%}", label="Test Accuracy", bordered=True),
            mo.stat(
                f"{_n_sv} ({_n_sv / len(_X_tr):.0%})",
                label="Support Vectors",
                caption="Points defining the margin",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=0.8,
    )

    if (
        svm_dataset.value.startswith("Breast Cancer")
        and svm_scaler.value == "None (Unscaled)"
        and svm_kernel.value == "rbf"
    ):
        _note = mo.callout(
            mo.md(
                "**Scaling Collapse Detected:** Without scaling, `mean area` spans ~`200–2500` while `mean smoothness` spans `0.05–0.16`. Notice almost vertical bands or majority-class predictions because `mean smoothness` is invisible to the RBF distance! Switch `Feature Scaler` to `MinMaxScaler` or `StandardScaler`."
            ),
            kind="danger",
        )
    else:
        _note = mo.callout(
            mo.md(
                "Black rings mark the **support vectors**—the only training points that actually determine the decision boundary. Lower `C` widens the margin (more support vectors, stronger regularization); higher `gamma` shrinks each point's influence radius."
            ),
            kind="info",
        )

    mo.vstack([_stats, mo.ui.plotly(_fig), _note], gap=0.75)


@app.cell
def _():
    mo.md("""
    ---
    ## Lab 3 · Decision Trees & Tree Ensembles (`Random Forest` & `Gradient Boosting`)

    Trees learn **axis-aligned if/else splits** and do **not** require feature scaling. A single deep tree overfits easily, whereas ensembles tame variance (**Random Forest**: averaging randomized trees) or bias (**Gradient Boosted Decision Trees**: fitting trees sequentially to residual errors).
    """)


@app.cell
def _():
    tree_dataset = mo.ui.dropdown(
        options=["Moons (Nonlinear)", "Circles (Concentric)", "Breast Cancer (Area vs Smoothness)"],
        value="Moons (Nonlinear)",
        label="Dataset",
    )
    tree_depth = mo.ui.slider(start=1, stop=12, step=1, value=4, show_value=True, label="max_depth")
    tree_n_est = mo.ui.slider(
        start=5, stop=100, step=5, value=30, show_value=True, label="n_estimators (Ensembles)"
    )
    gbdt_lr = mo.ui.dropdown(
        options={"0.05": 0.05, "0.1": 0.1, "0.3": 0.3, "0.8": 0.8},
        value="0.1",
        label="GBDT learning_rate",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Tree & Ensemble Controls** — Watch axis-aligned boxes smooth out into ensemble boundaries"
                ),
                mo.hstack(
                    [tree_dataset, tree_depth, tree_n_est, gbdt_lr],
                    justify="start",
                    wrap=True,
                    gap=1.0,
                ),
            ]
        ),
        kind="info",
    )
    return gbdt_lr, tree_dataset, tree_depth, tree_n_est


@app.cell
def _(gbdt_lr, tree_dataset, tree_depth, tree_n_est):
    _X, _y, _fnames, _cnames = get_2d_dataset(tree_dataset.value, noise=0.22)
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, test_size=0.28, stratify=_y, random_state=42
    )
    _depth = int(tree_depth.value)
    _n_est = int(tree_n_est.value)
    _lr = float(gbdt_lr.value)

    _models = [
        ("Decision Tree", DecisionTreeClassifier(max_depth=_depth, random_state=42)),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=_n_est, max_depth=_depth, random_state=42),
        ),
        (
            "Gradient Boosted Trees",
            GradientBoostingClassifier(
                n_estimators=_n_est, max_depth=min(_depth, 5), learning_rate=_lr, random_state=42
            ),
        ),
    ]

    _xx, _yy, _grid = make_mesh(_X, grid_size=95)
    _titles, _fitted = [], []
    for _name, _m in _models:
        _m.fit(_X_tr, _y_tr)
        _titles.append(
            f"{_name}<br><sup>Train {_m.score(_X_tr, _y_tr):.1%} · Test {_m.score(_X_te, _y_te):.1%}</sup>"
        )
        _fitted.append(_m)

    _fig = make_subplots(rows=1, cols=3, subplot_titles=_titles, horizontal_spacing=0.06)
    for _col, _m in enumerate(_fitted, start=1):
        add_boundary_panel(
            _fig,
            _m,
            _xx,
            _yy,
            _grid,
            _X_te,
            _y_te,
            _cnames,
            row=1,
            col=_col,
            showlegend=(_col == 1),
        )
        _fig.update_xaxes(title_text=_fnames[0], row=1, col=_col)
    _fig.update_yaxes(title_text=_fnames[1], row=1, col=1)
    style_fig(
        _fig,
        f"Tree vs. Bagging vs. Boosting (max_depth={_depth}, n_estimators={_n_est})",
        height=420,
    )

    _cancer = load_breast_cancer()
    _Xc_tr, _Xc_te, _yc_tr, _yc_te = train_test_split(
        _cancer.data, _cancer.target, stratify=_cancer.target, random_state=42
    )
    _depths = list(range(1, 11))
    _dt_te = [
        float(
            DecisionTreeClassifier(max_depth=_d, random_state=42)
            .fit(_Xc_tr, _yc_tr)
            .score(_Xc_te, _yc_te)
        )
        for _d in _depths
    ]
    _rf_te = [
        float(
            RandomForestClassifier(n_estimators=40, max_depth=_d, random_state=42)
            .fit(_Xc_tr, _yc_tr)
            .score(_Xc_te, _yc_te)
        )
        for _d in _depths
    ]
    _rf_full = RandomForestClassifier(n_estimators=60, max_depth=_depth, random_state=42).fit(
        _Xc_tr, _yc_tr
    )
    _top_idx = _rf_full.feature_importances_.argsort()[-8:]

    _fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Breast Cancer (30D): Depth vs Test Accuracy",
            f"Random Forest Top-8 Feature Importances (depth={_depth})",
        ],
        horizontal_spacing=0.18,
    )
    for _name, _vals, _color in [
        ("Single Tree Test Acc", _dt_te, "#EA580C"),
        ("Random Forest Test Acc", _rf_te, "#059669"),
    ]:
        _fig2.add_trace(
            go.Scatter(
                x=_depths,
                y=_vals,
                mode="lines+markers",
                name=_name,
                line={"color": _color, "width": 2.5},
            ),
            row=1,
            col=1,
        )
    _fig2.add_vline(x=_depth, line_dash="dash", line_color="#0F172A", row=1, col=1)
    _fig2.add_trace(
        go.Bar(
            x=_rf_full.feature_importances_[_top_idx],
            y=[_cancer.feature_names[i] for i in _top_idx],
            orientation="h",
            marker_color="#2563EB",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig2.update_xaxes(title_text="max_depth", row=1, col=1)
    _fig2.update_yaxes(title_text="Test Accuracy", range=[0.85, 1.0], row=1, col=1)
    _fig2.update_xaxes(title_text="Gini Importance", row=1, col=2)
    style_fig(_fig2, "Why Ensembles Outperform Single Trees on Real Tabular Data", height=380)

    mo.vstack([mo.ui.plotly(_fig), mo.ui.plotly(_fig2)], gap=0.75)


@app.cell
def _():
    mo.md("""
    ---
    ## Lab 4 · Classifier Showdown: All 8 Supervised Families Side-by-Side

    How do **Linear Models** (`LogisticRegression`), **Gaussian Naive Bayes**, **k-NN**, **Trees/Ensembles**, **RBF SVM**, and **MLP Neural Networks** solve the exact same 2D geometry? Select a dataset shape—especially **XOR (Checkerboard)** or **Circles**—to see linear vs. nonlinear boundaries immediately.
    """)


@app.cell
def _():
    showdown_dataset = mo.ui.dropdown(
        options=[
            "Moons (Nonlinear)",
            "Circles (Concentric)",
            "XOR (Checkerboard)",
            "Overlapping Blobs",
        ],
        value="XOR (Checkerboard)",
        label="Geometry",
    )
    showdown_c = mo.ui.dropdown(
        options={"0.05": 0.05, "0.5": 0.5, "1.0": 1.0, "10.0": 10.0},
        value="1.0",
        label="Linear / SVM C",
    )
    showdown_mlp = mo.ui.dropdown(
        options={"(16,)": (16,), "(64, 32)": (64, 32), "(80, 80)": (80, 80)},
        value="(64, 32)",
        label="MLP Hidden Layers",
    )
    showdown_act = mo.ui.radio(
        options=["relu", "tanh"],
        value="tanh",
        inline=True,
        label="MLP Activation",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Classifier Showdown Controls** — Compare all 8 supervised classifier families simultaneously"
                ),
                mo.hstack(
                    [showdown_dataset, showdown_c, showdown_mlp, showdown_act],
                    justify="start",
                    wrap=True,
                    gap=1.0,
                ),
            ]
        ),
        kind="info",
    )
    return showdown_act, showdown_c, showdown_dataset, showdown_mlp


@app.cell
def _(showdown_act, showdown_c, showdown_dataset, showdown_mlp):
    _X, _y, _, _cnames = get_2d_dataset(showdown_dataset.value, noise=0.22)
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, test_size=0.30, stratify=_y, random_state=42
    )
    _C = float(showdown_c.value)
    _hidden = tuple(showdown_mlp.value)
    _act = showdown_act.value

    _clfs = [
        ("k-NN (k=7)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        (
            "Logistic Regression",
            make_pipeline(StandardScaler(), LogisticRegression(C=_C, max_iter=1000)),
        ),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Decision Tree (depth=5)", DecisionTreeClassifier(max_depth=5, random_state=42)),
        (
            "Random Forest (35 trees)",
            RandomForestClassifier(n_estimators=35, max_depth=6, random_state=42),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42),
        ),
        ("RBF SVM", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=_C, gamma="scale"))),
        (
            f"MLP Neural Net {_hidden}",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=_hidden, activation=_act, max_iter=900, random_state=42
                ),
            ),
        ),
    ]

    _xx, _yy, _grid = make_mesh(_X, grid_size=75)
    _titles, _fitted = [], []
    for _name, _m in _clfs:
        _m.fit(_X_tr, _y_tr)
        _titles.append(f"{_name}<br><sup>Test Acc: {_m.score(_X_te, _y_te):.1%}</sup>")
        _fitted.append(_m)

    _fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.14,
    )
    for _idx, _m in enumerate(_fitted):
        add_boundary_panel(
            _fig,
            _m,
            _xx,
            _yy,
            _grid,
            _X_te,
            _y_te,
            _cnames,
            row=_idx // 4 + 1,
            col=_idx % 4 + 1,
            showlegend=(_idx == 0),
            marker_size=6,
            line_width=0.8,
        )

    style_fig(_fig, f"8 Supervised Classifier Families on {showdown_dataset.value}", height=620)
    _insight = mo.accordion(
        {
            "Why do Logistic Regression and Naive Bayes struggle on XOR / Circles?": mo.md(
                "- **Logistic Regression** finds a single straight hyperplane $w^T x + b = 0$—geometrically impossible to separate opposite checkerboard quadrants or concentric rings without polynomial/kernel features.\n"
                "- **Gaussian Naive Bayes** assumes each class is an axis-aligned Gaussian; opposite XOR quadrants cancel each other's class means around `(0, 0)`.\n"
                "- **RBF SVM** & **MLP Neural Networks** bend space implicitly (via kernels) or explicitly (via hidden-layer representations) to separate non-linear manifolds."
            )
        }
    )
    mo.vstack([mo.ui.plotly(_fig), _insight], gap=0.75)


@app.cell
def _():
    mo.md("""
    ---
    ## Lab 5 · Regression & Regularization (`Linear`, `Ridge L2`, `Lasso L1`)

    When fitting flexible polynomial features $1, x, x^2, \\dots, x^d$ to noisy data, **unregularized OLS regression** explodes its coefficients to memorize noise. **Ridge ($L_2$)** shrinks all weights smoothly toward zero, while **Lasso ($L_1$)** drives redundant weights to *exact zero* (automatic feature selection).
    """)


@app.cell
def _():
    reg_degree = mo.ui.slider(
        start=1, stop=12, step=1, value=9, show_value=True, label="Polynomial Degree"
    )
    reg_model = mo.ui.radio(
        options=["OLS (LinearRegression)", "Ridge (L2)", "Lasso (L1)"],
        value="Ridge (L2)",
        inline=True,
        label="Regression Model",
    )
    reg_alpha = mo.ui.dropdown(
        options={"0.0001": 1e-4, "0.001": 1e-3, "0.01": 1e-2, "0.1": 0.1, "1.0": 1.0, "10.0": 10.0},
        value="0.01",
        label="Regularization α",
    )
    reg_noise = mo.ui.slider(
        start=0.05, stop=0.50, step=0.05, value=0.20, show_value=True, label="Target Noise σ"
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Regression & Regularization Controls** — Compare high-degree OLS overfitting against Ridge & Lasso"
                ),
                mo.hstack(
                    [reg_degree, reg_model, reg_alpha, reg_noise],
                    justify="start",
                    wrap=True,
                    gap=1.0,
                ),
            ]
        ),
        kind="info",
    )
    return reg_alpha, reg_degree, reg_model, reg_noise


@app.cell
def _(reg_alpha, reg_degree, reg_model, reg_noise):
    _rng = np.random.default_rng(7)
    _n_tr, _n_te = 28, 40
    _X_tr = np.sort(_rng.uniform(0, 1, _n_tr))[:, None]
    _X_te = np.sort(_rng.uniform(0, 1, _n_te))[:, None]

    def _true_fn(x):
        return np.sin(1.6 * np.pi * x.ravel())

    _sigma = float(reg_noise.value)
    _y_tr = _true_fn(_X_tr) + _rng.normal(0, _sigma, _n_tr)
    _y_te = _true_fn(_X_te) + _rng.normal(0, _sigma, _n_te)

    _deg = int(reg_degree.value)
    _alpha = float(reg_alpha.value)
    if reg_model.value.startswith("OLS"):
        _est = LinearRegression()
    elif reg_model.value.startswith("Ridge"):
        _est = Ridge(alpha=_alpha)
    else:
        _est = Lasso(alpha=_alpha, max_iter=20000)

    _pipe = make_pipeline(
        PolynomialFeatures(degree=_deg, include_bias=False),
        StandardScaler(),
        _est,
    ).fit(_X_tr, _y_tr)
    _tr_r2 = float(_pipe.score(_X_tr, _y_tr))
    _te_r2 = float(_pipe.score(_X_te, _y_te))
    _te_rmse = float(np.sqrt(np.mean((_y_te - _pipe.predict(_X_te)) ** 2)))

    _x_plot = np.linspace(0, 1, 200)[:, None]
    _coefs = _pipe[-1].coef_
    _zero_coefs = int(np.sum(np.abs(_coefs) < 1e-6))

    _fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=[
            f"Fitted Curve (Degree {_deg}, {reg_model.value})",
            f"Learned Feature Weights w₁..w_{_deg} (Standardized)",
        ],
        horizontal_spacing=0.10,
    )
    _fig.add_trace(
        go.Scatter(
            x=_x_plot.ravel(),
            y=_true_fn(_x_plot),
            mode="lines",
            name="True sin(1.6πx)",
            line={"color": "#64748B", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_X_tr.ravel(),
            y=_y_tr,
            mode="markers",
            name="Train Samples",
            marker={"color": "#2563EB", "size": 8},
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_X_te.ravel(),
            y=_y_te,
            mode="markers",
            name="Test Samples",
            marker={"color": "#059669", "size": 7, "symbol": "diamond-open"},
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_x_plot.ravel(),
            y=_pipe.predict(_x_plot),
            mode="lines",
            name="Model Prediction",
            line={"color": "#EA580C", "width": 3},
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Bar(
            x=[f"x^{d}" for d in range(1, _deg + 1)],
            y=_coefs,
            marker_color=["#2563EB" if c >= 0 else "#DC2626" for c in _coefs],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig.update_xaxes(title_text="x", row=1, col=1)
    _fig.update_yaxes(title_text="y", range=[-2.2, 2.2], row=1, col=1)
    _fig.update_xaxes(title_text="Polynomial Term", row=1, col=2)
    _fig.update_yaxes(title_text="Coefficient Weight", row=1, col=2)
    style_fig(_fig, "Polynomial Regression & Weight Regularization", height=430)

    _stats = mo.hstack(
        [
            mo.stat(f"{_tr_r2:.3f}", label="Train R²", bordered=True),
            mo.stat(f"{_te_r2:.3f}", label="Test R²", bordered=True),
            mo.stat(f"{_te_rmse:.3f}", label="Test RMSE", bordered=True),
            mo.stat(
                f"{_zero_coefs} / {_deg}",
                label="Zeroed Weights",
                caption="Sparsity (L1)",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=0.8,
    )
    mo.vstack([_stats, mo.ui.plotly(_fig)], gap=0.75)


@app.cell
def _():
    _rows = [
        {
            "Algorithm": "k-Nearest Neighbors (k-NN)",
            "Tasks": "Class / Reg",
            "Scaling?": "Yes",
            "Strengths": "Simple baseline, non-parametric, zero training time",
            "Weaknesses": "Slow prediction on large N, struggles in high dimensions",
            "Key Hyperparameters": "n_neighbors, weights, metric",
        },
        {
            "Algorithm": "Linear Models (Ridge / Lasso / LogReg)",
            "Tasks": "Class / Reg",
            "Scaling?": "Yes",
            "Strengths": "Fast, interpretable weights, scales to huge sparse data",
            "Weaknesses": "Only linear boundaries without feature engineering",
            "Key Hyperparameters": "C (LogReg) / alpha (Ridge, Lasso)",
        },
        {
            "Algorithm": "Gaussian / Multinomial Naive Bayes",
            "Tasks": "Class",
            "Scaling?": "No",
            "Strengths": "Extremely fast, strong baseline for text & high-dim data",
            "Weaknesses": "Assumes conditional independence of features",
            "Key Hyperparameters": "alpha (smoothing), var_smoothing",
        },
        {
            "Algorithm": "Decision Trees",
            "Tasks": "Class / Reg",
            "Scaling?": "No",
            "Strengths": "Human-readable rules, handles mixed feature scales naturally",
            "Weaknesses": "High variance, prone to overfitting if unpruned",
            "Key Hyperparameters": "max_depth, min_samples_leaf, max_leaf_nodes",
        },
        {
            "Algorithm": "Random Forests",
            "Tasks": "Class / Reg",
            "Scaling?": "No",
            "Strengths": "Strong out-of-the-box accuracy, parallelizable, low tuning",
            "Weaknesses": "Larger memory footprint, less interpretable than 1 tree",
            "Key Hyperparameters": "n_estimators, max_depth, max_features",
        },
        {
            "Algorithm": "Gradient Boosted Trees (GBDT)",
            "Tasks": "Class / Reg",
            "Scaling?": "No",
            "Strengths": "State-of-the-art on tabular data, compact fast inference",
            "Weaknesses": "Sensitive to hyperparameters, sequential training",
            "Key Hyperparameters": "n_estimators, learning_rate, max_depth",
        },
        {
            "Algorithm": "Kernelized SVMs (SVC / SVR)",
            "Tasks": "Class / Reg",
            "Scaling?": "Yes (Critical)",
            "Strengths": "Powerful smooth nonlinear boundaries on medium-sized datasets",
            "Weaknesses": "Scales poorly >50k rows, sensitive to C, gamma, & scaling",
            "Key Hyperparameters": "kernel, C, gamma",
        },
        {
            "Algorithm": "Neural Networks (MLP)",
            "Tasks": "Class / Reg",
            "Scaling?": "Yes (Critical)",
            "Strengths": "Learns complex hierarchical nonlinear functions",
            "Weaknesses": "Needs tuning, more data, & careful regularization",
            "Key Hyperparameters": "hidden_layer_sizes, alpha, activation, learning_rate_init",
        },
    ]
    mo.vstack(
        [
            mo.md("---\n## Reference · Supervised Learning Algorithm Cheat Sheet"),
            mo.ui.table(_rows, selection=None, show_column_summaries=False),
        ],
        gap=0.5,
    )


if __name__ == "__main__":
    app.run()
