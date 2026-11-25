"""Generate high-DPI static preview visuals for README.md into assets/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer, load_digits, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
PALETTE = ["#2563EB", "#EA580C", "#059669", "#7C3AED"]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _base_layout(fig: go.Figure, title: str, width: int = 980, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color="#0F172A")),
        width=width,
        height=height,
        margin=dict(l=55, r=30, t=68, b=50),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A", size=12),
    )
    return fig


def build_supervised_preview() -> go.Figure:
    X, y = make_moons(n_samples=260, noise=0.22, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    pad = 0.25
    xx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 110)
    yy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 110)
    gx, gy = np.meshgrid(xx, yy)
    grid = np.c_[gx.ravel(), gy.ravel()]

    models = [
        ("k-NN (k=7)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        ("RBF SVM (Scaled)", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
        ),
    ]
    titles = []
    fitted = []
    for name, m in models:
        m.fit(X_tr, y_tr)
        titles.append(f"{name}<br><sup>Test Accuracy: {m.score(X_te, y_te):.1%}</sup>")
        fitted.append(m)

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.06)
    cscale = [[0.0, _rgba(PALETTE[0], 0.28)], [1.0, _rgba(PALETTE[1], 0.28)]]
    for col, m in enumerate(fitted, start=1):
        z = m.predict(grid).reshape(len(yy), len(xx))
        fig.add_trace(
            go.Heatmap(x=xx, y=yy, z=z, colorscale=cscale, showscale=False, hoverinfo="skip"),
            row=1,
            col=col,
        )
        for ci in (0, 1):
            mask = y_te == ci
            fig.add_trace(
                go.Scatter(
                    x=X_te[mask, 0],
                    y=X_te[mask, 1],
                    mode="markers",
                    name=f"Class {ci}",
                    showlegend=(col == 1),
                    marker=dict(color=PALETTE[ci], size=7, line=dict(width=1, color="white")),
                ),
                row=1,
                col=col,
            )
    fig = _base_layout(fig, "01 · Supervised Decision Boundaries on Nonlinear Moons", height=400)
    fig.update_layout(margin=dict(l=55, r=30, t=86, b=50))
    return fig


def build_clustering_preview() -> go.Figure:
    X, _ = make_moons(n_samples=220, noise=0.06, random_state=0)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(Xs)
    y_km = km.labels_
    y_agg = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Xs)
    y_db = DBSCAN(eps=0.35, min_samples=5).fit_predict(Xs)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "k-Means (Centroid Voronoi)",
            "Agglomerative (Ward Linkage)",
            "DBSCAN (Density-Based)",
        ],
        horizontal_spacing=0.06,
    )
    for col, (lbls, centers) in enumerate(
        [(y_km, km.cluster_centers_), (y_agg, None), (y_db, None)], start=1
    ):
        for cid in sorted(set(lbls)):
            mask = lbls == cid
            fig.add_trace(
                go.Scatter(
                    x=Xs[mask, 0],
                    y=Xs[mask, 1],
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color="#64748B" if cid == -1 else PALETTE[cid % len(PALETTE)],
                        size=7,
                        symbol="x" if cid == -1 else "circle",
                    ),
                ),
                row=1,
                col=col,
            )
        if centers is not None:
            fig.add_trace(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color="#0F172A", size=13, symbol="star", line=dict(width=1.5, color="white")
                    ),
                ),
                row=1,
                col=col,
            )
    return _base_layout(fig, "02 · Clustering Showdown: Centroids vs. Linkage vs. Density")


def build_evaluation_preview() -> go.Figure:
    digits = load_digits()
    y = (digits.target == 9).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(digits.data, y, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500)).fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)[:, 1]
    pred = (probs >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_te, probs)
    auc = roc_auc_score(y_te, probs)
    pr_p, pr_r, _ = precision_recall_curve(y_te, probs)
    ap = average_precision_score(y_te, probs)

    tp = int(np.sum((pred == 1) & (y_te == 1)))
    fp = int(np.sum((pred == 1) & (y_te == 0)))
    fn = int(np.sum((pred == 0) & (y_te == 1)))
    tn = int(np.sum((pred == 0) & (y_te == 0)))
    cur_fpr = fp / (fp + tn)
    cur_rec = tp / (tp + fn)
    cur_prec = tp / (tp + fp)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"ROC Curve (AUC = {auc:.3f})", f"Precision–Recall Curve (AP = {ap:.3f})"],
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name="LogReg ROC", line=dict(color="#2563EB", width=3)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[cur_fpr],
            y=[cur_rec],
            mode="markers",
            name="Threshold = 0.50",
            marker=dict(
                color="#FACC15", size=14, symbol="star", line=dict(width=1.5, color="#0F172A")
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pr_r, y=pr_p, mode="lines", name="LogReg PR", line=dict(color="#EA580C", width=3)
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[cur_rec],
            y=[cur_prec],
            mode="markers",
            showlegend=False,
            marker=dict(
                color="#FACC15", size=14, symbol="star", line=dict(width=1.5, color="#0F172A")
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="Recall (TPR)", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    return _base_layout(
        fig, "03 · Imbalanced Classification Curves (Digits == 9)", width=980, height=390
    )


def build_pipeline_preview() -> go.Figure:
    cancer = load_breast_cancer()
    X_tr, _, y_tr, _ = train_test_split(cancer.data, cancer.target, random_state=0)
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svc", SVC())])
    c_vals = [0.001, 0.01, 0.1, 1, 10, 100]
    g_vals = [0.001, 0.01, 0.1, 1, 10, 100]
    grid = GridSearchCV(
        pipe,
        param_grid={"svc__C": c_vals, "svc__gamma": g_vals},
        cv=StratifiedKFold(n_splits=5),
    ).fit(X_tr, y_tr)
    means = grid.cv_results_["mean_test_score"].reshape(len(c_vals), len(g_vals))

    best_r = c_vals.index(grid.best_params_["svc__C"])
    best_c = g_vals.index(grid.best_params_["svc__gamma"])
    text_mat = [
        [
            f"<b>★ {means[r, c]:.3f}</b>" if (r == best_r and c == best_c) else f"{means[r, c]:.3f}"
            for c in range(len(g_vals))
        ]
        for r in range(len(c_vals))
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=means,
            x=[str(g) for g in g_vals],
            y=[str(c) for c in c_vals],
            colorscale="Viridis",
            text=text_mat,
            texttemplate="%{text}",
            colorbar=dict(title="Mean CV"),
        )
    )
    fig.add_shape(
        type="rect",
        x0=best_c - 0.48,
        x1=best_c + 0.48,
        y0=best_r - 0.48,
        y1=best_r + 0.48,
        line=dict(color="#0F172A", width=3),
    )
    fig.update_xaxes(title_text="svc__gamma", type="category")
    fig.update_yaxes(title_text="svc__C", type="category")
    return _base_layout(
        fig,
        "04 · Pipeline GridSearchCV Surface (MinMaxScaler → SVC on Breast Cancer)",
        width=980,
        height=390,
    )


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("01_decision_boundaries.png", build_supervised_preview()),
        ("02_clustering_comparison.png", build_clustering_preview()),
        ("03_evaluation_curves.png", build_evaluation_preview()),
        ("04_gridsearch_heatmap.png", build_pipeline_preview()),
    ]
    for filename, fig in outputs:
        out_path = ASSETS_DIR / filename
        fig.write_image(str(out_path), scale=2)
        print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
