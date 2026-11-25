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
app = marimo.App(
    width="medium",
    app_title="03 · Model Evaluation & Refinement",
)


@app.cell
def _():
    import functools

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        KFold,
        ShuffleSplit,
        StratifiedKFold,
        cross_val_score,
        learning_curve,
        train_test_split,
        validation_curve,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return (
        DecisionTreeClassifier,
        DummyClassifier,
        GridSearchCV,
        KFold,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        ShuffleSplit,
        StandardScaler,
        StratifiedKFold,
        average_precision_score,
        confusion_matrix,
        cross_val_score,
        f1_score,
        functools,
        go,
        learning_curve,
        load_breast_cancer,
        load_digits,
        load_iris,
        make_pipeline,
        make_subplots,
        mo,
        np,
        pd,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        train_test_split,
        validation_curve,
    )


@app.cell
def _(mo):
    mo.md("""
    # 03 · Model Evaluation, Cross-Validation & Imbalanced Metrics

    A single `train_test_split` score can mislead you—both from lucky/unlucky splits and from class imbalance. This notebook turns abstract evaluation concepts into live, visual diagnostics.

    ### What You'll Explore
    1. **Lab 1 — Cross-Validation & Learning Curves:** See exactly how each fold splits sorted **Iris** data—and watch unshuffled 3-fold `KFold` collapse to **0.0% accuracy**.
    2. **Lab 2 — Validation Curves & `GridSearchCV`:** Diagnose bias vs. variance across hyperparameters and guard against test-set leakage using the 3-way split.
    3. **Lab 3 — Imbalanced Datasets & Threshold Tuning (`digits == 9`):** Escape the Accuracy Paradox (`DummyClassifier` at 90%!) with interactive confusion matrices, synced ROC/PR curves, and a misclassified digit gallery.
    4. **Reference — Evaluation & Metrics Cheat Sheet:** Choosing the right splitter and metric for any ML problem.
    """)


@app.function
def style_fig(fig, title: str, height: int = 430):
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


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 1 · Cross-Validation Visualizer (`Iris`) & Learning Curves

    The **Iris** dataset is sorted by species: rows `0–49` are *Setosa*, `50–99` are *Versicolor*, and `100–149` are *Virginica*. Watch what happens when you pick **`KFold (unshuffled)`** with `n_splits = 3` versus **`StratifiedKFold`**!
    """)


@app.cell
def _(mo):
    cv_strategy = mo.ui.radio(
        options=[
            "KFold (unshuffled)",
            "KFold (shuffled)",
            "StratifiedKFold",
            "ShuffleSplit",
        ],
        value="KFold (unshuffled)",
        inline=True,
        label="Cross-Validation Strategy",
    )
    cv_splits = mo.ui.slider(
        start=2, stop=10, step=1, value=3, show_value=True, label="Number of Folds (n_splits)"
    )
    lc_model = mo.ui.radio(
        options=["LogisticRegression", "DecisionTreeClassifier", "SVC (RBF)"],
        value="LogisticRegression",
        inline=True,
        label="Learning Curve Estimator (Breast Cancer)",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Cross-Validation Controls** — Notice the 0% collapse on unshuffled `n_splits=3`, then switch to `StratifiedKFold`"
                ),
                mo.hstack([cv_strategy, cv_splits], justify="start", wrap=True, gap=1.2),
                lc_model,
            ]
        ),
        kind="info",
    )
    return cv_splits, cv_strategy, lc_model


@app.cell
def _(
    DecisionTreeClassifier,
    KFold,
    LogisticRegression,
    SVC,
    ShuffleSplit,
    StandardScaler,
    StratifiedKFold,
    cross_val_score,
    cv_splits,
    cv_strategy,
    go,
    lc_model,
    learning_curve,
    load_breast_cancer,
    load_iris,
    make_pipeline,
    make_subplots,
    mo,
    np,
):
    _iris = load_iris()
    _k = int(cv_splits.value)
    _strat = cv_strategy.value

    if _strat == "KFold (unshuffled)":
        _cv = KFold(n_splits=_k, shuffle=False)
    elif _strat == "KFold (shuffled)":
        _cv = KFold(n_splits=_k, shuffle=True, random_state=42)
    elif _strat == "StratifiedKFold":
        _cv = StratifiedKFold(n_splits=_k)
    else:
        _cv = ShuffleSplit(n_splits=_k, test_size=0.25, random_state=42)

    _pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    _scores = cross_val_score(_pipe, _iris.data, _iris.target, cv=_cv)

    # Build fold assignment matrix: rows = [Fold 1..k], plus separate row for true class labels
    _n = len(_iris.target)
    _split_mat = np.zeros((_k, _n))
    for _fi, (_, _val_idx) in enumerate(_cv.split(_iris.data, _iris.target)):
        _split_mat[_fi, _val_idx] = 1

    _fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.22, 0.78],
        column_widths=[0.62, 0.38],
        specs=[[{"type": "heatmap"}, {"rowspan": 2, "type": "bar"}], [{"type": "heatmap"}, None]],
        subplot_titles=[
            "True Iris Class Order (Rows 0–149)",
            f"Per-Fold Accuracy (Mean {_scores.mean():.1%})",
            f"{_strat} Split Matrix (Blue = Train · Orange = Validation)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    _fig.add_trace(
        go.Heatmap(
            z=[_iris.target],
            y=["True Species"],
            colorscale=[[0.0, "#2563EB"], [0.5, "#059669"], [1.0, "#7C3AED"]],
            zmin=0,
            zmax=2,
            showscale=False,
            hovertemplate="Sample %{x} · Class %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Heatmap(
            z=_split_mat,
            y=[f"Fold {i + 1}" for i in range(_k)],
            colorscale=[[0.0, "#DBEAFE"], [1.0, "#EA580C"]],
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate="Sample %{x} · %{y} · Split=%{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _fig.add_trace(
        go.Bar(
            x=[f"Fold {i + 1}" for i in range(_k)],
            y=_scores,
            marker_color=["#DC2626" if s < 0.5 else "#2563EB" for s in _scores],
            text=[f"{s:.1%}" for s in _scores],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig.add_hline(
        y=float(_scores.mean()),
        line_dash="dash",
        line_color="#059669",
        annotation_text=f"Mean {_scores.mean():.1%}",
        row=1,
        col=2,
    )
    _fig.update_xaxes(showticklabels=False, row=1, col=1)
    _fig.update_xaxes(title_text="Sample Index (0–149)", row=2, col=1)
    _fig.update_yaxes(title_text="Accuracy", range=[0, 1.12], row=1, col=2)
    style_fig(_fig, f"Cross-Validation Anatomy: {_strat} (n_splits={_k})", height=430)

    # Learning curve on Breast Cancer
    _cancer = load_breast_cancer()
    if lc_model.value == "LogisticRegression":
        _lc_est = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    elif lc_model.value == "DecisionTreeClassifier":
        _lc_est = DecisionTreeClassifier(random_state=42)
    else:
        _lc_est = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))

    _sizes, _tr_s, _cv_s = learning_curve(
        _lc_est,
        _cancer.data,
        _cancer.target,
        cv=StratifiedKFold(n_splits=5),
        train_sizes=np.linspace(0.15, 1.0, 8),
    )
    _tr_mean, _tr_std = _tr_s.mean(axis=1), _tr_s.std(axis=1)
    _cv_mean, _cv_std = _cv_s.mean(axis=1), _cv_s.std(axis=1)

    _lc_fig = go.Figure()
    _lc_fig.add_trace(
        go.Scatter(
            x=np.concatenate([_sizes, _sizes[::-1]]),
            y=np.concatenate([_cv_mean + _cv_std, (_cv_mean - _cv_std)[::-1]]),
            fill="toself",
            fillcolor="rgba(234, 88, 12, 0.14)",
            line=dict(color="rgba(0,0,0,0)"),
            name="CV ±1 std",
            hoverinfo="skip",
        )
    )
    _lc_fig.add_trace(
        go.Scatter(
            x=_sizes,
            y=_tr_mean,
            mode="lines+markers",
            name="Train Score",
            line=dict(color="#2563EB", width=2.5),
        )
    )
    _lc_fig.add_trace(
        go.Scatter(
            x=_sizes,
            y=_cv_mean,
            mode="lines+markers",
            name="5-Fold CV Score",
            line=dict(color="#EA580C", width=2.5),
        )
    )
    _lc_fig.update_xaxes(title_text="Training Set Size (Samples)")
    _lc_fig.update_yaxes(title_text="Accuracy", range=[0.75, 1.03])
    style_fig(_lc_fig, f"Learning Curve on Breast Cancer ({lc_model.value})", height=370)

    _stats = mo.hstack(
        [
            mo.stat(f"{_scores.mean():.1%}", label="Mean CV Accuracy", bordered=True),
            mo.stat(f"±{_scores.std():.1%}", label="Fold Std Dev", bordered=True),
            mo.stat(
                f"{_scores.min():.1%} – {_scores.max():.1%}", label="Min – Max Fold", bordered=True
            ),
        ],
        widths="equal",
        gap=0.8,
    )

    if _strat == "KFold (unshuffled)" and _k == 3:
        _msg = mo.callout(
            mo.md(
                "**Zero-Accuracy Trap Observed!** Because Iris is sorted by class (`50` per species), unshuffled `3`-fold KFold trains on two species and validates on the third *unseen* species every fold—getting **0.0% accuracy**! Always use `StratifiedKFold` (scikit-learn's default for classifiers) or shuffle first."
            ),
            kind="danger",
        )
    else:
        _msg = mo.callout(
            mo.md(
                f"With `{_strat}`, every validation fold gets representative samples across classes. Compare the learning curve below: an unpruned `DecisionTreeClassifier` stays stuck at `100%` train accuracy with a wide CV gap (high variance)."
            ),
            kind="success",
        )

    mo.vstack([_stats, mo.ui.plotly(_fig), _msg, mo.ui.plotly(_lc_fig)], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 2 · Validation Curves, Grid Search & The 3-Way Split

    Tuning hyperparameters directly against your test set leaks test information into model selection. The safe workflow splits data into **Train**, **Validation (or CV folds)** for choosing hyperparameters, and an **untouched Held-Out Test Set** evaluated once at the end.
    """)


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    StandardScaler,
    StratifiedKFold,
    functools,
    load_breast_cancer,
    make_pipeline,
    mo,
    np,
    train_test_split,
    validation_curve,
):
    @functools.lru_cache(maxsize=6)
    def run_validation_and_grid(est_name: str, cv_folds: int):
        _cancer = load_breast_cancer()
        _X_tr, _X_te, _y_tr, _y_te = train_test_split(
            _cancer.data, _cancer.target, stratify=_cancer.target, test_size=0.25, random_state=42
        )
        _cv = StratifiedKFold(n_splits=cv_folds)

        if est_name == "SVC (RBF)":
            _pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
            _param_name = "svc__C"
            _param_range = np.logspace(-3, 3, 7)
            _x_labels = [f"{v:g}" for v in _param_range]
        elif est_name == "LogisticRegression":
            _pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))
            _param_name = "logisticregression__C"
            _param_range = np.logspace(-3, 3, 7)
            _x_labels = [f"{v:g}" for v in _param_range]
        else:
            _pipe = RandomForestClassifier(n_estimators=50, random_state=42)
            _param_name = "max_depth"
            _param_range = np.array([1, 2, 3, 4, 6, 8, 12])
            _x_labels = [str(v) for v in _param_range]

        _tr_s, _cv_s = validation_curve(
            _pipe,
            _X_tr,
            _y_tr,
            param_name=_param_name,
            param_range=_param_range,
            cv=_cv,
        )

        # Cached 2D SVC GridSearch landscape on Train+Val
        _svc_pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
        _c_vals = [0.01, 0.1, 1.0, 10.0, 100.0]
        _g_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
        _grid = GridSearchCV(
            _svc_pipe,
            param_grid={"svc__C": _c_vals, "svc__gamma": _g_vals},
            cv=_cv,
        ).fit(_X_tr, _y_tr)
        _grid_scores = _grid.cv_results_["mean_test_score"].reshape(len(_c_vals), len(_g_vals))
        _test_score = float(_grid.score(_X_te, _y_te))

        return (
            _param_name,
            _x_labels,
            _tr_s.mean(axis=1),
            _tr_s.std(axis=1),
            _cv_s.mean(axis=1),
            _cv_s.std(axis=1),
            _c_vals,
            _g_vals,
            _grid_scores,
            float(_grid.best_score_),
            _test_score,
            _grid.best_params_,
        )

    vc_est = mo.ui.radio(
        options=["SVC (RBF)", "LogisticRegression", "RandomForestClassifier"],
        value="SVC (RBF)",
        inline=True,
        label="Validation Curve Estimator",
    )
    vc_folds = mo.ui.radio(
        options={"3-Fold CV": 3, "5-Fold CV": 5, "10-Fold CV": 10},
        value="5-Fold CV",
        inline=True,
        label="CV Folds",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Hyperparameter Tuning Controls** — Compare validation curve sweet spots and 2D grid search surfaces"
                ),
                mo.hstack([vc_est, vc_folds], justify="start", wrap=True, gap=1.2),
            ]
        ),
        kind="info",
    )
    return run_validation_and_grid, vc_est, vc_folds


@app.cell
def _(go, make_subplots, mo, run_validation_and_grid, vc_est, vc_folds):
    (
        _pname,
        _x_labels,
        _tr_m,
        _tr_s,
        _cv_m,
        _cv_s,
        _c_vals,
        _g_vals,
        _grid_scores,
        _best_cv,
        _test_score,
        _best_params,
    ) = run_validation_and_grid(vc_est.value, int(vc_folds.value))

    _fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.48, 0.52],
        subplot_titles=[
            f"Validation Curve: {vc_est.value} ({_pname})",
            "2D GridSearchCV Surface (SVC: C × gamma)",
        ],
        horizontal_spacing=0.12,
    )
    _fig.add_trace(
        go.Scatter(
            x=_x_labels,
            y=_tr_m,
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color="#2563EB", width=2.5),
            error_y=dict(type="data", array=_tr_s, visible=True, thickness=1.2),
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_x_labels,
            y=_cv_m,
            mode="lines+markers",
            name="CV Accuracy",
            line=dict(color="#EA580C", width=2.5),
            error_y=dict(type="data", array=_cv_s, visible=True, thickness=1.2),
        ),
        row=1,
        col=1,
    )
    _fig.update_xaxes(title_text=_pname, row=1, col=1)
    _fig.update_yaxes(title_text="Accuracy", range=[0.55, 1.03], row=1, col=1)

    _c_lbls = [f"{c:g}" for c in _c_vals]
    _g_lbls = [f"{g:g}" for g in _g_vals]
    _fig.add_trace(
        go.Heatmap(
            z=_grid_scores,
            x=_g_lbls,
            y=_c_lbls,
            colorscale="Viridis",
            text=[[f"{v:.3f}" for v in row] for row in _grid_scores],
            texttemplate="%{text}",
            colorbar=dict(title="Mean CV"),
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Scatter(
            x=[f"{_best_params['svc__gamma']:g}"],
            y=[f"{_best_params['svc__C']:g}"],
            mode="markers",
            name="Best Grid Point",
            marker=dict(
                color="#FACC15",
                size=16,
                symbol="star",
                line=dict(width=1.5, color="#0F172A"),
            ),
        ),
        row=1,
        col=2,
    )
    _fig.update_xaxes(title_text="svc__gamma", type="category", row=1, col=2)
    _fig.update_yaxes(title_text="svc__C", type="category", row=1, col=2)
    style_fig(_fig, "Validation Curve & Grid Search Hyperparameter Landscape", height=430)

    _stats = mo.hstack(
        [
            mo.stat(f"{_best_cv:.1%}", label="Best GridSearch CV Score", bordered=True),
            mo.stat(f"{_test_score:.1%}", label="Untouched Final Test Score", bordered=True),
            mo.stat(
                f"C={_best_params['svc__C']:g}, γ={_best_params['svc__gamma']:g}",
                label="Winning SVC Parameters",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=0.8,
    )
    mo.vstack([_stats, mo.ui.plotly(_fig)], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 3 · Imbalanced Datasets, Accuracy Paradox & Threshold Tuning (`digits == 9`)

    In `load_digits`, only **10%** of images are the digit `9` and **90%** are `Not 9`. A naive `DummyClassifier(strategy="most_frequent")` predicts `Not 9` every single time and achieves **90% accuracy**—yet detects **0%** of actual nines!

    Drag the **Decision Threshold** slider below to trade off **Precision** vs. **Recall** and inspect the actual misclassified `8×8` digit images.
    """)


@app.cell
def _(mo):
    imb_threshold = mo.ui.slider(
        start=0.05,
        stop=0.95,
        step=0.05,
        value=0.50,
        show_value=True,
        label="Decision Threshold P(y=9)",
    )
    imb_c = mo.ui.dropdown(
        options={"0.01": 0.01, "0.1": 0.1, "1.0": 1.0, "10.0": 10.0},
        value="0.1",
        label="LogisticRegression C",
    )
    imb_balanced = mo.ui.switch(value=False, label="class_weight='balanced'")
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Imbalanced Classification Controls** — Lower the threshold to catch more `9`s (higher Recall) or raise it to avoid false alarms (higher Precision)"
                ),
                mo.hstack(
                    [imb_threshold, imb_c, imb_balanced], justify="start", wrap=True, gap=1.2
                ),
            ]
        ),
        kind="info",
    )
    return imb_balanced, imb_c, imb_threshold


@app.cell
def _(
    DummyClassifier,
    LogisticRegression,
    StandardScaler,
    average_precision_score,
    confusion_matrix,
    f1_score,
    go,
    imb_balanced,
    imb_c,
    imb_threshold,
    load_digits,
    make_pipeline,
    make_subplots,
    mo,
    np,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    train_test_split,
):
    _digits = load_digits()
    _y_bin = (_digits.target == 9).astype(int)
    _X_tr, _X_te, _y_tr, _y_te, _, _orig_te = train_test_split(
        _digits.data, _y_bin, _digits.target, random_state=0
    )

    _dummy = DummyClassifier(strategy="most_frequent").fit(_X_tr, _y_tr)
    _dum_pred = _dummy.predict(_X_te)

    _cw = "balanced" if imb_balanced.value else None
    _pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(imb_c.value), class_weight=_cw, max_iter=1500),
    ).fit(_X_tr, _y_tr)
    _probs = _pipe.predict_proba(_X_te)[:, 1]
    _thr = float(imb_threshold.value)
    _pred = (_probs >= _thr).astype(int)

    _acc = float(np.mean(_pred == _y_te))
    _prec = float(precision_score(_y_te, _pred, zero_division=0))
    _rec = float(recall_score(_y_te, _pred, zero_division=0))
    _f1 = float(f1_score(_y_te, _pred, zero_division=0))

    _cm = confusion_matrix(_y_te, _pred)
    _tn, _fp, _fn, _tp = _cm.ravel()
    _spec = float(_tn / (_tn + _fp)) if (_tn + _fp) > 0 else 0.0

    # Dummy vs LogisticRegression metric bars + Confusion Matrix
    _dum_acc = float(np.mean(_dum_pred == _y_te))
    _dum_prec = float(precision_score(_y_te, _dum_pred, zero_division=0))
    _dum_rec = float(recall_score(_y_te, _dum_pred, zero_division=0))
    _dum_f1 = float(f1_score(_y_te, _dum_pred, zero_division=0))

    _fig_top = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=[
            "Accuracy Paradox: Dummy Majority vs. LogisticRegression",
            f"Confusion Matrix at Threshold = {_thr:.2f}",
        ],
        horizontal_spacing=0.14,
    )
    _metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    _fig_top.add_trace(
        go.Bar(
            name="Dummy (Always 'Not 9')",
            x=_metrics,
            y=[_dum_acc, _dum_prec, _dum_rec, _dum_f1],
            marker_color="#94A3B8",
            text=[f"{v:.2f}" for v in [_dum_acc, _dum_prec, _dum_rec, _dum_f1]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    _fig_top.add_trace(
        go.Bar(
            name=f"LogReg (thr={_thr:.2f})",
            x=_metrics,
            y=[_acc, _prec, _rec, _f1],
            marker_color="#2563EB",
            text=[f"{v:.2f}" for v in [_acc, _prec, _rec, _f1]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    _fig_top.update_yaxes(range=[0, 1.15], title_text="Score", row=1, col=1)

    _cm_labels = [
        [f"TN = {_tn}<br>({_tn / (_tn + _fp):.1%})", f"FP = {_fp}<br>({_fp / (_tn + _fp):.1%})"],
        [f"FN = {_fn}<br>({_fn / (_fn + _tp):.1%})", f"TP = {_tp}<br>({_tp / (_fn + _tp):.1%})"],
    ]
    _fig_top.add_trace(
        go.Heatmap(
            z=_cm,
            x=["Pred: Not 9", "Pred: Is 9"],
            y=["Actual: Not 9", "Actual: Is 9"],
            colorscale="Blues",
            text=_cm_labels,
            texttemplate="%{text}",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    _fig_top.update_yaxes(autorange="reversed", row=1, col=2)
    _fig_top.update_layout(barmode="group")
    style_fig(_fig_top, "Imbalanced Evaluation: Why Accuracy Alone Hides Failure", height=390)

    # Synced ROC and Precision-Recall Curves
    _fpr, _tpr, _ = roc_curve(_y_te, _probs)
    _roc_auc = float(roc_auc_score(_y_te, _probs))
    _pr_p, _pr_r, _ = precision_recall_curve(_y_te, _probs)
    _ap = float(average_precision_score(_y_te, _probs))

    _cur_fpr = float(_fp / (_fp + _tn)) if (_fp + _tn) > 0 else 0.0
    _cur_tpr = _rec

    _fig_curves = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"ROC Curve (AUC = {_roc_auc:.3f})",
            f"Precision–Recall Curve (AP = {_ap:.3f})",
        ],
        horizontal_spacing=0.10,
    )
    _fig_curves.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")
        ),
        row=1,
        col=1,
    )
    _fig_curves.add_trace(
        go.Scatter(x=_fpr, y=_tpr, mode="lines", name="ROC", line=dict(color="#2563EB", width=2.5)),
        row=1,
        col=1,
    )
    _fig_curves.add_trace(
        go.Scatter(
            x=[_cur_fpr],
            y=[_cur_tpr],
            mode="markers",
            name=f"Threshold {_thr:.2f}",
            marker=dict(
                color="#FACC15", size=14, symbol="star", line=dict(width=1.5, color="#0F172A")
            ),
        ),
        row=1,
        col=1,
    )
    _fig_curves.add_trace(
        go.Scatter(
            x=_pr_r, y=_pr_p, mode="lines", name="PR Curve", line=dict(color="#EA580C", width=2.5)
        ),
        row=1,
        col=2,
    )
    _fig_curves.add_trace(
        go.Scatter(
            x=[_rec],
            y=[_prec],
            mode="markers",
            showlegend=False,
            marker=dict(
                color="#FACC15", size=14, symbol="star", line=dict(width=1.5, color="#0F172A")
            ),
        ),
        row=1,
        col=2,
    )
    _fig_curves.update_xaxes(title_text="False Positive Rate", range=[-0.02, 1.02], row=1, col=1)
    _fig_curves.update_yaxes(
        title_text="True Positive Rate (Recall)", range=[-0.02, 1.04], row=1, col=1
    )
    _fig_curves.update_xaxes(title_text="Recall", range=[-0.02, 1.02], row=1, col=2)
    _fig_curves.update_yaxes(title_text="Precision", range=[-0.02, 1.04], row=1, col=2)
    style_fig(_fig_curves, "Operating Point on ROC & Precision–Recall Curves", height=390)

    # Error Analysis Gallery: up to 4 False Positives + up to 4 False Negatives
    _fp_idx = np.where((_pred == 1) & (_y_te == 0))[0]
    _fn_idx = np.where((_pred == 0) & (_y_te == 1))[0]
    # Rank most confident errors first
    _fp_sorted = _fp_idx[np.argsort(-_probs[_fp_idx])][:4]
    _fn_sorted = _fn_idx[np.argsort(_probs[_fn_idx])][:4]

    _gallery_items = [("FP", int(i)) for i in _fp_sorted] + [("FN", int(i)) for i in _fn_sorted]
    if _gallery_items:
        _ncols = len(_gallery_items)
        _gal_titles = [
            f"{kind}: True={_orig_te[idx]}<br>P(9)={_probs[idx]:.2f}"
            for kind, idx in _gallery_items
        ]
        _fig_gal = make_subplots(
            rows=1, cols=_ncols, subplot_titles=_gal_titles, horizontal_spacing=0.03
        )
        for _ci, (_kind, _di) in enumerate(_gallery_items, start=1):
            _fig_gal.add_trace(
                go.Heatmap(
                    z=_X_te[_di].reshape(8, 8),
                    colorscale="Oranges" if _kind == "FP" else "Blues",
                    showscale=False,
                ),
                row=1,
                col=_ci,
            )
            _fig_gal.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=_ci)
            _fig_gal.update_xaxes(showticklabels=False, row=1, col=_ci)
        style_fig(
            _fig_gal,
            "Error Analysis: Hardest False Positives (Orange) & False Negatives (Blue)",
            height=230,
        )
        _gal_widget = mo.ui.plotly(_fig_gal)
    else:
        _gal_widget = mo.callout(
            mo.md("Zero misclassifications on the test set at this configuration!"), kind="success"
        )

    _stats = mo.hstack(
        [
            mo.stat(f"{_acc:.1%}", label="Accuracy", bordered=True),
            mo.stat(f"{_prec:.1%}", label="Precision", caption="TP / (TP + FP)", bordered=True),
            mo.stat(f"{_rec:.1%}", label="Recall", caption="TP / (TP + FN)", bordered=True),
            mo.stat(f"{_f1:.3f}", label="F1 Score", bordered=True),
            mo.stat(f"{_spec:.1%}", label="Specificity", caption="TN / (TN + FP)", bordered=True),
        ],
        widths="equal",
        gap=0.7,
    )

    mo.vstack(
        [_stats, mo.ui.plotly(_fig_top), mo.ui.plotly(_fig_curves), _gal_widget],
        gap=0.75,
    )


@app.cell
def _(mo, pd):
    mo.md("""
    ---
    ## Reference · Evaluation Strategy & Metric Selection Guide
    """)
    _guide = pd.DataFrame(
        [
            {
                "Scenario": "Balanced Multi-Class Classification",
                "Recommended Splitter": "StratifiedKFold",
                "Primary Metrics": "Accuracy, Macro-F1, Confusion Matrix",
                "Watch Out For": "Sorted target rows (never leave shuffle=False with plain KFold)",
            },
            {
                "Scenario": "Rare-Event / Imbalanced Binary (Fraud, Disease)",
                "Recommended Splitter": "StratifiedKFold",
                "Primary Metrics": "PR-AUC (Average Precision), F1, Recall@K",
                "Watch Out For": "Accuracy Paradox—a constant majority baseline can look 95%+ accurate",
            },
            {
                "Scenario": "Ranking / Probability Calibration",
                "Recommended Splitter": "StratifiedKFold",
                "Primary Metrics": "ROC-AUC, Log Loss, Brier Score",
                "Watch Out For": "Tuning decision thresholds on the final test set (use CV/val instead)",
            },
            {
                "Scenario": "Grouped Patients / Repeated Subjects",
                "Recommended Splitter": "GroupKFold / StratifiedGroupKFold",
                "Primary Metrics": "Task metric computed strictly across unseen groups",
                "Watch Out For": "Splitting same patient across train & test inflates scores via leakage",
            },
            {
                "Scenario": "Regression (Continuous Targets)",
                "Recommended Splitter": "KFold(shuffle=True)",
                "Primary Metrics": "RMSE, MAE, R²",
                "Watch Out For": "Heavy outliers distort RMSE/R²—compare against MAE & residual plots",
            },
        ]
    )
    mo.ui.table(_guide, selection=None, show_column_summaries=False)


if __name__ == "__main__":
    app.run()
