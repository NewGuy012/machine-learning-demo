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
app = marimo.App(width="medium", app_title="04 · Building Pipelines")


@app.cell
def _():
    import functools
    import time

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
    from sklearn.svm import SVC

    return (
        ColumnTransformer,
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegression,
        MinMaxScaler,
        OneHotEncoder,
        PCA,
        Pipeline,
        RandomForestClassifier,
        RobustScaler,
        SVC,
        SelectKBest,
        SelectPercentile,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        cross_val_score,
        f_classif,
        functools,
        go,
        load_breast_cancer,
        make_pipeline,
        mo,
        np,
        pd,
        time,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # 04 · Building Machine Learning Pipelines

    Real-world ML workflows chain together imputation, encoding, scaling, dimensionality reduction, and estimation. **`Pipeline`** bundles those steps into one scikit-learn estimator—eliminating boilerplate and **preventing subtle data leakage** during cross-validation and grid search.

    ### What You'll Explore
    1. **Lab 1 — Interactive Pipeline Builder:** Assemble custom steps (`Scaler → Reducer → Classifier`), compare `Pipeline` vs. `make_pipeline` syntax, and inspect scikit-learn's live HTML architecture diagram.
    2. **Lab 2 — Data Leakage Simulator:** Watch feature selection *outside* CV score **90%+ on pure random noise**, and see `Pipeline` restore honest generalization scores.
    3. **Lab 3 — Pipelines in `GridSearchCV`:** Visualize the `svc__C × svc__gamma` hyperparameter landscape and search across *multiple scalers & model families* in one grid.
    4. **Lab 4 — Mixed Tabular Workflows with `ColumnTransformer`:** Route numeric and categorical columns through separate preprocessing branches inside a single pipeline.
    5. **Reference — Pipeline Best Practices Cheat Sheet.**
    """)


@app.function
def style_fig(fig, title: str, height: int = 420):
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
    ## Lab 1 · Interactive Pipeline Builder (`Pipeline` vs. `make_pipeline`)

    Choose each step of your Breast Cancer pipeline below. Scikit-learn renders an **interactive block diagram** (click any step inside the diagram to inspect its hyperparameters) alongside side-by-side explicit `Pipeline` and shorthand `make_pipeline` Python code.
    """)


@app.cell
def _(mo):
    pipe_scaler = mo.ui.dropdown(
        options=["None (passthrough)", "MinMaxScaler", "StandardScaler", "RobustScaler"],
        value="MinMaxScaler",
        label="Step 1 · Scaler",
    )
    pipe_reducer = mo.ui.dropdown(
        options=["None", "PCA (n=10)", "SelectKBest (k=15)"],
        value="None",
        label="Step 2 · Feature Step (Optional)",
    )
    pipe_estimator = mo.ui.dropdown(
        options=[
            "SVC (C=100)",
            "LogisticRegression",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
        ],
        value="SVC (C=100)",
        label="Step 3 · Estimator",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md("**Assemble Your Breast Cancer Pipeline**"),
                mo.hstack(
                    [pipe_scaler, pipe_reducer, pipe_estimator], justify="start", wrap=True, gap=1.0
                ),
            ]
        ),
        kind="info",
    )
    return pipe_estimator, pipe_reducer, pipe_scaler


@app.cell
def _(
    GradientBoostingClassifier,
    LogisticRegression,
    MinMaxScaler,
    PCA,
    Pipeline,
    RandomForestClassifier,
    RobustScaler,
    SVC,
    SelectKBest,
    StandardScaler,
    cross_val_score,
    f_classif,
    load_breast_cancer,
    mo,
    pipe_estimator,
    pipe_reducer,
    pipe_scaler,
    time,
    train_test_split,
):
    _cancer = load_breast_cancer()
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(_cancer.data, _cancer.target, random_state=0)

    _steps = []
    _short_args = []

    if pipe_scaler.value == "MinMaxScaler":
        _steps.append(("scaler", MinMaxScaler()))
        _short_args.append("MinMaxScaler()")
    elif pipe_scaler.value == "StandardScaler":
        _steps.append(("scaler", StandardScaler()))
        _short_args.append("StandardScaler()")
    elif pipe_scaler.value == "RobustScaler":
        _steps.append(("scaler", RobustScaler()))
        _short_args.append("RobustScaler()")

    if pipe_reducer.value == "PCA (n=10)":
        _steps.append(("pca", PCA(n_components=10, random_state=42)))
        _short_args.append("PCA(n_components=10)")
    elif pipe_reducer.value == "SelectKBest (k=15)":
        _steps.append(("select", SelectKBest(f_classif, k=15)))
        _short_args.append("SelectKBest(k=15)")

    if pipe_estimator.value == "SVC (C=100)":
        _steps.append(("svc", SVC(C=100)))
        _short_args.append("SVC(C=100)")
    elif pipe_estimator.value == "LogisticRegression":
        _steps.append(("logreg", LogisticRegression(max_iter=1500)))
        _short_args.append("LogisticRegression(max_iter=1500)")
    elif pipe_estimator.value == "RandomForestClassifier":
        _steps.append(("rf", RandomForestClassifier(n_estimators=60, random_state=42)))
        _short_args.append("RandomForestClassifier(n_estimators=60)")
    else:
        _steps.append(("gbdt", GradientBoostingClassifier(n_estimators=60, random_state=42)))
        _short_args.append("GradientBoostingClassifier(n_estimators=60)")

    _pipe = Pipeline(_steps)
    _t0 = time.perf_counter()
    _cv_scores = cross_val_score(_pipe, _X_tr, _y_tr, cv=5)
    _pipe.fit(_X_tr, _y_tr)
    _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
    _test_acc = float(_pipe.score(_X_te, _y_te))

    _explicit_lines = ",\n    ".join(
        f'("{n}", {arg})' for (n, _), arg in zip(_steps, _short_args, strict=True)
    )
    _short_lines = ", ".join(_short_args)

    _code_compare = mo.hstack(
        [
            mo.md(
                f"""**Explicit `Pipeline` (Custom Step Names)**\n```python\npipe_long = Pipeline([\n    {_explicit_lines}\n])\n```"""
            ),
            mo.md(
                f"""**Shorthand `make_pipeline` (Auto Lowercase Names)**\n```python\npipe_short = make_pipeline({_short_lines})\n# Step names: {[s[0] for s in _pipe.steps]}\n```"""
            ),
        ],
        widths="equal",
        gap=1.0,
    )

    _stats = mo.hstack(
        [
            mo.stat(f"{_cv_scores.mean():.1%}", label="5-Fold CV Accuracy", bordered=True),
            mo.stat(f"{_test_acc:.1%}", label="Held-Out Test Accuracy", bordered=True),
            mo.stat(f"{_elapsed_ms:.1f} ms", label="CV + Fit Time", bordered=True),
            mo.stat(str(len(_steps)), label="Pipeline Steps", bordered=True),
        ],
        widths="equal",
        gap=0.8,
    )

    _diagram = mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Interactive Scikit-Learn Pipeline Diagram** *(click any block to expand parameters)*"
                ),
                mo.Html(_pipe._repr_html_()),
            ]
        ),
        kind="neutral",
    )

    if pipe_scaler.value == "None (passthrough)" and pipe_estimator.value in (
        "SVC (C=100)",
        "LogisticRegression",
    ):
        _warn = mo.callout(
            mo.md(
                "**No Scaler Selected:** Notice how `SVC` / `LogisticRegression` accuracy drops sharply without feature scaling! Add `MinMaxScaler` or `StandardScaler` in Step 1."
            ),
            kind="warn",
        )
    else:
        _warn = mo.callout(
            mo.md(
                f"Every transformer (`{', '.join(s[0] for s in _steps[:-1]) or 'none'}`) is fit **exclusively inside each training fold** during CV—zero information from validation samples leaks into scaling or selection."
            ),
            kind="success",
        )

    mo.vstack([_stats, _diagram, _code_compare, _warn], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 2 · Data Leakage Simulator ("The #1 Reason Pipelines Exist")

    Consider a dataset of **pure Gaussian random noise** ($X \\sim \\mathcal{N}(0, 1)$, random binary labels $y \\in \\{0, 1\\}$). Because $X$ and $y$ are completely independent, **no model can genuinely beat 50% accuracy**.

    Watch what happens if you run supervised feature selection (`SelectPercentile`) **outside** `cross_val_score` versus **inside** a `Pipeline`!
    """)


@app.cell
def _(mo):
    leak_n_features = mo.ui.slider(
        start=200,
        stop=3000,
        step=200,
        value=1600,
        show_value=True,
        label="Total Noise Features (p)",
    )
    leak_percentile = mo.ui.slider(
        start=1, stop=20, step=1, value=5, show_value=True, label="SelectPercentile (%)"
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**Data Leakage Controls** — Increase noise dimensionality `p` or lower `%` kept to watch illusory leaky CV skyrocket!"
                ),
                mo.hstack([leak_n_features, leak_percentile], justify="start", wrap=True, gap=1.2),
            ]
        ),
        kind="info",
    )
    return leak_n_features, leak_percentile


@app.cell
def _(
    LogisticRegression,
    SelectPercentile,
    StandardScaler,
    cross_val_score,
    f_classif,
    go,
    leak_n_features,
    leak_percentile,
    make_pipeline,
    mo,
    np,
    train_test_split,
):
    _rng = np.random.default_rng(42)
    _n_samples = 180
    _p = int(leak_n_features.value)
    _pct = int(leak_percentile.value)

    _X = _rng.normal(size=(_n_samples, _p))
    _y = _rng.integers(0, 2, size=_n_samples)
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, test_size=0.30, stratify=_y, random_state=42
    )

    # 1. LEAKY: SelectPercentile uses ALL y_tr BEFORE cross_val_score splits folds!
    _sel_leaky = SelectPercentile(f_classif, percentile=_pct)
    _X_tr_leaky = _sel_leaky.fit_transform(_X_tr, _y_tr)
    _leaky_cv = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        _X_tr_leaky,
        _y_tr,
        cv=5,
    )
    # Evaluate that leaky model on truly unseen test data
    _leaky_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        _X_tr_leaky, _y_tr
    )
    _leaky_test = float(_leaky_clf.score(_sel_leaky.transform(_X_te), _y_te))

    # 2. HONEST PIPELINE: SelectPercentile fits INSIDE each CV fold!
    _honest_pipe = make_pipeline(
        SelectPercentile(f_classif, percentile=_pct),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    _honest_cv = cross_val_score(_honest_pipe, _X_tr, _y_tr, cv=5)
    _honest_pipe.fit(_X_tr, _y_tr)
    _honest_test = float(_honest_pipe.score(_X_te, _y_te))

    _labels = [
        "Leaky CV<br>(Select BEFORE CV)",
        "Honest Pipeline CV<br>(Select INSIDE CV)",
        "Unseen Test Set<br>(True Generalization)",
    ]
    _scores = [float(_leaky_cv.mean()), float(_honest_cv.mean()), _honest_test]
    _colors = ["#DC2626", "#059669", "#2563EB"]

    _fig = go.Figure(
        go.Bar(
            x=_labels,
            y=_scores,
            marker_color=_colors,
            text=[f"{s:.1%}" for s in _scores],
            textposition="outside",
            width=0.48,
        )
    )
    _fig.add_hline(
        y=0.50,
        line_dash="dash",
        line_color="#475569",
        annotation_text="True Signal Ceiling (50% Coin Flip)",
        annotation_position="top left",
    )
    _fig.update_yaxes(title_text="Accuracy", range=[0.0, 1.12])
    style_fig(
        _fig,
        f"Data Leakage on Pure Random Noise (p={_p} features, top {_pct}% selected)",
        height=400,
    )

    _inflation = float(_leaky_cv.mean()) - _honest_test
    _stats = mo.hstack(
        [
            mo.stat(f"{_leaky_cv.mean():.1%}", label="Leaky CV Accuracy (Fake!)", bordered=True),
            mo.stat(
                f"{_honest_cv.mean():.1%}", label="Pipeline CV Accuracy (Honest)", bordered=True
            ),
            mo.stat(f"{_honest_test:.1%}", label="Unseen Test Accuracy", bordered=True),
            mo.stat(f"{_inflation:+.1%}", label="Illusory Leakage Gain", bordered=True),
        ],
        widths="equal",
        gap=0.8,
    )

    _callout = mo.callout(
        mo.md(
            f"**Why did Leaky CV score {_leaky_cv.mean():.1%} on pure coin-flip noise?** With `{_p}` random features, a few dozen accidentally correlate with `y_tr` by pure chance. Fitting `SelectPercentile` before `cross_val_score` lets those spurious features peek at the CV validation targets! Wrapping selection inside a `Pipeline` keeps CV at `{_honest_cv.mean():.1%}`, accurately warning you that the model has zero real signal (`{_honest_test:.1%}` on test)."
        ),
        kind="danger" if _inflation > 0.12 else "info",
    )

    mo.vstack([_stats, mo.ui.plotly(_fig), _callout], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 3 · Pipelines in `GridSearchCV`: Hyperparameter Landscape & Multi-Model Search

    In a `Pipeline`, `GridSearchCV` addresses any step's hyperparameters using **`<stepname>__<parameter>`** (two underscores). Better yet, passing a **list of parameter dicts** lets a single `GridSearchCV` compare completely different scalers and estimators without leaking fold statistics.
    """)


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    MinMaxScaler,
    Pipeline,
    RandomForestClassifier,
    RobustScaler,
    SVC,
    StandardScaler,
    StratifiedKFold,
    functools,
    load_breast_cancer,
    mo,
    train_test_split,
):
    @functools.lru_cache(maxsize=6)
    def run_svc_pipeline_grid(scaler_name: str, cv_folds: int):
        _cancer = load_breast_cancer()
        _X_tr, _X_te, _y_tr, _y_te = train_test_split(_cancer.data, _cancer.target, random_state=0)
        _scalers = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
        }
        _pipe = Pipeline([("scaler", _scalers[scaler_name]), ("svc", SVC())])
        _c_vals = [0.001, 0.01, 0.1, 1, 10, 100]
        _g_vals = [0.001, 0.01, 0.1, 1, 10, 100]
        _grid = GridSearchCV(
            _pipe,
            param_grid={"svc__C": _c_vals, "svc__gamma": _g_vals},
            cv=StratifiedKFold(n_splits=cv_folds),
        ).fit(_X_tr, _y_tr)
        return _c_vals, _g_vals, _grid, float(_grid.score(_X_te, _y_te))

    @functools.lru_cache(maxsize=3)
    def run_multimodel_grid(cv_folds: int):
        _cancer = load_breast_cancer()
        _X_tr, _X_te, _y_tr, _y_te = train_test_split(_cancer.data, _cancer.target, random_state=0)
        _pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
        _param_grid = [
            {
                "preprocessing": [StandardScaler(), MinMaxScaler(), "passthrough"],
                "classifier": [LogisticRegression(max_iter=2000)],
                "classifier__C": [0.1, 1.0, 10.0],
            },
            {
                "preprocessing": [StandardScaler(), MinMaxScaler()],
                "classifier": [SVC()],
                "classifier__C": [1.0, 10.0, 100.0],
                "classifier__gamma": [0.01, 0.1],
            },
            {
                "preprocessing": ["passthrough"],
                "classifier": [RandomForestClassifier(n_estimators=60, random_state=42)],
                "classifier__max_depth": [3, 6, None],
            },
        ]
        _grid = GridSearchCV(
            _pipe, param_grid=_param_grid, cv=StratifiedKFold(n_splits=cv_folds)
        ).fit(_X_tr, _y_tr)
        return _grid, float(_grid.score(_X_te, _y_te))

    gs_scaler = mo.ui.radio(
        options=["MinMaxScaler", "StandardScaler", "RobustScaler"],
        value="MinMaxScaler",
        inline=True,
        label="Pipeline Scaler",
    )
    gs_folds = mo.ui.radio(
        options={"3-Fold": 3, "5-Fold": 5, "10-Fold": 10},
        value="5-Fold",
        inline=True,
        label="CV Folds",
    )
    mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**GridSearchCV Pipeline Controls** — Explore `svc__C × svc__gamma` and multi-model search"
                ),
                mo.hstack([gs_scaler, gs_folds], justify="start", wrap=True, gap=1.2),
            ]
        ),
        kind="info",
    )
    return gs_folds, gs_scaler, run_multimodel_grid, run_svc_pipeline_grid


@app.cell
def _(
    go,
    gs_folds,
    gs_scaler,
    mo,
    pd,
    run_multimodel_grid,
    run_svc_pipeline_grid,
):
    _c_vals, _g_vals, _grid, _test_acc = run_svc_pipeline_grid(gs_scaler.value, int(gs_folds.value))
    _means = _grid.cv_results_["mean_test_score"].reshape(len(_c_vals), len(_g_vals))
    _stds = _grid.cv_results_["std_test_score"].reshape(len(_c_vals), len(_g_vals))

    _c_lbls = [str(c) for c in _c_vals]
    _g_lbls = [str(g) for g in _g_vals]
    _hover = [
        [
            f"svc__C={_c_vals[r]}, svc__gamma={_g_vals[c]}<br>Mean CV={_means[r, c]:.4f} ± {_stds[r, c]:.4f}"
            for c in range(len(_g_vals))
        ]
        for r in range(len(_c_vals))
    ]

    _fig_hm = go.Figure()
    _fig_hm.add_trace(
        go.Heatmap(
            z=_means,
            x=_g_lbls,
            y=_c_lbls,
            colorscale="Viridis",
            text=[[f"{v:.3f}" for v in row] for row in _means],
            texttemplate="%{text}",
            hovertext=_hover,
            hoverinfo="text",
            colorbar=dict(title="Mean CV"),
        )
    )
    _fig_hm.add_trace(
        go.Scatter(
            x=[str(_grid.best_params_["svc__gamma"])],
            y=[str(_grid.best_params_["svc__C"])],
            mode="markers",
            name="Best Params",
            marker=dict(
                color="#FACC15",
                size=18,
                symbol="star",
                line=dict(width=1.6, color="#0F172A"),
            ),
        )
    )
    _fig_hm.update_xaxes(title_text="svc__gamma", type="category")
    _fig_hm.update_yaxes(title_text="svc__C", type="category")
    style_fig(
        _fig_hm,
        f"GridSearchCV Heatmap: Pipeline({gs_scaler.value} → SVC) on Breast Cancer",
        height=430,
    )

    # Multi-model pipeline comparison
    _mm_grid, _mm_test = run_multimodel_grid(int(gs_folds.value))
    _res_df = pd.DataFrame(_mm_grid.cv_results_).sort_values("rank_test_score").head(10).copy()
    _res_df["Pipeline Config"] = _res_df["params"].apply(
        lambda d: (
            f"{d['preprocessing'] if isinstance(d['preprocessing'], str) else type(d['preprocessing']).__name__}"
            f" → {type(d['classifier']).__name__}"
            + (f" (C={d['classifier__C']})" if "classifier__C" in d else "")
            + (f" (gamma={d['classifier__gamma']})" if "classifier__gamma" in d else "")
            + (f" (depth={d['classifier__max_depth']})" if "classifier__max_depth" in d else "")
        )
    )
    _res_sorted = _res_df.iloc[::-1]

    _fig_mm = go.Figure(
        go.Bar(
            x=_res_sorted["mean_test_score"],
            y=_res_sorted["Pipeline Config"],
            orientation="h",
            error_x=dict(type="data", array=_res_sorted["std_test_score"], visible=True),
            marker_color=[
                "#059669" if r == 1 else "#2563EB" for r in _res_sorted["rank_test_score"]
            ],
            text=[f"{s:.2%}" for s in _res_sorted["mean_test_score"]],
            textposition="outside",
        )
    )
    _fig_mm.update_xaxes(title_text="Mean Cross-Validation Accuracy", range=[0.88, 1.03])
    style_fig(_fig_mm, "Top 10 Configurations from Multi-Model Pipeline GridSearchCV", height=420)

    _stats = mo.hstack(
        [
            mo.stat(f"{_grid.best_score_:.2%}", label="Best SVC CV Score", bordered=True),
            mo.stat(f"{_test_acc:.2%}", label="SVC Test Score", bordered=True),
            mo.stat(
                f"C={_grid.best_params_['svc__C']}, γ={_grid.best_params_['svc__gamma']}",
                label="Best SVC Params",
                bordered=True,
            ),
            mo.stat(f"{_mm_test:.2%}", label="Best Multi-Model Test Score", bordered=True),
        ],
        widths="equal",
        gap=0.7,
    )

    _syntax_tip = mo.callout(
        mo.md(
            "Notice the double-underscore convention: `param_grid = {'svc__C': [...], 'svc__gamma': [...]}` tells `GridSearchCV` to route `C` and `gamma` to the step named `'svc'` inside `Pipeline`."
        ),
        kind="info",
    )

    mo.vstack([_stats, mo.ui.plotly(_fig_hm), _syntax_tip, mo.ui.plotly(_fig_mm)], gap=0.75)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Lab 4 · Real-World Tabular Data with `ColumnTransformer`

    Real datasets mix **numeric columns** (with missing values & vastly different scales) and **categorical columns** (strings that need one-hot encoding). **`ColumnTransformer`** sends each column type down its own leak-free preprocessing branch before feeding the combined features into any classifier.
    """)


@app.cell
def _(mo):
    ct_model = mo.ui.radio(
        options=["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"],
        value="LogisticRegression",
        inline=True,
        label="Downstream Classifier",
    )
    mo.callout(
        mo.hstack([mo.md("**ColumnTransformer Classifier:**"), ct_model], justify="start", gap=1.0),
        kind="info",
    )
    return (ct_model,)


@app.cell
def _(
    ColumnTransformer,
    GradientBoostingClassifier,
    LogisticRegression,
    OneHotEncoder,
    Pipeline,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
    cross_val_score,
    ct_model,
    go,
    mo,
    np,
    pd,
    train_test_split,
):
    # Deterministic mixed-type synthetic customer renewal dataset
    _rng = np.random.default_rng(123)
    _n = 420
    _age = _rng.integers(19, 72, size=_n).astype(float)
    _income = np.round(_rng.lognormal(mean=10.9, sigma=0.45, size=_n), 0)
    _tenure = _rng.integers(1, 72, size=_n).astype(float)
    _tier = _rng.choice(["Basic", "Pro", "Enterprise"], size=_n, p=[0.45, 0.35, 0.20])
    _region = _rng.choice(["North", "South", "West", "East"], size=_n)
    _autopay = _rng.choice(["Yes", "No"], size=_n, p=[0.65, 0.35])

    # Underlying renewal probability driven by tenure, tier, autopay, and income
    _logit = (
        -1.2
        + 0.04 * _tenure
        + 0.9 * (_tier == "Enterprise")
        + 0.45 * (_tier == "Pro")
        + 0.85 * (_autopay == "Yes")
        + 0.25 * ((_income - 55000) / 30000)
        + _rng.normal(0, 0.65, size=_n)
    )
    _y = (_logit > 0.5).astype(int)

    # Inject realistic missing values
    _age[_rng.random(_n) < 0.08] = np.nan
    _income[_rng.random(_n) < 0.06] = np.nan
    _tier_series = pd.Series(_tier, dtype="object")
    _tier_series.iloc[_rng.random(_n) < 0.05] = np.nan

    _df = pd.DataFrame(
        {
            "age": _age,
            "annual_income": _income,
            "tenure_months": _tenure,
            "plan_tier": _tier_series,
            "region": _region,
            "autopay": _autopay,
        }
    )

    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _df, _y, stratify=_y, test_size=0.25, random_state=42
    )

    _num_cols = ["age", "annual_income", "tenure_months"]
    _cat_cols = ["plan_tier", "region", "autopay"]

    _preproc = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                _num_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                _cat_cols,
            ),
        ]
    )

    if ct_model.value == "LogisticRegression":
        _clf = LogisticRegression(max_iter=1000)
    elif ct_model.value == "RandomForestClassifier":
        _clf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    else:
        _clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)

    _full_pipe = Pipeline([("preprocess", _preproc), ("model", _clf)])
    _cv = cross_val_score(_full_pipe, _X_tr, _y_tr, cv=5)
    _full_pipe.fit(_X_tr, _y_tr)
    _te_acc = float(_full_pipe.score(_X_te, _y_te))

    _feat_names = list(_full_pipe.named_steps["preprocess"].get_feature_names_out())
    _est = _full_pipe.named_steps["model"]
    if hasattr(_est, "coef_"):
        _imp = _est.coef_[0]
        _imp_title = "LogisticRegression Weights (Log-Odds Impact)"
    else:
        _imp = _est.feature_importances_
        _imp_title = f"{ct_model.value} Feature Importances"

    _order = np.argsort(np.abs(_imp))
    _fig = go.Figure(
        go.Bar(
            x=_imp[_order],
            y=[_feat_names[i] for i in _order],
            orientation="h",
            marker_color=["#2563EB" if v >= 0 else "#EA580C" for v in _imp[_order]],
        )
    )
    _fig.update_xaxes(title_text="Weight / Importance")
    style_fig(_fig, _imp_title, height=390)

    _stats = mo.hstack(
        [
            mo.stat(f"{_cv.mean():.1%}", label="5-Fold CV Accuracy", bordered=True),
            mo.stat(f"{_te_acc:.1%}", label="Test Accuracy", bordered=True),
            mo.stat(
                f"{_df.shape[1]} raw → {len(_feat_names)} encoded",
                label="Feature Expansion",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=0.8,
    )

    _preview = mo.accordion(
        {
            "Inspect Raw Mixed-Type Input Data (first 6 rows with NaNs & strings)": mo.ui.table(
                _df.head(6), selection=None, show_column_summaries=False
            )
        }
    )

    mo.vstack(
        [
            _stats,
            mo.callout(
                mo.vstack(
                    [
                        mo.md(
                            "**Branching Pipeline Architecture (`ColumnTransformer` + Classifier)**"
                        ),
                        mo.Html(_full_pipe._repr_html_()),
                    ]
                ),
                kind="neutral",
            ),
            mo.ui.plotly(_fig),
            _preview,
        ],
        gap=0.75,
    )


@app.cell
def _(mo, pd):
    mo.md("""
    ---
    ## Reference · Pipeline & Preprocessing Best Practices
    """)
    _cheat = pd.DataFrame(
        [
            {
                "Component": "Pipeline([...])",
                "Purpose": "Chain transformers & final estimator with explicit step names",
                "Key Rule / Gotcha": "All intermediate steps must implement .fit() and .transform()",
            },
            {
                "Component": "make_pipeline(...)",
                "Purpose": "Shorthand Pipeline with automatic lowercase step names",
                "Key Rule / Gotcha": "Duplicate class names auto-suffix (pca-1, pca-2); check .steps",
            },
            {
                "Component": "ColumnTransformer([...])",
                "Purpose": "Apply different pipelines to numeric vs. categorical columns",
                "Key Rule / Gotcha": "Default remainder='drop' drops unlisted columns unless set to 'passthrough'",
            },
            {
                "Component": "GridSearchCV + Pipeline",
                "Purpose": "Tune preprocessing and model hyperparameters jointly without leakage",
                "Key Rule / Gotcha": "Prefix every param with '<stepname>__' (double underscore)",
            },
            {
                "Component": "Data Leakage Prevention",
                "Purpose": "Keep validation/test distributions strictly isolated during fitting",
                "Key Rule / Gotcha": "Never call scaler/imputer/selector .fit() or .fit_transform() before CV/split",
            },
        ]
    )
    mo.ui.table(_cheat, selection=None, show_column_summaries=False)


if __name__ == "__main__":
    app.run()
