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
app = marimo.App(width="medium", app_title="04 · Building Pipelines")

with app.setup:
    import functools
    import time

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
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
    from sklearn.utils import estimator_html_repr


@app.function
def style_fig(fig, title: str, height: int = 420):
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
def build_lab1_steps(scaler_name: str, reducer_name: str, estimator_name: str):
    scalers = {
        "MinMaxScaler": ("scaler", MinMaxScaler(), "MinMaxScaler()"),
        "StandardScaler": ("scaler", StandardScaler(), "StandardScaler()"),
        "RobustScaler": ("scaler", RobustScaler(), "RobustScaler()"),
    }
    reducers = {
        "PCA (n=10)": ("pca", PCA(n_components=10, random_state=42), "PCA(n_components=10)"),
        "SelectKBest (k=15)": ("select", SelectKBest(f_classif, k=15), "SelectKBest(k=15)"),
    }
    estimators = {
        "SVC (C=100)": ("svc", SVC(C=100), "SVC(C=100)"),
        "LogisticRegression": (
            "logreg",
            LogisticRegression(solver="liblinear", max_iter=1000),
            'LogisticRegression(solver="liblinear", max_iter=1000)',
        ),
        "RandomForestClassifier": (
            "rf",
            RandomForestClassifier(n_estimators=60, random_state=42),
            "RandomForestClassifier(n_estimators=60)",
        ),
        "GradientBoostingClassifier": (
            "gbdt",
            GradientBoostingClassifier(n_estimators=60, random_state=42),
            "GradientBoostingClassifier(n_estimators=60)",
        ),
    }
    chosen = [
        item
        for item in (
            scalers.get(scaler_name),
            reducers.get(reducer_name),
            estimators[estimator_name],
        )
        if item is not None
    ]
    steps = [(name, est) for name, est, _ in chosen]
    short_args = [arg for _, _, arg in chosen]
    return steps, short_args


@app.function
@functools.lru_cache(maxsize=6)
def run_svc_pipeline_grid(scaler_name: str, cv_folds: int):
    cancer = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(cancer.data, cancer.target, random_state=0)
    scalers = {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }
    pipe = Pipeline([("scaler", scalers[scaler_name]), ("svc", SVC())])
    c_vals = [0.001, 0.01, 0.1, 1, 10, 100]
    g_vals = [0.001, 0.01, 0.1, 1, 10, 100]
    grid = GridSearchCV(
        pipe,
        param_grid={"svc__C": c_vals, "svc__gamma": g_vals},
        cv=StratifiedKFold(n_splits=cv_folds),
    ).fit(X_tr, y_tr)
    return c_vals, g_vals, grid, float(grid.score(X_te, y_te))


@app.function
@functools.lru_cache(maxsize=3)
def run_multimodel_grid(cv_folds: int):
    cancer = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(cancer.data, cancer.target, random_state=0)
    pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
    param_grid = [
        {
            "preprocessing": [StandardScaler(), MinMaxScaler(), "passthrough"],
            "classifier": [LogisticRegression(solver="liblinear", max_iter=1000)],
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
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=StratifiedKFold(n_splits=cv_folds)).fit(
        X_tr, y_tr
    )
    return grid, float(grid.score(X_te, y_te))


@app.function
def format_pipeline_config(params: dict) -> str:
    pre = params["preprocessing"]
    pre_name = pre if isinstance(pre, str) else type(pre).__name__
    clf_name = type(params["classifier"]).__name__
    parts = [f"{pre_name} → {clf_name}"]
    for key, label in (
        ("classifier__C", "C"),
        ("classifier__gamma", "gamma"),
        ("classifier__max_depth", "depth"),
    ):
        if key in params:
            parts.append(f"({label}={params[key]})")
    return " ".join(parts)


@app.function
@functools.lru_cache(maxsize=1)
def make_renewal_dataset():
    rng = np.random.default_rng(123)
    n = 420
    age = rng.integers(19, 72, size=n).astype(float)
    income = np.round(rng.lognormal(mean=10.9, sigma=0.45, size=n), 0)
    tenure = rng.integers(1, 72, size=n).astype(float)
    tier = rng.choice(["Basic", "Pro", "Enterprise"], size=n, p=[0.45, 0.35, 0.20])
    region = rng.choice(["North", "South", "West", "East"], size=n)
    autopay = rng.choice(["Yes", "No"], size=n, p=[0.65, 0.35])

    logit = (
        -1.2
        + 0.04 * tenure
        + 0.9 * (tier == "Enterprise")
        + 0.45 * (tier == "Pro")
        + 0.85 * (autopay == "Yes")
        + 0.25 * ((income - 55000) / 30000)
        + rng.normal(0, 0.65, size=n)
    )
    y = (logit > 0.5).astype(int)

    age[rng.random(n) < 0.08] = np.nan
    income[rng.random(n) < 0.06] = np.nan
    tier_obj = tier.astype(object)
    tier_obj[rng.random(n) < 0.05] = np.nan

    col_names = ["age", "annual_income", "tenure_months", "plan_tier", "region", "autopay"]
    X = np.column_stack(
        [
            age.astype(object),
            income.astype(object),
            tenure.astype(object),
            tier_obj,
            region.astype(object),
            autopay.astype(object),
        ]
    )
    preview_rows = [
        {
            name: (None if isinstance(val, float) and np.isnan(val) else val)
            for name, val in zip(col_names, row, strict=True)
        }
        for row in X[:6]
    ]
    return X, y, col_names, preview_rows


@app.cell
def _():
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


@app.cell
def _():
    mo.md("""
    ---
    ## Lab 1 · Interactive Pipeline Builder (`Pipeline` vs. `make_pipeline`)

    Choose each step of your Breast Cancer pipeline below. Scikit-learn renders an **interactive block diagram** (click any step inside the diagram to inspect its hyperparameters) alongside side-by-side explicit `Pipeline` and shorthand `make_pipeline` Python code.
    """)


@app.cell
def _():
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
def _(pipe_estimator, pipe_reducer, pipe_scaler):
    _cancer = load_breast_cancer()
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(_cancer.data, _cancer.target, random_state=0)

    _steps, _short_args = build_lab1_steps(
        pipe_scaler.value, pipe_reducer.value, pipe_estimator.value
    )
    _pipe = Pipeline(_steps)
    _auto_names = [s[0] for s in make_pipeline(*(est for _, est in _steps)).steps]

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
                f"""**Shorthand `make_pipeline` (Auto Lowercase Names)**\n```python\npipe_short = make_pipeline({_short_lines})\n# Step names: {_auto_names}\n```"""
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
                mo.Html(estimator_html_repr(_pipe)),
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
def _():
    mo.md("""
    ---
    ## Lab 2 · Data Leakage Simulator ("The #1 Reason Pipelines Exist")

    Consider a dataset of **pure Gaussian random noise** ($X \\sim \\mathcal{N}(0, 1)$, random binary labels $y \\in \\{0, 1\\}$). Because $X$ and $y$ are completely independent, **no model can genuinely beat 50% accuracy**.

    Watch what happens if you run supervised feature selection (`SelectPercentile`) **outside** `cross_val_score` versus **inside** a `Pipeline`!
    """)


@app.cell
def _():
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
def _(leak_n_features, leak_percentile):
    _rng = np.random.default_rng(42)
    _n_samples = 180
    _p = int(leak_n_features.value)
    _pct = int(leak_percentile.value)

    _X = _rng.normal(size=(_n_samples, _p))
    _y = _rng.integers(0, 2, size=_n_samples)
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, test_size=0.30, stratify=_y, random_state=42
    )

    # Leaky: feature selection sees all y_tr before CV folds are split
    _X_tr_leaky = SelectPercentile(f_classif, percentile=_pct).fit_transform(_X_tr, _y_tr)
    _leaky_cv = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        _X_tr_leaky,
        _y_tr,
        cv=5,
    )

    # Honest: feature selection runs inside each CV fold
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

    _fig = go.Figure(
        go.Bar(
            x=_labels,
            y=_scores,
            marker_color=["#DC2626", "#059669", "#2563EB"],
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
def _():
    mo.md("""
    ---
    ## Lab 3 · Pipelines in `GridSearchCV`: Hyperparameter Landscape & Multi-Model Search

    In a `Pipeline`, `GridSearchCV` addresses any step's hyperparameters using **`<stepname>__<parameter>`** (two underscores). Better yet, passing a **list of parameter dicts** lets a single `GridSearchCV` compare completely different scalers and estimators without leaking fold statistics.
    """)


@app.cell
def _():
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
    return gs_folds, gs_scaler


@app.cell
def _(gs_folds, gs_scaler):
    _c_vals, _g_vals, _grid, _test_acc = run_svc_pipeline_grid(gs_scaler.value, int(gs_folds.value))
    _means = _grid.cv_results_["mean_test_score"].reshape(len(_c_vals), len(_g_vals))
    _stds = _grid.cv_results_["std_test_score"].reshape(len(_c_vals), len(_g_vals))

    _hover = [
        [
            f"svc__C={_c_vals[r]}, svc__gamma={_g_vals[c]}<br>Mean CV={_means[r, c]:.4f} ± {_stds[r, c]:.4f}"
            for c in range(len(_g_vals))
        ]
        for r in range(len(_c_vals))
    ]
    _best_r = _c_vals.index(_grid.best_params_["svc__C"])
    _best_c = _g_vals.index(_grid.best_params_["svc__gamma"])
    _hm_text = [
        [
            f"<b>★ {_means[r, c]:.3f}</b>"
            if (r == _best_r and c == _best_c)
            else f"{_means[r, c]:.3f}"
            for c in range(len(_g_vals))
        ]
        for r in range(len(_c_vals))
    ]

    _fig_hm = go.Figure(
        go.Heatmap(
            z=_means,
            x=[str(g) for g in _g_vals],
            y=[str(c) for c in _c_vals],
            colorscale="Viridis",
            text=_hm_text,
            texttemplate="%{text}",
            hovertext=_hover,
            hoverinfo="text",
            colorbar={"title": "Mean CV"},
        )
    )
    _fig_hm.add_shape(
        type="rect",
        x0=_best_c - 0.48,
        x1=_best_c + 0.48,
        y0=_best_r - 0.48,
        y1=_best_r + 0.48,
        line={"color": "#0F172A", "width": 3},
    )
    _fig_hm.update_xaxes(title_text="svc__gamma", type="category")
    _fig_hm.update_yaxes(title_text="svc__C", type="category")
    style_fig(
        _fig_hm,
        f"GridSearchCV Heatmap: Pipeline({gs_scaler.value} → SVC) on Breast Cancer",
        height=430,
    )

    _mm_grid, _mm_test = run_multimodel_grid(int(gs_folds.value))
    _cv_res = _mm_grid.cv_results_
    _top10 = sorted(
        zip(
            _cv_res["rank_test_score"],
            _cv_res["mean_test_score"],
            _cv_res["std_test_score"],
            _cv_res["params"],
            strict=True,
        ),
        key=lambda row: (int(row[0]), -float(row[1])),
    )[:10][::-1]

    _fig_mm = go.Figure(
        go.Bar(
            x=[float(m) for _, m, _, _ in _top10],
            y=[format_pipeline_config(p) for _, _, _, p in _top10],
            orientation="h",
            error_x={"type": "data", "array": [float(s) for _, _, s, _ in _top10], "visible": True},
            marker_color=["#059669" if int(r) == 1 else "#2563EB" for r, _, _, _ in _top10],
            text=[f"{float(m):.2%}" for _, m, _, _ in _top10],
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
def _():
    mo.md("""
    ---
    ## Lab 4 · Real-World Tabular Data with `ColumnTransformer`

    Real datasets mix **numeric columns** (with missing values & vastly different scales) and **categorical columns** (strings that need one-hot encoding). **`ColumnTransformer`** sends each column type down its own leak-free preprocessing branch before feeding the combined features into any classifier.
    """)


@app.cell
def _():
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
def _(ct_model):
    _X, _y, _col_names, _preview_rows = make_renewal_dataset()
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(
        _X, _y, stratify=_y, test_size=0.25, random_state=42
    )

    _preproc = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                [0, 1, 2],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                [3, 4, 5],
            ),
        ]
    )

    _clfs = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=80, max_depth=6, random_state=42
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        ),
    }
    _full_pipe = Pipeline([("preprocess", _preproc), ("model", _clfs[ct_model.value])])
    _cv = cross_val_score(_full_pipe, _X_tr, _y_tr, cv=5)
    _full_pipe.fit(_X_tr, _y_tr)
    _te_acc = float(_full_pipe.score(_X_te, _y_te))

    _feat_names = list(_full_pipe.named_steps["preprocess"].get_feature_names_out(_col_names))
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
                f"{_X.shape[1]} raw → {len(_feat_names)} encoded",
                label="Feature Expansion",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=0.8,
    )

    _preview = mo.accordion(
        {
            "Inspect Raw Mixed-Type Input Data (first 6 rows with missing values & strings)": mo.ui.table(
                _preview_rows, selection=None, show_column_summaries=False
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
                        mo.Html(estimator_html_repr(_full_pipe)),
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
def _():
    _rows = [
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
    mo.vstack(
        [
            mo.md("---\n## Reference · Pipeline & Preprocessing Best Practices"),
            mo.ui.table(_rows, selection=None, show_column_summaries=False),
        ],
        gap=0.5,
    )


if __name__ == "__main__":
    app.run()
