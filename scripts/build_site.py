"""Build static HTML-WASM site for zero-install GitHub Pages deployment."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
SITE_DIR = ROOT / "_site"
SLUG_RE = re.compile(r"^[a-z0-9_]+$")

NOTEBOOK_CATALOG = [
    {
        "slug": "01_supervised_learning",
        "title": "01 · Supervised Learning",
        "subtitle": "Classification & Regression Labs",
        "description": (
            "Explore Iris scatter matrices, k-NN decision boundaries & bias–variance curves, "
            "SVM scaling on Breast Cancer, Trees vs. Ensembles, an 8-classifier showdown, and Ridge/Lasso regularization."
        ),
        "tags": ["k-NN", "SVM", "Trees & Ensembles", "MLP", "Ridge & Lasso"],
    },
    {
        "slug": "02_unsupervised_learning",
        "title": "02 · Unsupervised Learning",
        "subtitle": "Transformations & Clustering",
        "description": (
            "Compare scaled vs. unscaled PCA on Breast Cancer, NMF parts & t-SNE manifolds on Digits, "
            "DBSCAN core/border/noise geometry, and k-Means vs. Agglomerative vs. DBSCAN."
        ),
        "tags": ["PCA", "NMF", "t-SNE", "DBSCAN", "k-Means", "Agglomerative"],
    },
    {
        "slug": "03_model_evaluation",
        "title": "03 · Model Evaluation",
        "subtitle": "Validation & Imbalanced Metrics",
        "description": (
            "Visualize unshuffled vs. stratified K-Fold splits, learning & validation curves, "
            "the Accuracy Paradox on imbalanced Digits, synced ROC/PR curves, and misclassified images."
        ),
        "tags": ["Cross-Validation", "GridSearchCV", "ROC / PR", "Threshold Tuning"],
    },
    {
        "slug": "04_building_pipelines",
        "title": "04 · Building Pipelines",
        "subtitle": "Leak-Free ML Workflows",
        "description": (
            "Assemble interactive scikit-learn Pipelines, simulate data leakage on high-dimensional noise, "
            "visualize GridSearchCV heatmaps, and preprocess mixed tabular data with ColumnTransformer."
        ),
        "tags": ["Pipeline", "Data Leakage", "GridSearchCV", "ColumnTransformer"],
    },
]

INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="referrer" content="no-referrer" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none';" />
  <title>Machine Learning Demo · Interactive Marimo Labs</title>
  <style>
    :root {{
      --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #475569;
      --border: #e2e8f0; --accent: #2563eb; --accent-hover: #1d4ed8;
      --tag-bg: #eff6ff; --tag-text: #1e40af;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0b1120; --card: #111827; --text: #f1f5f9; --muted: #94a3b8;
        --border: #1e293b; --accent: #3b82f6; --accent-hover: #60a5fa;
        --tag-bg: #1e293b; --tag-text: #93c5fd;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.6;
    }}
    .container {{ max-width: 1040px; margin: 0 auto; padding: 3rem 1.5rem 4rem; }}
    header {{ margin-bottom: 2.5rem; }}
    .eyebrow {{
      display: inline-block; font-size: 0.78rem; font-weight: 700;
      letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent);
      margin-bottom: 0.5rem;
    }}
    h1 {{ margin: 0 0 0.75rem; font-size: clamp(1.9rem, 3.2vw, 2.6rem); line-height: 1.15; }}
    .lead {{ margin: 0; max-width: 720px; color: var(--muted); font-size: 1.05rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.25rem; }}
    .card {{
      display: flex; flex-direction: column; background: var(--card);
      border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem;
      transition: transform 150ms ease, border-color 150ms ease;
    }}
    .card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
    .subtitle {{ font-size: 0.82rem; font-weight: 600; color: var(--muted); margin-bottom: 0.25rem; }}
    .card h2 {{ margin: 0 0 0.6rem; font-size: 1.25rem; }}
    .card p {{ margin: 0 0 1.1rem; color: var(--muted); font-size: 0.94rem; flex-grow: 1; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.2rem; }}
    .tag {{
      background: var(--tag-bg); color: var(--tag-text); border-radius: 999px;
      padding: 0.2rem 0.65rem; font-size: 0.76rem; font-weight: 600;
    }}
    .btn {{
      display: inline-flex; align-items: center; justify-content: center;
      background: var(--accent); color: #ffffff; text-decoration: none;
      font-weight: 600; font-size: 0.92rem; padding: 0.65rem 1rem;
      border-radius: 8px; transition: background 150ms ease;
    }}
    .btn:hover {{ background: var(--accent-hover); }}
    footer {{
      margin-top: 2.75rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
      color: var(--muted); font-size: 0.88rem; display: flex; flex-wrap: wrap;
      justify-content: space-between; gap: 0.75rem;
    }}
    footer a {{ color: var(--accent); text-decoration: none; }}
    code {{
      background: var(--tag-bg); color: var(--tag-text);
      padding: 0.15rem 0.4rem; border-radius: 5px; font-size: 0.84rem;
    }}
  </style>
</head>
<body>
  <main class="container">
    <header>
      <span class="eyebrow">Zero-Install Browser Edition · Powered by Marimo + Pyodide</span>
      <h1>Machine Learning Demo</h1>
      <p class="lead">
        Interactive, visual machine learning labs built with <strong>scikit-learn</strong>,
        <strong>Plotly</strong>, and <strong>Marimo</strong>. Everything runs directly in your browser—no
        local Python setup required. First load takes a few seconds while WebAssembly initializes.
      </p>
    </header>
    <section class="grid">
      {cards}
    </section>
    <footer>
      <span>Run locally anytime with <code>uv sync && uv run marimo edit notebooks/</code></span>
      <span>
        <a href="https://github.com/NewGuy012/machine-learning-demo">View Source on GitHub</a>
      </span>
    </footer>
  </main>
</body>
</html>
"""

CARD_TEMPLATE = """      <article class="card">
        <div class="subtitle">{subtitle}</div>
        <h2>{title}</h2>
        <p>{description}</p>
        <div class="tags">{tags}</div>
        <a class="btn" href="./{slug}/">Launch Interactive Lab &rarr;</a>
      </article>"""


def validate_slug(slug: str) -> str:
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"Invalid notebook slug: {slug!r}")
    return slug


def resolve_notebook_path(slug: str, notebooks_dir: Path = NOTEBOOKS_DIR) -> Path:
    validate_slug(slug)
    base = notebooks_dir.resolve()
    nb_path = (base / f"{slug}.py").resolve()
    if not nb_path.is_relative_to(base):
        raise ValueError(f"Notebook path escapes notebooks dir: {nb_path}")
    if not nb_path.is_file():
        raise FileNotFoundError(f"Expected notebook not found: {nb_path}")
    return nb_path


def validate_catalog(catalog: list[dict[str, object]], notebooks_dir: Path = NOTEBOOKS_DIR) -> None:
    slugs = [validate_slug(str(entry["slug"])) for entry in catalog]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Duplicate slugs found in NOTEBOOK_CATALOG")
    disk_slugs = {p.stem for p in notebooks_dir.glob("*.py")}
    catalog_slugs = set(slugs)
    if catalog_slugs != disk_slugs:
        missing = sorted(disk_slugs - catalog_slugs)
        extra = sorted(catalog_slugs - disk_slugs)
        raise ValueError(f"Catalog/notebook drift detected (missing={missing}, extra={extra})")


def render_card(entry: dict[str, object]) -> str:
    slug = validate_slug(str(entry["slug"]))
    tags = [str(t) for t in entry["tags"]]  # type: ignore[union-attr]
    tags_html = "".join(f'<span class="tag">{html.escape(t, quote=True)}</span>' for t in tags)
    return CARD_TEMPLATE.format(
        slug=html.escape(slug, quote=True),
        title=html.escape(str(entry["title"]), quote=True),
        subtitle=html.escape(str(entry["subtitle"]), quote=True),
        description=html.escape(str(entry["description"]), quote=True),
        tags=tags_html,
    )


def render_index(catalog: list[dict[str, object]]) -> str:
    return INDEX_TEMPLATE.format(cards="\n".join(render_card(entry) for entry in catalog))


def build_site() -> None:
    validate_catalog(NOTEBOOK_CATALOG, NOTEBOOKS_DIR)
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir(parents=True)

    for entry in NOTEBOOK_CATALOG:
        slug = validate_slug(str(entry["slug"]))
        nb_path = resolve_notebook_path(slug, NOTEBOOKS_DIR)
        out_dir = SITE_DIR / slug
        cmd = [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(nb_path),
            "-o",
            str(out_dir),
            "--mode",
            "run",
            "--show-code",
            "--no-sandbox",
            "-f",
        ]
        print(f"Exporting {nb_path.name} -> {out_dir} ...")
        subprocess.run(cmd, check=True)  # noqa: S603

    (SITE_DIR / "index.html").write_text(render_index(NOTEBOOK_CATALOG), encoding="utf-8")
    (SITE_DIR / ".nojekyll").touch()
    print(f"Static WASM site ready at: {SITE_DIR}")


if __name__ == "__main__":
    build_site()
