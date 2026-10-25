# GitHub Pages deployment

This repository is published as the GitHub Pages project site
`https://barrel111.github.io/talks/`. The Deerwalk talk is exported into the
`deerwalk-26/` subdirectory and is therefore available at:

```text
https://barrel111.github.io/talks/deerwalk-26/
```

The source remains in `jl-deerwalk-26/`; the source directory name does not
determine the public URL.

## Repository layout

```text
talks/
├── .github/workflows/deploy-pages.yml
└── jl-deerwalk-26/
    ├── notebooks/
    │   ├── jl_nb.py
    │   ├── talk.css
    │   └── public/mnist_10k.npz
    └── requirements-pages.txt
```

The workflow creates this deployment artifact:

```text
_site/
├── .nojekyll
└── deerwalk-26/
    ├── index.html
    └── assets/
```

## Enable deployment

1. Commit and push the repository to the `main` branch.
2. In `barrel111/talks`, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Open **Actions → Deploy Marimo WASM app to GitHub Pages** and inspect the
   run triggered by the push. If needed, start it with **Run workflow**.
5. After the deploy job succeeds, visit
   `https://barrel111.github.io/talks/deerwalk-26/`.

## Local verification

Run these commands from the repository root (`talks/`):

```bash
python -m pip install -r jl-deerwalk-26/requirements-pages.txt
marimo check jl-deerwalk-26/notebooks/jl_nb.py

cd jl-deerwalk-26/notebooks
UV_CACHE_DIR=/tmp/jl-deerwalk-uv-cache \
  marimo export html-wasm jl_nb.py \
  -o ../../_site/deerwalk-26 --mode run --force

cd ../..
touch _site/.nojekyll
python -m http.server --directory _site 8000
```

Then visit `http://localhost:8000/deerwalk-26/`. The WASM export must be
served over HTTP; opening `index.html` directly from disk is insufficient.

## Adding another talk

Export each additional talk into another directory beneath `_site/`, such as
`_site/another-talk/`, before the artifact-upload step. Every directory will
become another URL under `https://barrel111.github.io/talks/`.
