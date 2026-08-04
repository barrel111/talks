# GitHub Pages deployment

This project exports the talk as a readonly Marimo WebAssembly application.
Python runs entirely in the visitor's browser; GitHub Pages only serves static
files.

## Files to migrate

Copy these paths into the root of the destination GitHub Pages repository while
preserving their relative locations:

- `.github/workflows/deploy-pages.yml`
- `notebooks/jl_nb.py`
- `notebooks/talk.css`
- `notebooks/__marimo__/session/jl_nb.py.json`
- `notebooks/public/mnist_10k.npz`
- `requirements-pages.txt`

The `scripts/prepare_wasm_assets.py` script and source `data/mnist_10k.npz` are
only needed if the browser bundle must be regenerated.

## Enable deployment

1. Push the migrated files to the destination repository's `main` branch.
2. Open **Settings → Pages** in that repository.
3. Under **Build and deployment**, choose **GitHub Actions** as the source.
4. Open **Actions → Deploy Marimo WASM app to GitHub Pages** and run the
   workflow if the initial push did not start it automatically.
5. The deployment job reports the published URL. For a project repository it is
   normally `https://<username>.github.io/<repository>/`.

The workflow validates the notebook, exports it with `marimo export html-wasm`,
uploads `_site` as the Pages artifact, and deploys it. Marimo creates the root
`index.html` and `.nojekyll` file automatically.

## Local verification

From the project root:

```bash
python -m pip install -r requirements-pages.txt
cd notebooks

UV_CACHE_DIR=/tmp/jl-deerwalk-uv-cache \
UV_TOOL_DIR=/tmp/jl-deerwalk-uv-tools \
UV_TOOL_BIN_DIR=/tmp/jl-deerwalk-uv-bin \
marimo export html-wasm jl_nb.py -o ../_site --mode run --force

python -m http.server --directory ../_site 8000
```

Then visit `http://localhost:8000`. A WASM export must be served over HTTP; it
does not run correctly when `index.html` is opened directly from disk.

## Existing Pages sites

This workflow deploys `_site` as the entire Pages artifact. If the destination
repository already hosts other pages, integrate `_site` into that repository's
existing build rather than enabling this workflow unchanged. The generated app
uses relative asset URLs and can be placed in a subdirectory such as
`jl-deerwalk/`.
