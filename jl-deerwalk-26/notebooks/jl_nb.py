import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import sklearn
    from matplotlib.ticker import FuncFormatter
    from scipy.stats import chi2
    from sklearn.decomposition import PCA

    return FuncFormatter, PCA, Path, chi2, np, plt, sns


@app.cell(hide_code=True)
def _(FuncFormatter):
    def latex_plot_ticks(axis, *, x=True, y=True):
        def _formatter(value, _position):
            if abs(value) < 1e-12:
                value = 0
            return rf"${value:g}$"

        if x:
            axis.xaxis.set_major_formatter(FuncFormatter(_formatter))
        if y:
            axis.yaxis.set_major_formatter(FuncFormatter(_formatter))

    return (latex_plot_ticks,)


@app.cell(hide_code=True)
def _(mo, show_lecture_notes):
    _notes_are_visible = show_lecture_notes.value

    def lecture_note(content):
        if not _notes_are_visible:
            return None
        if isinstance(content, str):
            return mo.md(content)
        return content

    def presentation_md(content):
        if isinstance(content, str):
            return mo.md(content)
        return content

    def visualization_block(content):
        if _notes_are_visible:
            return content
        return mo.callout(content, kind="neutral")

    return lecture_note, presentation_md, visualization_block


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    # Randomness and Structure
    ## The Johnson-Lindenstrauss Lemma
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    show_lecture_notes = mo.ui.switch(
        value=True,
        label="Detailed Notes",
    )
    show_lecture_notes
    return (show_lecture_notes,)


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    ### An Introduction to MNIST
    """)
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    We start by considering the MNIST dataset-- a large dataset of handwritten digits. The images are black and white, centered to fit in a $28 \times 28$ bounding box.
    """)
    return


@app.cell(hide_code=True)
def _(Path, sns):
    RANDOM_SEED = 42
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MNIST_PATH = PROJECT_ROOT / "data" / "mnist_10k.npz"
    PLOT_FONT_SIZES = {
        "figure_title": 13,
        "panel_title": 9,
        "axis_label": 9,
        "detail": 8,
    }

    sns.set_theme(
        style="white",
        context="talk",
        font="DejaVu Sans",
        rc={
            "mathtext.fontset": "stix",
            "figure.titlesize": PLOT_FONT_SIZES["figure_title"],
            "axes.titlesize": PLOT_FONT_SIZES["figure_title"],
            "axes.labelsize": PLOT_FONT_SIZES["axis_label"],
            "xtick.labelsize": PLOT_FONT_SIZES["detail"],
            "ytick.labelsize": PLOT_FONT_SIZES["detail"],
            "legend.fontsize": PLOT_FONT_SIZES["detail"],
        },
    )
    return MNIST_PATH, PLOT_FONT_SIZES, RANDOM_SEED


@app.cell(hide_code=True)
def _(MNIST_PATH, np):
    if not MNIST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MNIST_PATH}. Run `python scripts/prepare_mnist.py` "
            "from the project root first."
        )

    with np.load(MNIST_PATH) as _mnist:
        X_uint8 = _mnist["images"]
        y = _mnist["labels"]

    X = X_uint8.astype(np.float32) / 255.0

    assert X.shape == (10_000, 784)
    assert y.shape == (10_000,)
    assert np.array_equal(np.unique(y), np.arange(10, dtype=np.uint8))
    return X, X_uint8, y


@app.cell(hide_code=True)
def _():
    # lecture_note(f"""
    # #### Setup check
    # - **Dataset:** `{MNIST_PATH.name}` loaded locally
    # - **Examples:** {X.shape[0]:,}
    # - **Original dimension:** {X.shape[1]}
    # - **Classes:** {len(set(y.tolist()))}
    # - **scikit-learn:** {sklearn.__version__}
    # """)
    return


@app.cell(hide_code=True)
def _(PLOT_FONT_SIZES, X_uint8, np, plt, sns, visualization_block, y):
    _fig, _axes = plt.subplots(2, 5, figsize=(6.4, 3.0), dpi=120)
    for _digit, _axis in enumerate(_axes.flat):
        _index = np.flatnonzero(y == _digit)[0]
        sns.heatmap(
            X_uint8[_index].reshape(28, 28),
            ax=_axis,
            cmap="gray_r",
            cbar=False,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        _axis.set_title(
            rf"${_digit}$",
            fontsize=PLOT_FONT_SIZES["panel_title"],
        )
        _axis.axis("off")

    _fig.suptitle(
        "MNIST Dataset",
        fontsize=PLOT_FONT_SIZES["figure_title"],
    )
    _fig.tight_layout()
    visualization_block(_fig)
    return


@app.cell
def _(presentation_md):
    presentation_md(r"""
    Really, we may view these as elements of $\mathbb{R}^{28 \times 28} \cong \mathbb{R}^{784}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    digit_dropdown = mo.ui.dropdown(
        options={str(_digit): _digit for _digit in range(10)},
        value="0",
        label="Digit",
    )
    refresh_button = mo.ui.button(
        value=0,
        on_click=lambda _count: _count + 1,
        label="Refresh",
    )
    vector_view = mo.ui.checkbox(
        value=False,
        label="View as vectors in ℝ⁷⁸⁴",
    )
    digit_controls = mo.hstack(
        [digit_dropdown, refresh_button, vector_view],
        justify="start",
        align="center",
        gap=1,
    )
    return digit_controls, digit_dropdown, refresh_button, vector_view


@app.cell(hide_code=True)
def _(
    PLOT_FONT_SIZES,
    RANDOM_SEED,
    X_uint8,
    digit_controls,
    digit_dropdown,
    mo,
    np,
    plt,
    refresh_button,
    sns,
    vector_view,
    visualization_block,
    y,
):
    _digit = digit_dropdown.value
    _refresh_count = refresh_button.value
    _rng = np.random.default_rng(
        RANDOM_SEED + 1_000 * _digit + _refresh_count
    )
    _digit_indices = np.flatnonzero(y == _digit)
    _sample_indices = _rng.choice(_digit_indices, size=15, replace=False)

    if vector_view.value:
        _fig, _axis = plt.subplots(figsize=(7.0, 3.2), dpi=120)
        sns.heatmap(
            X_uint8[_sample_indices],
            ax=_axis,
            cmap="gray_r",
            cbar=False,
            xticklabels=False,
            yticklabels=[rf"${_sample}$" for _sample in range(1, 16)],
        )
        _axis.set_xticks(
            [0.5, 196.5, 392.5, 588.5, 783.5],
            labels=[r"$1$", r"$197$", r"$393$", r"$589$", r"$784$"],
        )
        _axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            labelsize=PLOT_FONT_SIZES["detail"],
            length=3,
            width=0.8,
            bottom=True,
            left=True,
        )
        _axis.set_xlabel(
            r"Coordinate in $\mathbb{R}^{784}$",
            fontsize=PLOT_FONT_SIZES["axis_label"],
            labelpad=4,
        )
        _axis.set_ylabel(
            "Sample",
            fontsize=PLOT_FONT_SIZES["axis_label"],
            labelpad=4,
        )
        for _spine in _axis.spines.values():
            _spine.set_visible(True)
            _spine.set_linewidth(1.1)
            _spine.set_edgecolor("0.25")
    else:
        _fig, _axes = plt.subplots(3, 5, figsize=(7.0, 4.2), dpi=120)
        for _index, _axis in zip(_sample_indices, _axes.flat):
            sns.heatmap(
                X_uint8[_index].reshape(28, 28),
                ax=_axis,
                cmap="gray_r",
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            _axis.set_xticks([])
            _axis.set_yticks([])
            _axis.set_xlabel("")
            _axis.set_ylabel("")
            for _spine in _axis.spines.values():
                _spine.set_visible(True)
                _spine.set_linewidth(1.0)
                _spine.set_edgecolor("0.35")

    _fig.suptitle(
        "MNIST Digit Classes",
        fontsize=PLOT_FONT_SIZES["figure_title"],
    )
    _fig.tight_layout()
    visualization_block(mo.vstack([digit_controls, _fig], gap=0.5))
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    How can a computer figure out which "class" of digits a given picture corresponds to? Intuitively, we expect the relative activations of a pixel to be similar. Does Euclidean distance capture this notion? In some sense, yes-- the closest neighbors usually correspond to the appropriate digit class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    neighbor_digit = mo.ui.dropdown(
        options={str(_digit): _digit for _digit in range(10)},
        value="3",
        label="Query digit",
    )
    neighbor_refresh = mo.ui.button(
        value=0,
        on_click=lambda _count: _count + 1,
        label="New query",
    )
    neighbor_controls = mo.hstack(
        [neighbor_digit, neighbor_refresh],
        justify="start",
        align="center",
        gap=1,
    )
    return neighbor_controls, neighbor_digit, neighbor_refresh


@app.cell(hide_code=True)
def _(RANDOM_SEED, neighbor_digit, neighbor_refresh, np, y):
    _digit = neighbor_digit.value
    _refresh_count = neighbor_refresh.value
    _rng = np.random.default_rng(
        RANDOM_SEED + 100_000 + 1_000 * _digit + _refresh_count
    )
    _eligible_queries = np.flatnonzero(y == _digit)
    query_index = int(_rng.choice(_eligible_queries))
    return (query_index,)


@app.cell(hide_code=True)
def _(
    PLOT_FONT_SIZES,
    X,
    X_uint8,
    lecture_note,
    mo,
    neighbor_controls,
    np,
    plt,
    query_index,
    sns,
    visualization_block,
    y,
):
    _squared_distances = np.sum((X - X[query_index]) ** 2, axis=1)
    _squared_distances[query_index] = np.inf
    _neighbor_indices = np.argpartition(_squared_distances, 10)[:10]
    _neighbor_indices = _neighbor_indices[
        np.argsort(_squared_distances[_neighbor_indices])
    ]
    _neighbor_distances = np.sqrt(_squared_distances[_neighbor_indices])
    _query_label = int(y[query_index])
    _matching_neighbors = int(
        np.sum(y[_neighbor_indices] == _query_label)
    )

    _fig = plt.figure(figsize=(7.6, 3.4), dpi=120)
    _grid = _fig.add_gridspec(
        2,
        6,
        width_ratios=(1.2, 1, 1, 1, 1, 1),
        wspace=0.18,
        hspace=0.50,
    )

    _query_axis = _fig.add_subplot(_grid[:, 0])
    sns.heatmap(
        X_uint8[query_index].reshape(28, 28),
        ax=_query_axis,
        cmap="gray_r",
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    _query_axis.set_title(
        f"Query\nlabel ${_query_label}$",
        fontsize=PLOT_FONT_SIZES["panel_title"],
    )
    _query_axis.set_xticks([])
    _query_axis.set_yticks([])
    for _spine in _query_axis.spines.values():
        _spine.set_visible(True)
        _spine.set_linewidth(1.8)
        _spine.set_edgecolor("0.15")

    for _rank, (_index, _distance) in enumerate(
        zip(_neighbor_indices, _neighbor_distances), start=1
    ):
        _row = (_rank - 1) // 5
        _column = 1 + (_rank - 1) % 5
        _axis = _fig.add_subplot(_grid[_row, _column])
        sns.heatmap(
            X_uint8[_index].reshape(28, 28),
            ax=_axis,
            cmap="gray_r",
            cbar=False,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        _neighbor_label = int(y[_index])
        _match_symbol = "✓" if _neighbor_label == _query_label else "×"
        _axis.set_title(
            f"#${_rank}$ · label ${_neighbor_label}$ {_match_symbol}\n"
            f"$d={_distance:.2f}$",
            fontsize=PLOT_FONT_SIZES["detail"],
            pad=3,
        )
        _axis.set_xticks([])
        _axis.set_yticks([])
        for _spine in _axis.spines.values():
            _spine.set_visible(True)
            _spine.set_linewidth(0.9)
            _spine.set_edgecolor("0.35")

    _fig.suptitle(
        r"Euclidean Nearest Neighbors in ℝ$^{784}$",
        fontsize=PLOT_FONT_SIZES["figure_title"],
        fontfamily="DejaVu Sans",
        math_fontfamily="dejavusans",
        y=0.985,
    )
    _fig.subplots_adjust(top=0.82, bottom=0.06, left=0.02, right=0.99)
    _caption = lecture_note(
        f"**${_matching_neighbors}/10$ neighbors share the query label.** "
        "Distances are computed from normalized pixel intensities."
    )
    _visualization_items = [neighbor_controls, _fig]
    if _caption is not None:
        _visualization_items.append(_caption)
    visualization_block(
        mo.vstack(_visualization_items, gap=0.25)
    )
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    This isn't to say that we can make a good classifier just by considering the Euclidean distance. Just that, among other things, it does carry some important semantic information that we _may be interested in preserving_.
    """)
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    ### Dimensionality Reduction
    """)
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    Suppose we have very stringent computational or storage requirements that make working in a high-dimensional space untenable. How should we resolve it? This calls for some dimensionality reduction! For this to be any useful, we must preserve structures of the dataset that corresponds to the semantic information we care about. It is hard to ascertain what this structure exactly is (let alone decide how we can "preserve" it)-- for this talk, we use Euclidean distances as a proxy motivated by our prior investigations. Note that MNIST is being used here just as a toy example, the regime where high-dimensional spaces become a practical issue is really when we get to millions of dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    More formally, we want a map $P: \mathbb{R}^{d} \to \mathbb{R}^{k}$, where $d = 784$ and $k \ll d$, such that

    \[
    \lVert P\mathbf{x}^{(i)}-P\mathbf{x}^{(j)}\rVert_2
    \approx
    \lVert \mathbf{x}^{(i)}-\mathbf{x}^{(j)}\rVert_2
    \]

    for every pair of points in our dataset.
    """)
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    Given access to the dataset, PCA is the natural solution. It learns a subspace optimized for MNIST. Can we choose the map before inspecting the data and still preserve its pairwise geometry?
    """)
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    Choose $P \in \mathbb{R}^{k \times d}$ independently of the data, with

    \[
    P_{ab} \overset{\mathrm{iid}}{\sim} \mathcal{N}\!\left(0,\frac{1}{k}\right),
    \qquad \mathbf{x} \longmapsto P\mathbf{x}.
    \]

    Seemingly we haven't even considered the contents of the MNIST dataset. Why should it work at all?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    projected_digit = mo.ui.dropdown(
        options={str(_digit): _digit for _digit in range(10)},
        value="3",
        label="Digit",
    )
    projected_dimension = mo.ui.dropdown(
        options={
            str(_dimension): _dimension
            for _dimension in (25, 50, 100, 200, 400)
        },
        value="100",
        label="Output dimension k",
    )
    projected_sample_refresh = mo.ui.button(
        value=0,
        on_click=lambda _count: _count + 1,
        label="New sample",
    )
    projected_gaussian_refresh = mo.ui.button(
        value=0,
        on_click=lambda _count: _count + 1,
        label="Resample Gaussian map",
    )
    projected_digit_controls = mo.vstack(
        [
            mo.hstack(
                [projected_digit, projected_dimension],
                justify="start",
                align="center",
                gap=1,
            ),
            mo.hstack(
                [projected_sample_refresh, projected_gaussian_refresh],
                justify="start",
                align="center",
                gap=1,
            ),
        ],
        gap=0.4,
    )
    return (
        projected_digit,
        projected_digit_controls,
        projected_dimension,
        projected_gaussian_refresh,
        projected_sample_refresh,
    )


@app.cell(hide_code=True)
def _(
    PLOT_FONT_SIZES,
    RANDOM_SEED,
    X,
    lecture_note,
    mo,
    np,
    pca_components,
    pca_coordinates,
    pca_mean,
    plt,
    projected_digit,
    projected_digit_controls,
    projected_dimension,
    projected_gaussian_refresh,
    projected_sample_refresh,
    sns,
    y,
):
    _digit = projected_digit.value
    _dimension = projected_dimension.value
    _sample_count = projected_sample_refresh.value
    _gaussian_count = projected_gaussian_refresh.value

    _sample_rng = np.random.default_rng(
        RANDOM_SEED + 600_000 + 1_000 * _digit + _sample_count
    )
    _digit_indices = np.flatnonzero(y == _digit)
    _sample_indices = _sample_rng.choice(
        _digit_indices,
        size=5,
        replace=False,
    )

    _gaussian_rng = np.random.default_rng(
        RANDOM_SEED + 700_000 + 10_000 * _dimension + _gaussian_count
    )
    _gaussian_matrix = _gaussian_rng.normal(
        0,
        1 / np.sqrt(_dimension),
        size=(X.shape[1], _dimension),
    ).astype(np.float32)
    _projected_samples = X[_sample_indices] @ _gaussian_matrix
    _gaussian_reconstructions = np.clip(
        _projected_samples @ np.linalg.pinv(_gaussian_matrix),
        0,
        1,
    )
    _pca_reconstructions = np.clip(
        pca_mean
        + pca_coordinates[_sample_indices, :_dimension]
        @ pca_components[:_dimension],
        0,
        1,
    )

    _rows = (
        (X[_sample_indices], "Original"),
        (_gaussian_reconstructions, "Gaussian\n$P^\\dagger P\\mathbf{x}$"),
        (_pca_reconstructions, "PCA"),
    )
    _fig, _axes = plt.subplots(3, 5, figsize=(7.2, 4.5), dpi=120)
    for _row, (_samples, _row_label) in enumerate(_rows):
        for _sample, _axis in zip(_samples, _axes[_row]):
            sns.heatmap(
                _sample.reshape(28, 28),
                ax=_axis,
                cmap="gray_r",
                vmin=0,
                vmax=1,
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            _axis.set_xticks([])
            _axis.set_yticks([])
            _axis.set_xlabel("")
            for _spine in _axis.spines.values():
                _spine.set_visible(True)
                _spine.set_linewidth(1.0)
                _spine.set_edgecolor("0.35")
        _axes[_row, 0].set_ylabel(
            _row_label,
            fontsize=PLOT_FONT_SIZES["axis_label"],
            rotation=0,
            ha="right",
            va="center",
            labelpad=10,
        )

    _fig.suptitle(
        rf"Gaussian vs. PCA Reconstruction — $k={_dimension}$",
        fontsize=PLOT_FONT_SIZES["figure_title"],
    )
    _fig.tight_layout()
    _caption = lecture_note(
        r"The Gaussian row uses the Moore--Penrose reconstruction "
        r"$P^\dagger P\mathbf{x}$; the PCA row uses "
        r"$\boldsymbol{\mu}+U_kU_k^\mathsf{T}"
        r"(\mathbf{x}-\boldsymbol{\mu})$. PCA is optimized for this "
        r"reconstruction task, while JL concerns distances in "
        r"$\mathbb R^k$."
    )
    _gallery_items = [projected_digit_controls, _fig]
    if _caption is not None:
        _gallery_items.append(_caption)
    reconstruction_gallery = mo.vstack(
        _gallery_items,
        gap=0.25,
    )
    return (reconstruction_gallery,)


@app.cell(hide_code=True)
def _(mo):
    comparison_dimension = mo.ui.dropdown(
        options={
            str(_dimension): _dimension
            for _dimension in (25, 50, 100, 200, 400)
        },
        value="100",
        label="Output dimension",
    )
    gaussian_refresh = mo.ui.button(
        value=0,
        on_click=lambda _count: _count + 1,
        label="Resample Gaussian map",
    )
    comparison_controls = mo.hstack(
        [comparison_dimension, gaussian_refresh],
        justify="start",
        align="center",
        gap=1,
    )
    return comparison_controls, comparison_dimension, gaussian_refresh


@app.cell(hide_code=True)
def _(PCA, RANDOM_SEED, X, np):
    _pca = PCA(
        n_components=400,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )
    pca_coordinates = _pca.fit_transform(X)
    pca_components = _pca.components_
    pca_explained_variance = _pca.explained_variance_
    pca_mean = _pca.mean_
    total_variance = np.var(X, axis=0, ddof=1).sum()
    return (
        pca_components,
        pca_coordinates,
        pca_explained_variance,
        pca_mean,
        total_variance,
    )


@app.cell(hide_code=True)
def _(RANDOM_SEED, X, np):
    _rng = np.random.default_rng(RANDOM_SEED + 300_000)
    comparison_pair_i = _rng.integers(0, len(X), size=3_500)
    comparison_pair_j = _rng.integers(0, len(X), size=3_500)
    _distinct = comparison_pair_i != comparison_pair_j
    comparison_pair_i = comparison_pair_i[_distinct]
    comparison_pair_j = comparison_pair_j[_distinct]
    comparison_differences = (
        X[comparison_pair_i] - X[comparison_pair_j]
    )
    original_comparison_distances = np.linalg.norm(
        comparison_differences, axis=1
    )
    _nonzero = original_comparison_distances > 0
    comparison_pair_i = comparison_pair_i[_nonzero]
    comparison_pair_j = comparison_pair_j[_nonzero]
    comparison_differences = comparison_differences[_nonzero]
    original_comparison_distances = original_comparison_distances[_nonzero]
    return (
        comparison_differences,
        comparison_pair_i,
        comparison_pair_j,
        original_comparison_distances,
    )


@app.cell(hide_code=True)
def _(
    PLOT_FONT_SIZES,
    RANDOM_SEED,
    comparison_controls,
    comparison_differences,
    comparison_dimension,
    comparison_pair_i,
    comparison_pair_j,
    gaussian_refresh,
    latex_plot_ticks,
    lecture_note,
    mo,
    np,
    original_comparison_distances,
    pca_coordinates,
    pca_explained_variance,
    plt,
    sns,
    total_variance,
    visualization_block,
):
    _dimension = comparison_dimension.value
    _refresh_count = gaussian_refresh.value

    _retained_variance = pca_explained_variance[:_dimension].sum()
    _pca_scale = np.sqrt(total_variance / _retained_variance)
    _pca_differences = _pca_scale * (
        pca_coordinates[comparison_pair_i, :_dimension]
        - pca_coordinates[comparison_pair_j, :_dimension]
    )
    _pca_distances = np.linalg.norm(_pca_differences, axis=1)

    _rng = np.random.default_rng(
        RANDOM_SEED + 400_000 + _refresh_count
    )
    _gaussian_matrix = _rng.normal(
        0,
        1 / np.sqrt(_dimension),
        size=(comparison_differences.shape[1], _dimension),
    ).astype(np.float32)
    _gaussian_distances = np.linalg.norm(
        comparison_differences @ _gaussian_matrix, axis=1
    )

    _pca_errors = np.abs(
        _pca_distances / original_comparison_distances - 1
    )
    _gaussian_errors = np.abs(
        _gaussian_distances / original_comparison_distances - 1
    )

    _fig, _axes = plt.subplots(
        1,
        2,
        figsize=(7.6, 3.4),
        dpi=120,
        layout="constrained",
    )
    _fig.get_layout_engine().set(
        w_pad=0.04,
        h_pad=0.04,
        wspace=0.08,
        hspace=0.02,
    )
    _distance_limit = 1.05 * max(
        original_comparison_distances.max(),
        _pca_distances.max(),
        _gaussian_distances.max(),
    )
    _series = (
        (
            _axes[0],
            _pca_distances,
            _pca_errors,
            "PCA — learned from MNIST",
        ),
        (
            _axes[1],
            _gaussian_distances,
            _gaussian_errors,
            "Gaussian — no fitting",
        ),
    )
    for _axis, _distances, _errors, _title in _series:
        sns.scatterplot(
            x=original_comparison_distances,
            y=_distances,
            ax=_axis,
            s=12,
            alpha=0.20,
            linewidth=0,
            color=sns.color_palette("deep")[0],
        )
        _axis.plot(
            [0, _distance_limit],
            [0, _distance_limit],
            color="0.25",
            linewidth=1,
            linestyle="--",
        )
        _axis.text(
            0.04,
            0.96,
            f"median error: ${100 * np.median(_errors):.1f}\\%$\n"
            f"$90$th percentile: "
            f"${100 * np.quantile(_errors, 0.9):.1f}\\%$",
            transform=_axis.transAxes,
            va="top",
            fontsize=PLOT_FONT_SIZES["detail"],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 2,
            },
        )
        _axis.set(
            xlim=(0, _distance_limit),
            ylim=(0, _distance_limit),
            aspect="equal",
        )
        _axis.set_anchor("N")
        _axis.set_title(
            _title,
            fontsize=PLOT_FONT_SIZES["panel_title"],
            pad=7,
        )
        _axis.set_xlabel(
            "Original pair distance",
            fontsize=PLOT_FONT_SIZES["axis_label"],
            labelpad=4,
        )
        _axis.set_ylabel(
            "Projected pair distance",
            fontsize=PLOT_FONT_SIZES["axis_label"],
            labelpad=4,
        )
        _axis.tick_params(
            labelsize=PLOT_FONT_SIZES["detail"],
            length=3,
        )
        latex_plot_ticks(_axis)
        for _spine in _axis.spines.values():
            _spine.set_visible(True)
            _spine.set_linewidth(0.9)
            _spine.set_edgecolor("0.35")

    _fig.suptitle(
        rf"Learned vs. Oblivious Projection — $k={_dimension}$",
        fontsize=PLOT_FONT_SIZES["figure_title"],
    )
    _pair_count_latex = f"{len(original_comparison_distances):,}".replace(
        ",", "{,}"
    )
    _caption = lecture_note(
        f"The same **${_pair_count_latex}$ pairs** appear "
        "in both panels. PCA is globally rescaled to preserve total "
        "variance; the Gaussian map is chosen without inspecting MNIST."
    )
    _visualization_items = [comparison_controls, _fig]
    if _caption is not None:
        _visualization_items.append(_caption)
    visualization_block(
        mo.vstack(_visualization_items, gap=0.25)
    )
    return


@app.cell(hide_code=True)
def _(reconstruction_gallery, visualization_block):
    visualization_block(reconstruction_gallery)
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    The Gaussian random projection works surprisingly well despite not using any of the data itself. To understand why, forget the whole dataset and fix one nonzero vector $\mathbf{v} \in \mathbb{R}^d$. We first ask what the projection does to $\lVert \mathbf{v}\rVert_2$.
    """)
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    \[
    \begin{aligned}
    P &=
    \begin{bmatrix}
    \vert & \vert &        & \vert\\
    \mathbf{p}^{(1)} & \mathbf{p}^{(2)} & \cdots & \mathbf{p}^{(d)}\\
    \vert & \vert &        & \vert
    \end{bmatrix},
    &\qquad
    (P\mathbf{v})_j=\sum_{i=1}^d P_{ji}v_i,\\[4pt]
    \mathbb E[(P\mathbf{v})_j^2]
    &=\sum_{i=1}^d v_i^2\,\mathbb E[P_{ji}^2]
      =\frac{\lVert \mathbf{v}\rVert_2^2}{k},\\[4pt]
    \mathbb E[\lVert P\mathbf{v}\rVert_2^2]
    &=\sum_{j=1}^k\mathbb E[(P\mathbf{v})_j^2]
      =\lVert \mathbf{v}\rVert_2^2.
    \end{aligned}
    \]

    The cross terms vanish because the entries of $P$ are independent and have mean zero.
    """)
    return


@app.cell
def _(presentation_md):
    presentation_md(r"""
    The matrix defined by $P_{ji}\sim\mathcal N(0,1/k)$ preserves squared norm **in expectation**. We can even quantitatively show that $\lVert P v \rVert_2^2$ remain close to $\lVert v \rVert_2^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    concentration_dimension = mo.ui.dropdown(
        options={
            str(_dimension): _dimension
            for _dimension in (2, 5, 10, 25, 50, 100, 200)
        },
        value="25",
        label="Output dimension",
    )
    concentration_refresh = mo.ui.button(
        value=0,
        on_click=lambda _count: _count + 1,
        label="Resample experiment",
    )
    concentration_controls = mo.hstack(
        [concentration_dimension, concentration_refresh],
        justify="start",
        align="center",
        gap=1,
    )
    return (
        concentration_controls,
        concentration_dimension,
        concentration_refresh,
    )


@app.cell(hide_code=True)
def _(
    PLOT_FONT_SIZES,
    RANDOM_SEED,
    chi2,
    concentration_controls,
    concentration_dimension,
    concentration_refresh,
    latex_plot_ticks,
    lecture_note,
    mo,
    np,
    plt,
    sns,
    visualization_block,
):
    _dimension = concentration_dimension.value
    _refresh_count = concentration_refresh.value
    _rng = np.random.default_rng(
        RANDOM_SEED + 500_000 + _refresh_count
    )
    _gaussians = _rng.standard_normal((6_000, _dimension))
    _norm_ratios = np.mean(_gaussians**2, axis=1)

    _x = np.linspace(0.001, 3.0, 600)
    _density = _dimension * chi2.pdf(
        _dimension * _x, df=_dimension
    )

    _fig, _axis = plt.subplots(figsize=(7.2, 3.0), dpi=120)
    sns.histplot(
        _norm_ratios,
        bins=55,
        stat="density",
        ax=_axis,
        color=sns.color_palette("deep")[0],
        alpha=0.35,
        edgecolor=None,
        label="Simulation",
    )
    _axis.plot(
        _x,
        _density,
        color=sns.color_palette("deep")[3],
        linewidth=1.7,
        label=r"$\chi_k^2/k$ density",
    )
    _axis.axvline(
        1,
        color="0.25",
        linewidth=1,
        linestyle="--",
        label="Original squared norm",
    )
    _axis.set_xlim(0, 3)
    _axis.set_xlabel(
        r"Squared norm ratio  $\Vert P\mathbf{v}\Vert_2^2"
        r"/\Vert\mathbf{v}\Vert_2^2$",
        fontsize=PLOT_FONT_SIZES["axis_label"],
    )
    _axis.set_ylabel(
        "Density",
        fontsize=PLOT_FONT_SIZES["axis_label"],
    )
    _axis.set_title(
        rf"Gaussian Norm Concentration — $k={_dimension}$",
        fontsize=PLOT_FONT_SIZES["figure_title"],
        pad=9,
    )
    _axis.tick_params(
        labelsize=PLOT_FONT_SIZES["detail"],
        length=3,
    )
    latex_plot_ticks(_axis)
    _axis.legend(
        fontsize=PLOT_FONT_SIZES["detail"],
        frameon=False,
    )
    for _spine in _axis.spines.values():
        _spine.set_visible(True)
        _spine.set_linewidth(0.9)
        _spine.set_edgecolor("0.35")

    _fig.tight_layout()
    _caption = lecture_note(
        f"For a fixed $\\mathbf{{v}}$, "
        f"$\\lVert P\\mathbf{{v}}\\rVert_2^2 / "
        f"\\lVert \\mathbf{{v}}\\rVert_2^2 "
        f"\\sim \\chi^2_{{{_dimension}}}/"
        f"{_dimension}$. Its mean is $1$."
    )
    _visualization_items = [concentration_controls, _fig]
    if _caption is not None:
        _visualization_items.append(_caption)
    visualization_block(
        mo.vstack(_visualization_items, gap=0.25)
    )
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    We will use the following concentration inequality without proving it. For every fixed nonzero $\mathbf{v}\in\mathbb R^d$ and $0<\epsilon<1$,

    \[
    \Pr\!\left[
    \left|
    \frac{\lVert P\mathbf{v}\rVert_2^2}
         {\lVert \mathbf{v}\rVert_2^2}-1
    \right|>\epsilon
    \right]
    \leq 2\exp\!\left(-\frac{k\epsilon^2}{8}\right).
    \]

    Here $\epsilon$ is the squared-distance distortion we tolerate. The important feature is that the probability of failure for one fixed vector decreases **exponentially in $k\epsilon^2$**.
    """)
    return


@app.cell(hide_code=True)
def _(lecture_note, mo):
    concentration_notes = lecture_note(
        mo.accordion(
            {
                "Optional notes: deriving the concentration bound": mo.md(
                    r"""
                #### 1. Reduce to a chi-squared random variable

                For a fixed nonzero vector $\mathbf{v}$, define

                \[
                g_j=\frac{\sqrt{k}(P\mathbf{v})_j}
                {\lVert\mathbf{v}\rVert_2},
                \qquad j=1,\ldots,k.
                \]

                Each $g_j\sim\mathcal N(0,1)$, and the $g_j$ are independent
                because they come from independent rows of $P$. Consequently,

                \[
                Z:=\sum_{j=1}^k g_j^2
                =\frac{k\lVert P\mathbf{v}\rVert_2^2}
                {\lVert\mathbf{v}\rVert_2^2}
                \sim\chi_k^2.
                \]

                #### 2. Compute the MGF

                If $g\sim\mathcal N(0,1)$ and $\lambda<1/2$, then

                \[
                \begin{aligned}
                \mathbb E[e^{\lambda g^2}]
                &=\frac{1}{\sqrt{2\pi}}
                  \int_{-\infty}^{\infty}
                  e^{\lambda t^2}e^{-t^2/2}\,dt\\
                &=\frac{1}{\sqrt{2\pi}}
                  \int_{-\infty}^{\infty}
                  e^{-(1-2\lambda)t^2/2}\,dt\\
                &=(1-2\lambda)^{-1/2}.
                \end{aligned}
                \]

                The last equality is the Gaussian integral after the change
                of variables $u=\sqrt{1-2\lambda}\,t$. Independence now gives

                \[
                \mathbb E[e^{\lambda Z}]
                =\prod_{j=1}^k\mathbb E[e^{\lambda g_j^2}]
                =(1-2\lambda)^{-k/2}.
                \]

                #### 3. Optimize the upper tail

                For $0<\lambda<1/2$, Markov's inequality gives

                \[
                \begin{aligned}
                \Pr[Z\geq k(1+\epsilon)]
                &=\Pr[e^{\lambda Z}
                  \geq e^{\lambda k(1+\epsilon)}]\\
                &\leq
                e^{-\lambda k(1+\epsilon)}
                (1-2\lambda)^{-k/2}.
                \end{aligned}
                \]

                Taking logarithms turns the right-hand side into

                \[
                \phi_+(\lambda)
                =-\lambda k(1+\epsilon)
                -\frac{k}{2}\log(1-2\lambda).
                \]

                Its derivative vanishes when

                \[
                \phi_+'(\lambda)
                =-k(1+\epsilon)+\frac{k}{1-2\lambda}=0,
                \qquad
                \lambda_*=\frac{\epsilon}{2(1+\epsilon)}.
                \]

                Since $\phi_+''(\lambda)=2k/(1-2\lambda)^2>0$, this is the
                minimizer. Moreover,
                $1-2\lambda_*=1/(1+\epsilon)$. Using
                $\log(1/u)=-\log u$,

                \[
                \begin{aligned}
                \phi_+(\lambda_*)
                &=-\frac{k\epsilon}{2}
                  -\frac{k}{2}\log\!\left(\frac{1}{1+\epsilon}\right)\\
                &=-\frac{k}{2}
                  \bigl(\epsilon-\log(1+\epsilon)\bigr).
                \end{aligned}
                \]

                Therefore

                \[
                \Pr[Z\geq k(1+\epsilon)]
                \leq
                \exp\!\left[-\frac{k}{2}
                \bigl(\epsilon-\log(1+\epsilon)\bigr)\right].
                \]

                #### 4. Optimize the lower tail

                For $t>0$, apply Markov's inequality to $e^{-tZ}$:

                \[
                \Pr[Z\leq k(1-\epsilon)]
                \leq
                e^{tk(1-\epsilon)}(1+2t)^{-k/2}.
                \]

                The logarithm of this bound is

                \[
                \phi_-(t)=tk(1-\epsilon)
                -\frac{k}{2}\log(1+2t).
                \]

                Differentiating and setting the derivative to zero gives

                \[
                \phi_-'(t)
                =k(1-\epsilon)-\frac{k}{1+2t}=0,
                \qquad
                t_*=\frac{\epsilon}{2(1-\epsilon)},
                \qquad
                1+2t_*=\frac{1}{1-\epsilon}.
                \]

                Since $\phi_-''(t)=2k/(1+2t)^2>0$, this is the minimizer.
                Thus, using $\log(1/u)=-\log u$,

                \[
                \begin{aligned}
                \phi_-(t_*)
                &=\frac{k\epsilon}{2}
                  -\frac{k}{2}\log\!\left(\frac{1}{1-\epsilon}\right)\\
                &=-\frac{k}{2}
                  \bigl(-\epsilon-\log(1-\epsilon)\bigr),
                \end{aligned}
                \]

                and hence

                \[
                \Pr[Z\leq k(1-\epsilon)]
                \leq
                \exp\!\left[-\frac{k}{2}
                \bigl(-\epsilon-\log(1-\epsilon)\bigr)\right].
                \]

                #### 5. Simplify the logarithmic terms

                For $0<\epsilon<1$, the fundamental theorem of calculus gives

                \[
                \begin{aligned}
                \epsilon-\log(1+\epsilon)
                &=\int_0^\epsilon\frac{s}{1+s}\,ds
                \geq\int_0^\epsilon\frac{s}{2}\,ds
                =\frac{\epsilon^2}{4},\\[4pt]
                -\epsilon-\log(1-\epsilon)
                &=\int_0^\epsilon\frac{s}{1-s}\,ds
                \geq\int_0^\epsilon s\,ds
                =\frac{\epsilon^2}{2}.
                \end{aligned}
                \]

                Substituting these inequalities yields

                \[
                \Pr[Z\geq k(1+\epsilon)]\leq e^{-k\epsilon^2/8},
                \qquad
                \Pr[Z\leq k(1-\epsilon)]\leq e^{-k\epsilon^2/4}.
                \]

                Finally, adding the two tail probabilities gives

                \[
                \Pr\!\left[
                \left|\frac{Z}{k}-1\right|>\epsilon
                \right]
                \leq e^{-k\epsilon^2/8}+e^{-k\epsilon^2/4}
                \leq 2e^{-k\epsilon^2/8}.
                \]

                We used the standard identities $\log(a^r)=r\log a$ and
                $\log(1/u)=-\log u$; the integral identities above are what
                turn the exact Chernoff exponents into the simpler
                $e^{-\Theta(k\epsilon^2)}$ form.
                    """
                )
            }
        )
    )
    concentration_notes
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    For each pair of data points $i<j$, define the difference vector

    \[
    \mathbf{v}^{(i,j)}=\mathbf{x}^{(i)}-\mathbf{x}^{(j)}.
    \]

    Since $P$ is linear,

    \[
    \lVert P\mathbf{x}^{(i)}-P\mathbf{x}^{(j)}\rVert_2
    =\lVert P(\mathbf{x}^{(i)}-\mathbf{x}^{(j)})\rVert_2
    =\lVert P\mathbf{v}^{(i,j)}\rVert_2.
    \]

    Therefore, preserving every pairwise distance is exactly the problem of preserving the norms of all the difference vectors $\mathbf{v}^{(i,j)}$. There are

    \[
    m=\binom n2=\frac{n(n-1)}2
    \]

    of them. Notice that this is a finite list fixed before we sample $P$.
    """)
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    Let $B^{(i,j)}$ be the event that the distance between $\mathbf{x}^{(i)}$ and $\mathbf{x}^{(j)}$ is distorted by more than $\epsilon$. The **union bound** says

    \[
    \Pr\!\left[\text{at least one pair fails}\right]
    =\Pr\!\left[\bigcup_{i<j}B^{(i,j)}\right]
    \leq\sum_{i<j}\Pr[B^{(i,j)}].
    \]
    """)
    return


@app.cell(hide_code=True)
def _(lecture_note):
    lecture_note(r"""
    Combining our prior result with the union bound then gives us.

    \[
    \Pr[\text{at least one pair fails}]
    \leq
    2\binom n2\exp\!\left(-\frac{k\epsilon^2}{8}\right).
    \]

    To make this at most $\delta$, it is enough to choose

    \[
    k\geq
    \frac{8}{\epsilon^2}
    \log\!\left(\frac{2\binom n2}{\delta}\right)
    =O\!\left(\frac{\log(n/\delta)}{\epsilon^2}\right).
    \]

    There are quadratically many "bad events", but the probability decays exponentially in $k$.
    """)
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    ### Johnson--Lindenstrauss Lemma
    """)
    return


@app.cell(hide_code=True)
def _(presentation_md):
    presentation_md(r"""
    Let $\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(n)}\in\mathbb R^d$ and $0<\epsilon,\delta<1$. If

    \[
    P_{ab}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1/k)
    \qquad\text{and}\qquad
    k\geq C\epsilon^{-2}\log(n/\delta),
    \]

    then, with probability at least $1-\delta$, every pair $i<j$ satisfies

    \[
    (1-\epsilon)\lVert \mathbf{x}^{(i)}-\mathbf{x}^{(j)}\rVert_2^2
    \leq
    \lVert P\mathbf{x}^{(i)}-P\mathbf{x}^{(j)}\rVert_2^2
    \leq
    (1+\epsilon)\lVert \mathbf{x}^{(i)}-\mathbf{x}^{(j)}\rVert_2^2.
    \]
    """)
    return


if __name__ == "__main__":
    app.run()
