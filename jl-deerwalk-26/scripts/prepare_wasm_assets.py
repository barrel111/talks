"""Build the browser-ready MNIST and PCA bundle used by the WASM app."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "mnist_10k.npz"
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "public" / "mnist_10k.npz"
PCA_COMPONENTS = 400
RANDOM_SEED = 42


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_PATH}. Run scripts/prepare_mnist.py first."
        )

    with np.load(SOURCE_PATH) as mnist:
        images = mnist["images"]
        labels = mnist["labels"]

    features = images.astype(np.float32) / 255.0
    print(f"Fitting {PCA_COMPONENTS}-component PCA for the browser bundle ...")
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )
    coordinates = pca.fit_transform(features).astype(np.float32)
    total_variance = np.var(features, axis=0, ddof=1).sum()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        images=images,
        labels=labels,
        pca_coordinates=coordinates,
        pca_components=pca.components_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        total_variance=np.asarray(total_variance, dtype=np.float32),
        source=np.array("OpenML mnist_784 version 1; PCA precomputed for WASM"),
        random_seed=np.array(RANDOM_SEED),
    )
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved browser bundle to {OUTPUT_PATH} ({size_mb:.1f} MiB).")


if __name__ == "__main__":
    main()
