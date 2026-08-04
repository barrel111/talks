"""Download MNIST and create the deterministic subset used by the talk."""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "scikit_learn"
OUTPUT_PATH = PROJECT_ROOT / "data" / "mnist_10k.npz"
SUBSET_SIZE = 10_000
RANDOM_SEED = 42


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"MNIST subset already exists: {OUTPUT_PATH}")
        return

    print("Fetching MNIST from OpenML (the download is cached locally) ...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=CACHE_DIR,
        parser="auto",
    )

    images = np.asarray(mnist.data, dtype=np.uint8)
    labels = np.asarray(mnist.target, dtype=np.uint8)
    indices = np.arange(images.shape[0])
    subset_indices, _ = train_test_split(
        indices,
        train_size=SUBSET_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        images=images[subset_indices],
        labels=labels[subset_indices],
        source=np.array("OpenML mnist_784 version 1"),
        random_seed=np.array(RANDOM_SEED),
    )
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(
        f"Saved {SUBSET_SIZE:,} examples to {OUTPUT_PATH} "
        f"({size_mb:.1f} MiB)."
    )


if __name__ == "__main__":
    main()
