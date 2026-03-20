"""
KazeFlow - FAISS Index Builder.

Reads SPIN v2 embeddings from logs/<model>/spin/*.npy and builds an
IVFFlat FAISS index for speaker feature retrieval during inference.

Algorithm choices:
  Auto   — use MiniBatchKMeans to compress to 10k centroids when N > 200k,
            then build IVFFlat on the centroids.
  Faiss  — feed all raw vectors directly into IVFFlat (no KMeans).
  KMeans — always run MiniBatchKMeans even when N <= 200k.
"""

import logging
import os
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

logger = logging.getLogger("kazeflow.train.build_index")


def build_index(
    exp_dir: str,
    index_algorithm: str = "Auto",
) -> str:
    """
    Build a FAISS IVFFlat index from SPIN embeddings in exp_dir/spin/.

    Args:
        exp_dir:          Path to the experiment directory (e.g. logs/mymodel).
        index_algorithm:  "Auto" | "Faiss" | "KMeans".

    Returns:
        Path to the written .index file.

    Raises:
        ImportError:  if faiss is not installed.
        FileNotFoundError: if the spin feature directory is empty.
        ValueError:   if index_algorithm is not recognised.
    """
    try:
        import faiss
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        raise ImportError(
            "faiss and scikit-learn are required for index building. "
            "Install them with: pip install faiss-cpu scikit-learn"
        ) from e

    if index_algorithm not in ("Auto", "Faiss", "KMeans"):
        raise ValueError(
            f"Unknown index_algorithm {index_algorithm!r}. "
            "Choose 'Auto', 'Faiss', or 'KMeans'."
        )

    exp_dir = Path(exp_dir)
    feature_dir = exp_dir / "spin"

    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Feature directory not found: {feature_dir}\n"
            "Run Preprocess & Extract Features first."
        )

    # ── Load all embeddings ───────────────────────────────────────────────
    npy_files = sorted(
        f for f in os.listdir(feature_dir) if f.endswith(".npy")
    )
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {feature_dir}. "
            "Run Preprocess & Extract Features first."
        )

    logger.info("Loading %d embedding files from %s", len(npy_files), feature_dir)
    npys = []
    for name in npy_files:
        arr = np.load(feature_dir / name)   # shape: (T, 768)
        npys.append(arr.astype("float32"))
    big_npy = np.concatenate(npys, axis=0)  # (N, 768)
    logger.info("Total vectors: %d  dim: %d", big_npy.shape[0], big_npy.shape[1])

    # ── Shuffle ───────────────────────────────────────────────────────────
    idx = np.arange(big_npy.shape[0])
    np.random.shuffle(idx)
    big_npy = big_npy[idx]

    # ── Optional KMeans compression ───────────────────────────────────────
    use_kmeans = (
        index_algorithm == "KMeans"
        or (index_algorithm == "Auto" and big_npy.shape[0] > 200_000)
    )
    if use_kmeans:
        n_clusters = min(10_000, big_npy.shape[0])
        logger.info(
            "Running MiniBatchKMeans (n_clusters=%d, N=%d)...",
            n_clusters, big_npy.shape[0],
        )
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            verbose=True,
            batch_size=256 * cpu_count(),
            compute_labels=False,
            init="random",
        )
        big_npy = km.fit(big_npy).cluster_centers_.astype("float32")
        logger.info("KMeans done — compressed to %d centroids", big_npy.shape[0])

    # ── Build IVFFlat index ───────────────────────────────────────────────
    dim = big_npy.shape[1]   # 768
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)
    logger.info("Building IVF%d,Flat index (dim=%d, N=%d)", n_ivf, dim, big_npy.shape[0])

    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    ivf = faiss.extract_index_ivf(index)
    ivf.nprobe = 1

    logger.info("Training index...")
    index.train(big_npy)

    logger.info("Adding vectors (batch size 8192)...")
    for i in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[i : i + 8192])

    # ── Save ──────────────────────────────────────────────────────────────
    model_name = exp_dir.name
    out_path = exp_dir / f"{model_name}.index"
    faiss.write_index(index, str(out_path))
    logger.info("Index saved → %s  (%d vectors)", out_path, index.ntotal)

    return str(out_path)
