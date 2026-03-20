"""
FAISS Index for KazeFlow speaker feature retrieval.

Builds an IVF,Flat index from SPIN v2 (768-dim HuBERT) embeddings
and retrieves nearest neighbors at inference time to improve
speaker similarity.

Workflow:
1. build_index(): Collect SPIN features from training data → build FAISS index
2. retrieve_and_blend(): At inference, blend source SPIN features with
   retrieved target speaker features using index_rate

Ported from RVC's FAISS pipeline with adaptations for KazeFlow's
Flow Matching architecture.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
import torch

logger = logging.getLogger("kazeflow.infer.index")


def build_index(
    feature_dir: str,
    output_path: str,
    feature_dim: int = 768,
) -> str:
    """
    Build a FAISS IVF,Flat index from extracted SPIN v2 features.

    Args:
        feature_dir: Directory containing .npy files with SPIN features.
                     Each file shape: (T_frames, 768)
        output_path: Path to save the .index file
        feature_dim: Embedding dimension (768 for SPIN v2)

    Returns:
        Path to saved index file
    """
    # Collect all features
    npy_files = sorted(Path(feature_dir).glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {feature_dir}")

    features_list = []
    for f in npy_files:
        feat = np.load(str(f)).astype(np.float32)
        if feat.ndim == 3:
            # (1, dim, T) → (T, dim)
            feat = feat.squeeze(0).T
        elif feat.ndim == 2 and feat.shape[1] != feature_dim:
            feat = feat.T
        features_list.append(feat)

    big_npy = np.concatenate(features_list, axis=0)
    n_samples = big_npy.shape[0]
    logger.info(f"Collected {n_samples} feature vectors from {len(npy_files)} files")

    # IVF cluster count: RVC formula
    n_ivf = min(int(16 * np.sqrt(n_samples)), n_samples // 39)
    n_ivf = max(n_ivf, 1)

    logger.info(f"Building IVF{n_ivf},Flat index (dim={feature_dim})")

    index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
    index.train(big_npy)
    index.add(big_npy)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    faiss.write_index(index, output_path)
    logger.info(f"Index saved to {output_path} ({index.ntotal} vectors)")

    return output_path


def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from file."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    index = faiss.read_index(index_path)
    logger.info(f"Loaded index from {index_path} ({index.ntotal} vectors)")
    return index


def retrieve_and_blend(
    source_features: torch.Tensor,
    index: faiss.Index,
    index_rate: float = 0.5,
    k: int = 8,
    temperature: float = 0.25,
) -> torch.Tensor:
    """
    Retrieve nearest neighbors from index and blend with source features.

    Uses cosine-similarity softmax weighting (same as RVC):
    - Search k nearest neighbors
    - Compute cosine similarity between query and neighbors  
    - Softmax with temperature τ to get weights
    - Weighted average of neighbor features
    - Blend: output = (1 - index_rate) * source + index_rate * retrieved

    Args:
        source_features: (1, 768, T) source SPIN features
        index: FAISS index containing target speaker features
        index_rate: Blending factor (0.0 = no retrieval, 1.0 = full retrieval)
        k: Number of nearest neighbors
        temperature: Softmax temperature (lower = sharper)

    Returns:
        blended_features: (1, 768, T) blended features
    """
    if index_rate <= 0.0 or index is None:
        return source_features

    device = source_features.device
    # (1, 768, T) → (T, 768)
    feats_np = source_features.squeeze(0).T.cpu().numpy().astype(np.float32)
    T, dim = feats_np.shape

    # Clamp k to index size
    k_actual = min(k, index.ntotal)

    # Search
    index.nprobe = min(int(np.sqrt(index.ntotal)), 64)
    distances, indices = index.search(feats_np, k_actual)
    # distances: (T, k), indices: (T, k)

    # Retrieve neighbor features
    # Reconstruct vectors from index
    neighbors = np.zeros((T, k_actual, dim), dtype=np.float32)
    for i in range(T):
        for j in range(k_actual):
            idx = indices[i, j]
            if idx >= 0:
                neighbors[i, j] = index.reconstruct(int(idx))

    # Cosine similarity weighting
    # Normalize query and neighbors
    query_norm = feats_np / (np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-8)
    neighbors_norm = neighbors / (np.linalg.norm(neighbors, axis=2, keepdims=True) + 1e-8)

    # (T, k) cosine similarities
    cos_sim = np.einsum("td,tkd->tk", query_norm, neighbors_norm)

    # Softmax with temperature
    cos_sim_scaled = cos_sim / temperature
    cos_sim_scaled -= cos_sim_scaled.max(axis=1, keepdims=True)  # numerical stability
    exp_sim = np.exp(cos_sim_scaled)
    weights = exp_sim / (exp_sim.sum(axis=1, keepdims=True) + 1e-8)  # (T, k)

    # Weighted average of neighbor features
    retrieved = np.einsum("tk,tkd->td", weights, neighbors)  # (T, dim)

    # Blend
    retrieved_tensor = torch.from_numpy(retrieved).float().to(device)
    source_flat = source_features.squeeze(0).T  # (T, dim)
    blended = (1.0 - index_rate) * source_flat + index_rate * retrieved_tensor

    return blended.T.unsqueeze(0)  # (1, dim, T)
