import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_embeddings(cache_path: str) -> Tuple[np.ndarray, List[str]]:
    data = np.load(cache_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32, copy=False)
    paths = [str(p) for p in data["paths"].tolist()]
    return embeddings, paths


def resolve_cache_paths(cache_dir: str, backend: str, model: str, pretrained: str) -> Tuple[str, str]:
    cache_tag = f"{backend}_{model}_{pretrained}"
    unique_cache = os.path.join(cache_dir, f"unique_{cache_tag}.npz")
    real_cache = os.path.join(cache_dir, f"real_{cache_tag}.npz")
    # Legacy CLIP cache fallback (no backend prefix)
    legacy_unique = os.path.join(cache_dir, f"unique_{model}_{pretrained}.npz")
    legacy_real = os.path.join(cache_dir, f"real_{model}_{pretrained}.npz")
    if not os.path.isfile(unique_cache):
        if backend == "clip" and os.path.isfile(legacy_unique):
            unique_cache = legacy_unique
        else:
            raise FileNotFoundError(f"Missing unique cache: {unique_cache}")
    if not os.path.isfile(real_cache):
        if backend == "clip" and os.path.isfile(legacy_real):
            real_cache = legacy_real
        else:
            raise FileNotFoundError(f"Missing real cache: {real_cache}")
    return unique_cache, real_cache


def compute_argmax_clusters(real: np.ndarray, unique: np.ndarray, chunk_size: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
    """
    real: [N, D] L2-normalized
    unique: [M, D] L2-normalized
    Returns:
      - best_idx: [N] index of nearest centroid by cosine similarity
      - best_sim: [N] corresponding similarity value
    """
    N, D = real.shape
    M, D2 = unique.shape
    assert D == D2, "Embedding dims must match"
    best_idx = np.empty(N, dtype=np.int64)
    best_sim = np.empty(N, dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sims = real[start:end] @ unique.T  # [chunk, M]
        idx = sims.argmax(axis=1)
        val = sims[np.arange(sims.shape[0]), idx]
        best_idx[start:end] = idx
        best_sim[start:end] = val
    return best_idx, best_sim


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Assign each real image to the nearest unique centroid using cached embeddings (cosine argmax)."
    )
    parser.add_argument("--backend", type=str, default="clip", choices=["clip", "dinov2", "sscd"], help="Embedding backend used for caches")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="Model tag (for sscd TorchScript, provide full path to .pt)")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained tag (clip only; kept for cache key)")
    parser.add_argument("--out_dir", type=str, default="/workspace/clip_results", help="Pipeline output directory")
    parser.add_argument("--cache_dir", type=str, default=None, help="Embeddings cache directory (defaults to out_dir/embeddings)")
    parser.add_argument("--counts_only", action="store_true", help="Also output per-cluster counts CSV")
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or os.path.join(args.out_dir, "embeddings")
    unique_cache, real_cache = resolve_cache_paths(cache_dir, args.backend, args.model, args.pretrained)

    unique_embeds, unique_paths = load_embeddings(unique_cache)
    real_embeds, real_paths = load_embeddings(real_cache)

    # Compute nearest centroid per real image
    best_idx, best_sim = compute_argmax_clusters(real_embeds, unique_embeds)
    best_unique_paths = [unique_paths[i] for i in best_idx.tolist()]
    best_unique_basenames = [os.path.basename(p) for p in best_unique_paths]

    # Write assignments CSV
    os.makedirs(args.out_dir, exist_ok=True)
    assign_csv = os.path.join(args.out_dir, "cluster_assignments.csv")
    df = pd.DataFrame({
        "image_path": real_paths,
        "cluster_unique_image": best_unique_paths,
        "cluster_unique_basename": best_unique_basenames,
        "similarity": best_sim.astype(np.float32),
        "cluster_index": best_idx.astype(np.int64),
    })
    df.to_csv(assign_csv, index=False)
    print(f"Wrote cluster assignments to: {assign_csv}")

    if args.counts_only:
        counts = df.groupby(["cluster_unique_image", "cluster_unique_basename"]).size().reset_index(name="count")
        counts_csv = os.path.join(args.out_dir, "cluster_counts.csv")
        counts.to_csv(counts_csv, index=False)
        print(f"Wrote cluster counts to: {counts_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])


