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


def resolve_cache_paths(cache_dir: str, backend: str, model: str, pretrained: str) -> str:
    cache_tag = f"{backend}_{model}_{pretrained}"
    real_cache = os.path.join(cache_dir, f"real_{cache_tag}.npz")
    legacy_real = os.path.join(cache_dir, f"real_{model}_{pretrained}.npz")
    if os.path.isfile(real_cache):
        return real_cache
    if backend == "clip" and os.path.isfile(legacy_real):
        return legacy_real
    raise FileNotFoundError(f"Missing real cache: {real_cache}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (x / norms).astype(np.float32, copy=False)


def dedupe_range_search(real: np.ndarray, threshold: float) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Use FAISS range search (cosine via IP on normalized vectors) to build duplicate groups.
    Greedy grouping: take i as representative, assign all neighbors >= threshold not yet assigned.
    Returns list of groups (indices) and parallel list of similarities to representative.
    """
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError("faiss is required: pip install faiss-cpu") from exc

    n, d = real.shape
    index = faiss.IndexFlatIP(d)
    index.add(real)
    # Range search
    lims, D, I = index.range_search(real, threshold)
    groups: List[List[int]] = []
    group_sims: List[List[float]] = []
    assigned = np.zeros(n, dtype=bool)
    for i in range(n):
        if assigned[i]:
            continue
        start, end = lims[i], lims[i + 1]
        neigh_idx = I[start:end]
        neigh_sim = D[start:end]
        # Include only neighbors not yet assigned, and avoid duplicates; keep i first
        group = [i]
        sims = [1.0]
        for j, s in zip(neigh_idx, neigh_sim):
            if j == i:
                continue
            if s >= threshold and not assigned[j]:
                group.append(int(j))
                sims.append(float(s))
        for j in group:
            assigned[j] = True
        groups.append(group)
        group_sims.append(sims)
    return groups, group_sims


def write_outputs(out_dir: str, paths: List[str], groups: List[List[int]], group_sims: List[List[float]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    keep = []
    remove = []
    records = []
    for gid, (idxs, sims) in enumerate(zip(groups, group_sims), start=1):
        rep = idxs[0]
        rep_path = paths[rep]
        keep.append(rep_path)
        for k, (j, s) in enumerate(zip(idxs, sims)):
            records.append(
                {
                    "group_id": gid,
                    "representative": rep_path,
                    "member_path": paths[j],
                    "similarity_to_rep": float(s),
                    "is_representative": int(k == 0),
                }
            )
            if k > 0:
                remove.append(paths[j])
    pd.DataFrame.from_records(records).to_csv(os.path.join(out_dir, "dedupe_groups.csv"), index=False)
    with open(os.path.join(out_dir, "dedupe_keep.txt"), "w") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
    with open(os.path.join(out_dir, "dedupe_remove.txt"), "w") as f:
        f.write("\n".join(remove) + ("\n" if remove else ""))


def main(argv: List[str]) -> None:
    p = argparse.ArgumentParser(description="Near-duplicate removal using cached embeddings and FAISS range search.")
    p.add_argument("--backend", type=str, default="sscd", choices=["clip", "dinov2", "sscd"], help="Embedding backend used for cache")
    p.add_argument("--model", type=str, required=True, help="Model tag (for sscd TorchScript, provide path to .pt used to build cache)")
    p.add_argument("--pretrained", type=str, default="openai", help="Pretrained tag (clip only; kept for cache key)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory where embeddings live and results will be written")
    p.add_argument("--cache_dir", type=str, default=None, help="Embeddings cache directory (defaults to out_dir/embeddings)")
    p.add_argument("--threshold", type=float, default=0.75, help="Cosine similarity threshold to consider duplicates")
    args = p.parse_args(argv)

    cache_dir = args.cache_dir or os.path.join(args.out_dir, "embeddings")
    real_cache = resolve_cache_paths(cache_dir, args.backend, args.model, args.pretrained)
    real_embeds, real_paths = load_embeddings(real_cache)
    # Ensure normalized (should already be)
    real_embeds = l2_normalize(real_embeds)

    groups, group_sims = dedupe_range_search(real_embeds, args.threshold)
    write_outputs(args.out_dir, real_paths, groups, group_sims)
    print(f"Done. Wrote dedupe_groups.csv, dedupe_keep.txt, dedupe_remove.txt to {args.out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])


