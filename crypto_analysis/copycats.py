"""Copycats: coins whose image repeats, exactly or as a CLIP near-duplicate, an image launched before them.

Two images are near-duplicates when the cosine of their CLIP ViT-B/32 image embeddings is at least ``THRESHOLD``
(0.95). In a 100k-image sample (notebook 04), 12% of images had such a neighbour, and half of those were not pHash
near-duplicates: the same image redrawn, recoloured or re-posed. Neighbours are direct (no chaining), so a coin is
compared only with images within 0.95 of its own.

For every distinct image (``sha256``) of a coin with a topic: the first launch that used it, its creator and its
first graduation. A GPU pass over all images then finds, for each image, the earliest first launch and the earliest
graduation among its near-duplicates (other ``sha256`` within ``THRESHOLD``). A coin launched at time t with image i is

* an **exact copy** when an earlier coin used the same ``sha256``;
* a **near copy** when it is not an exact copy and a near-duplicate of i was launched before t;
* an **original** otherwise.

A copy is a **copy of a graduate** when the same image or one of its near-duplicates had graduated before t, and a
**self-copy** when its creator launched the earliest source image (the same image for an exact copy, the earliest
near-duplicate for a near copy). ``role`` follows ``datalake.build_token_flags``: original, self-relaunch, copy
(other creator), copy of a graduate (which takes precedence).

Outputs in ``<out>/copycats/``: ``images.parquet`` (sha256, coins, first launch, creator and graduation; neighbours
within the threshold, highest cosine, earliest neighbour launch and graduation) and ``coins.parquet`` (mint, sha256,
created_at, creator, graduated, kind, role, self_copy, after_graduation, source_sha256). Usage::

    export PUMPFUN_DATA_DIR=/path/to/pumpfun
    python -m crypto_analysis.copycats --out-dir data/topics --clip-dir data/clip
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

THRESHOLD = 0.95
CHUNK = 256


def copycat_dir(out_dir):
    return Path(out_dir) / "copycats"


def coin_table(out_dir):
    """Coins with a topic and an image: mint, sha256, created_at, creator, grad_time (NaN if never)."""
    from crypto_analysis.datalake import DATA_DIR, connect

    coins = pd.read_parquet(Path(out_dir) / "token_topics.parquet", columns=["mint", "sha256", "multimodal_topic_reduced"])
    coins = coins.dropna(subset=["sha256", "multimodal_topic_reduced"])[["mint", "sha256"]]
    con = connect(DATA_DIR, threads=8)
    con.register("wanted", coins)
    df = con.execute("""
        SELECT w.mint, w.sha256, t.created_at, t.coin_creator AS creator, g.grad_time
        FROM wanted w JOIN tokens t USING (mint)
        LEFT JOIN (SELECT mint, min(block_time) AS grad_time FROM graduations GROUP BY 1) g USING (mint)
    """).df()
    df["grad_time"] = df.grad_time.astype(float)
    return df


def image_table(coins):
    """One row per image: coins, first launch, its creator and the first graduation of a coin using it."""
    first = coins.sort_values(["created_at", "mint"]).drop_duplicates("sha256")
    images = coins.groupby("sha256").agg(coins=("mint", "size"), first_grad=("grad_time", "min"))
    images = images.join(first.set_index("sha256")[["created_at", "creator"]].rename(
        columns={"created_at": "first_created", "creator": "first_creator"}))
    return images.reset_index()


def neighbours(images, clip_dir, threshold=THRESHOLD, chunk=CHUNK):
    """For each image, among the other images within ``threshold`` cosine: how many, the highest cosine, the earliest
    first launch (and which image) and the earliest graduation."""
    import torch

    from crypto_analysis.clip_embeddings import load_embeddings

    keys, emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    rows = pd.Index(keys).get_indexer(images.sha256)
    if (rows < 0).any():
        print(f"{int((rows < 0).sum()):,} images have no CLIP embedding; they are left out", flush=True)
    images = images[rows >= 0].reset_index(drop=True)
    X = emb[rows[rows >= 0]].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    dev = torch.device("cuda")
    E = torch.from_numpy(X.astype(np.float16)).to(dev)
    t0 = images.first_created.min()
    inf = torch.tensor(float("inf"), device=dev)
    # times relative to the first launch, in float32 (2-second resolution over the period)
    created = torch.from_numpy((images.first_created - t0).to_numpy(np.float32)).to(dev)
    grad = torch.from_numpy((images.first_grad - t0).fillna(np.inf).to_numpy(np.float32)).to(dev)
    n = len(images)
    count = np.empty(n, np.int64)
    best = np.empty(n, np.float32)
    nb_created = np.empty(n, np.float64)
    nb_source = np.empty(n, np.int64)
    nb_grad = np.empty(n, np.float64)
    start = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = E[s:e] @ E.T
        sims[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = -1  # not its own neighbour
        mask = sims >= threshold
        count[s:e] = torch.count_nonzero(mask, dim=1).cpu().numpy()
        best[s:e] = sims.max(1).values.float().cpu().numpy()
        v, k = torch.where(mask, created[None, :], inf).min(1)
        nb_created[s:e], nb_source[s:e] = v.cpu().numpy(), k.cpu().numpy()
        nb_grad[s:e] = torch.where(mask, grad[None, :], inf).min(1).values.cpu().numpy()
        if (s // chunk) % 2000 == 0:
            print(f"  {e:,}/{n:,} images, {time.time() - start:.0f} s", flush=True)
    has = np.isfinite(nb_created)
    images["neighbours"] = count
    images["max_cosine"] = best
    images["nb_first_created"] = np.where(has, nb_created + t0, np.nan)
    images["nb_source_sha256"] = np.where(has, images.sha256.to_numpy()[nb_source], None)
    images["nb_first_grad"] = np.where(np.isfinite(nb_grad), nb_grad + t0, np.nan)
    return images


def classify(coins, images):
    """Kind (original, exact copy, near copy), self_copy, after_graduation and role of every coin."""
    img = images.set_index("sha256")
    c = coins.join(img[["first_created", "first_creator", "first_grad", "nb_first_created", "nb_source_sha256",
                        "nb_first_grad"]], on="sha256", how="inner")
    t = c.created_at
    exact = t > c.first_created
    near = ~exact & (c.nb_first_created < t)
    c["kind"] = np.select([exact, near], ["exact copy", "near copy"], "original")
    c["source_sha256"] = np.where(exact, c.sha256, np.where(near, c.nb_source_sha256, None))
    source_creator = c.source_sha256.map(img.first_creator)
    c["self_copy"] = (exact | near) & (c.creator == source_creator)
    first_grad = np.fmin(c.first_grad.to_numpy(), c.nb_first_grad.to_numpy())
    c["after_graduation"] = (exact | near) & (first_grad < t)
    c["role"] = np.select([c.kind == "original", c.after_graduation, c.self_copy],
                          ["original", "copy of a graduate", "self-relaunch"], "copy (other creator)")
    c["graduated"] = c.grad_time.notna()
    return c[["mint", "sha256", "created_at", "creator", "graduated", "kind", "role", "self_copy", "after_graduation",
              "source_sha256"]]


BINS = [(0.95, 0.96), (0.96, 0.98), (0.98, 1.01)]


def pairs_figure(out_dir, clip_dir, path, per_bin=9, seed=42):
    """Validation grid: near copies and their source image, ``per_bin`` random pairs per cosine bin (``BINS``), the
    coin names under each image (slurs masked); images the CLIP screen reads as nudity, gore or hate symbols are
    skipped. Also writes ``pairs.csv`` next to ``path``."""
    from crypto_analysis.clip_embeddings import load_embeddings
    from crypto_analysis.narrative_figures import EXCLUDED_IMAGES, _screen, image_names, panel_grid

    out = copycat_dir(out_dir)
    coins = pd.read_parquet(out / "coins.parquet", columns=["sha256", "kind", "source_sha256"])
    pairs = coins[coins.kind == "near copy"].drop_duplicates("sha256")[["sha256", "source_sha256"]]
    keys, emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    index = pd.Index(keys)
    rng = np.random.default_rng(seed)
    pairs = pairs.iloc[rng.permutation(len(pairs))[:200_000]].reset_index(drop=True)
    a = emb[index.get_indexer(pairs.sha256)].astype(np.float32)
    b = emb[index.get_indexer(pairs.source_sha256)].astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    pairs["cosine"] = (a * b).sum(1)
    pairs = pairs[~_screen(a) & ~_screen(b) & ~pairs.sha256.isin(EXCLUDED_IMAGES) & ~pairs.source_sha256.isin(EXCLUDED_IMAGES)]
    picked = pd.concat([pairs[(pairs.cosine >= lo) & (pairs.cosine < hi)].head(per_bin).assign(bin=f"{lo:.2f}-{min(hi, 1):.2f}")
                        for lo, hi in BINS], ignore_index=True)
    names = image_names(list(picked.sha256) + list(picked.source_sha256))
    picked["name"], picked["source_name"] = picked.sha256.map(names), picked.source_sha256.map(names)
    picked.to_csv(Path(path).with_name("copycat_pairs.csv"), index=False)
    panels = []
    for bin_label, grp in picked.groupby("bin", sort=False):
        for first in range(0, len(grp), 3):
            chunk = grp.iloc[first:first + 3]
            images = [(r.source_sha256, f"source: {r.source_name or ''}", False) for r in chunk.itertuples()]
            images += [(r.sha256, f"cos {r.cosine:.3f}: {r.name or ''}", False) for r in chunk.itertuples()]
            panels.append({"title": f"cosine {bin_label}", "subtitle": "top: earlier source; bottom: near copy",
                           "images": images})
    panel_grid(panels, path, cols=3)


def build(out_dir, clip_dir, threshold=THRESHOLD):
    out = copycat_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coins = coin_table(out_dir)
    images = neighbours(image_table(coins), clip_dir, threshold)
    images.to_parquet(out / "images.parquet", index=False)
    classified = classify(coins, images)
    classified.to_parquet(out / "coins.parquet", index=False)
    share = classified.kind.value_counts(normalize=True)
    print(f"{len(images):,} images ({(images.neighbours > 0).mean():.1%} with a neighbour >= {threshold}); "
          f"{len(classified):,} coins: " + ", ".join(f"{k} {v:.1%}" for k, v in share.items()), flush=True)
    print(classified.groupby("role").graduated.agg(["size", "mean"]).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--pairs", type=Path, help="only write the validation grid of near-copy pairs to this PNG")
    args = parser.parse_args()
    if args.pairs:
        pairs_figure(args.out_dir, args.clip_dir, args.pairs)
    else:
        build(args.out_dir, args.clip_dir, args.threshold)


if __name__ == "__main__":
    main()
