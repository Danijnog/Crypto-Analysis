"""Per-image features from the stored WebP blobs: colour statistics and a perceptual hash.

The blob shards (``$IMAGES_DIR/shard_NNNNNN.parquet``) hold one row per distinct image
(``sha256``, ``bytes``). ``shard_features`` processes one shard file and is a top-level function
so it can be mapped over shards with a ``ProcessPoolExecutor``.
"""

import io

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

_PHASH_SIZE = 32
_k = np.arange(_PHASH_SIZE)
# Orthonormal DCT-II matrix, so the 2-D DCT of X is _DCT @ X @ _DCT.T
_DCT = np.sqrt(2 / _PHASH_SIZE) * np.cos(np.pi * (2 * _k[None, :] + 1) * _k[:, None] / (2 * _PHASH_SIZE))
_DCT[0] /= np.sqrt(2)
_BIT_WEIGHTS = np.left_shift(np.uint64(1), np.arange(63, -1, -1, dtype=np.uint64))


def _bits_to_int(bits):
    return int((bits.astype(np.uint64) * _BIT_WEIGHTS).sum())


def phash(gray):
    """64-bit perceptual hash: sign of the 8x8 low-frequency DCT block of a 32x32 grayscale image
    relative to its median (DC term excluded from the median).

    Splitting at the median sets ~32 bits, so Hamming distances between hashes are almost always even.
    """
    px = np.asarray(gray.resize((_PHASH_SIZE, _PHASH_SIZE), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT @ px @ _DCT.T)[:8, :8].ravel()
    return _bits_to_int(low > np.median(low[1:]))


def color_features(img):
    """Brightness, saturation, Hasler-Süsstrunk colourfulness, luminance contrast and background
    share (fraction of pixels within 12 grey levels of the median, high for flat backgrounds)."""
    thumb = img.resize((64, 64), Image.Resampling.BILINEAR)
    rgb = np.asarray(thumb, dtype=np.float32).reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32).reshape(-1, 3) / 255
    gray = np.asarray(thumb.convert("L"), dtype=np.float32)
    return {
        "brightness": float(hsv[:, 2].mean()),
        "saturation": float(hsv[:, 1].mean()),
        "colorfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
        "contrast": float(gray.std()),
        "background_share": float((np.abs(gray - np.median(gray)) <= 12).mean()),
    }


def image_features(raw):
    """All features of one encoded image."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return {**color_features(img), "phash": phash(img.convert("L"))}


def shard_features(path):
    """Features for every image in one blob shard.

    Returns ``(features, errors)``: one row per decodable image (``phash`` as ``uint64``; a float
    column would silently corrupt 64-bit hashes), and one row per blob that failed to decode.
    """
    table = pq.read_table(path, columns=["sha256", "bytes"])
    rows, errors = [], []
    for sha, raw in zip(table["sha256"].to_pylist(), table["bytes"].to_pylist()):
        try:
            rows.append({"sha256": sha, **image_features(raw)})
        except (OSError, ValueError) as e:  # corrupt blob (Pillow): keep going, but record it
            errors.append({"sha256": sha, "error": repr(e)})
    df = pd.DataFrame(rows)
    df["phash"] = np.array(df["phash"].tolist(), dtype=np.uint64)
    return df, pd.DataFrame(errors, columns=["sha256", "error"])


def hamming_neighbors(hashes, max_dist, block=256):
    """All-pairs Hamming search over 64-bit hashes (brute force, O(n^2), fine up to ~200k).

    Returns ``nn_dist`` (distance from each hash to its nearest *other* hash) and the pairs
    ``(i, j, d)`` with ``i < j`` and ``d <= max_dist`` as a DataFrame.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    nn_dist = np.full(n, 64, dtype=np.uint8)
    pi, pj, pd_ = [], [], []
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = np.bitwise_count(h[start:stop, None] ^ h[None, :]).astype(np.uint8)
        rows = np.arange(stop - start)
        d[rows, rows + start] = 64  # ignore self-distance
        nn_dist[start:stop] = d.min(axis=1)
        a, b = np.nonzero(d <= max_dist)
        a += start
        keep = b > a
        pi.append(a[keep])
        pj.append(b[keep])
        pd_.append(d[a[keep] - start, b[keep]])
    pairs = pd.DataFrame({"i": np.concatenate(pi), "j": np.concatenate(pj), "d": np.concatenate(pd_)})
    return nn_dist, pairs


def _pairs_within(idx, h, max_dist, block):
    """Pairs (a, b, d), a < b in the original numbering, among the hashes ``h[idx]``."""
    sub = h[idx]
    out = []
    for start in range(0, len(idx), block):
        d = np.bitwise_count(sub[start:start + block, None] ^ sub[None, :])
        a, b = np.nonzero(d <= max_dist)
        a += start
        keep = b > a
        a, b = a[keep], b[keep]
        out.append((idx[a], idx[b], d[a - start, b].astype(np.uint8)))
    return out


def multi_index_pairs(hashes, max_dist=4, block=2048):
    """All pairs of 64-bit hashes within Hamming distance ``max_dist``, exactly, at scale.

    Multi-index hashing: split the 64 bits into ``max_dist + 1`` chunks. Two hashes within
    ``max_dist`` bits must agree on at least one whole chunk (pigeonhole), so only hashes sharing a
    chunk value are compared. Pass *distinct* hash values (``np.unique``) so identical images do not
    inflate the buckets. Returns a DataFrame ``(i, j, d)`` with ``i < j``, deduplicated.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    n_chunks = max_dist + 1
    widths = [64 // n_chunks + (k < 64 % n_chunks) for k in range(n_chunks)]
    parts, shift = [], 64
    for w in widths:
        shift -= w
        key = (h >> np.uint64(shift)) & np.uint64((1 << w) - 1)
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        cuts = np.flatnonzero(np.diff(sorted_key)) + 1
        starts, ends = np.r_[0, cuts], np.r_[cuts, n]
        for s, e in zip(starts, ends):
            if e - s > 1:
                parts += _pairs_within(order[s:e], h, max_dist, block)
    if not parts:
        return pd.DataFrame({"i": [], "j": [], "d": []})
    i = np.concatenate([p[0] for p in parts]).astype(np.int64)
    j = np.concatenate([p[1] for p in parts]).astype(np.int64)
    d = np.concatenate([p[2] for p in parts])
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    _, first = np.unique(lo * n + hi, return_index=True)  # the same pair can match on several chunks
    return pd.DataFrame({"i": lo[first], "j": hi[first], "d": d[first]})


def star_clusters(n, i, j):
    """Greedy star clustering of an undirected graph given as edge arrays.

    Nodes are visited by decreasing degree; each unassigned node becomes a centre and takes all of
    its still-unassigned neighbours. Every member is therefore directly linked to its centre, so,
    unlike connected components, clusters cannot chain (A~B, B~C, ... with A far from the end).
    Returns the centre's index for every node (a node alone is its own centre).
    """
    neighbors = [[] for _ in range(n)]
    for a, b in zip(np.asarray(i), np.asarray(j)):
        neighbors[a].append(b)
        neighbors[b].append(a)
    order = np.argsort([-len(nb) for nb in neighbors], kind="stable")
    centre = np.full(n, -1)
    for c in order:
        if centre[c] != -1:
            continue
        centre[c] = c
        for m in neighbors[c]:
            if centre[m] == -1:
                centre[m] = c
    return centre
