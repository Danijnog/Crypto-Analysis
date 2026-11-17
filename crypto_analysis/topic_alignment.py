"""Compare topic models across modalities (image, text, multimodal) by document overlap.

Implements the *symmetry ratio* of Steffen (ICWSM 2025), "More than Memes": for every topic of the
source model, take the target topic sharing the most documents with it; the pair is *symmetric* when
that target topic's best match in the source model is the original topic. The ratio is the share of
source topics with a symmetric match. Outlier topics (-1) are ignored.
"""

import pandas as pd


def overlap_table(a, b):
    """Documents shared by every (topic in ``a``, topic in ``b``) pair. ``a``/``b``: Series doc -> topic."""
    joined = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    joined = joined[(joined.a != -1) & (joined.b != -1)]
    return joined.groupby(["a", "b"]).size().rename("shared").reset_index()


def best_matches(overlap, source="a", target="b"):
    """For each ``source`` topic: the ``target`` topic sharing most documents, and the share it covers."""
    totals = overlap.groupby(source).shared.sum().rename("source_docs")
    best = overlap.sort_values("shared", ascending=False).drop_duplicates(source).set_index(source)
    best = best.join(totals)
    best["coverage"] = best.shared / best.source_docs
    return best[[target, "shared", "source_docs", "coverage"]]


def symmetry_ratio(a, b):
    """Symmetry ratio of model ``a`` with respect to ``b`` and the per-topic match table."""
    overlap = overlap_table(a, b)
    forward = best_matches(overlap, "a", "b")
    backward = best_matches(overlap, "b", "a")
    forward["symmetric"] = [backward.a.get(t) == s for s, t in zip(forward.index, forward.b)]
    return forward.symmetric.mean(), forward
