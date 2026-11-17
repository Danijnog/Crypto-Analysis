"""Sampling experiments for the per-modality BERTopic models: the topics are learned on a sample of
documents and every document of the modality is then assigned to them.

Two schemes, of the same size (``SAMPLE_SIZE`` documents) in every modality:

- ``random``: uniform over the modality's distinct documents;
- ``temporal``: month quotas proportional to the month's number of coins, each month split 50/50
  between documents of graduated and of non-graduated coins. Graduated coins are 1% of the coins, so
  the size is bounded by the month with the fewest graduated documents per coin: about 61k for text
  and 77k for image and multimodal; ``SAMPLE_SIZE`` is the same in all three so they compare.

A document's month is the launch month of the first coin using it; it is graduated when any coin
using it graduated. For each scheme and modality the ``topic_search`` grid selects the configuration
on the sample; it is refit on the sample (cuML UMAP + HDBSCAN with prediction data), every other
document is placed by ``UMAP.transform`` and HDBSCAN ``approximate_predict``, and the outliers are
then reassigned with BERTopic's ``reduce_outliers`` strategies at their default threshold (an outlier
with no positive similarity to any topic stays an outlier), per modality (``REDUCTION``): c-TF-IDF for
text, where the clusters come from the words' embedding; embeddings for image and multimodal, whose
words (token names) are not what was clustered, so the outlier goes to the topic with the most similar
mean embedding of its sample documents. Both assignments are kept.

Outputs in ``<out>/sampling/``: ``coins.parquet`` (mint, sha256, text_key, month, graduated) and, per
``<scheme>/<modality>/``: ``sample.parquet``, ``grid/``, ``topics.parquet`` (key, topic,
topic_reduced, in_sample), ``words.json``, ``info.parquet``, ``representatives.parquet``; one row per
fit in ``results.parquet``. ``compare`` writes per-coin topics of every scheme and modality
(``<scheme>/token_topics.parquet``) and the tables of ``comparison/``. Usage::

    python -m crypto_analysis.topic_sampling coins   --out-dir data/topics
    python -m crypto_analysis.topic_sampling fit     --out-dir data/topics --scheme temporal --modality text
    python -m crypto_analysis.topic_sampling reduce  --out-dir data/topics --modality image
    python -m crypto_analysis.topic_sampling fit     --out-dir data/topics --modality multimodal_concat
    python -m crypto_analysis.topic_sampling reduce-full --out-dir data/topics --modality multimodal
    python -m crypto_analysis.topic_sampling compare --out-dir data/topics
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crypto_analysis.datalake import DATA_DIR, connect
from crypto_analysis.topic_alignment import symmetry_ratio
from crypto_analysis.topic_search import (
    FUSIONS,
    MODALITIES,
    SEED,
    _append,
    _host,
    fit_labels,
    grid_search,
    load_modality,
    reference,
    representatives,
    score_topics,
)

SCHEMES = ["random", "temporal"]
SAMPLE_SIZE = 60_000
CHUNK = 500_000  # documents transformed / predicted / reassigned at a time
MODELS = ["text", "image", *FUSIONS]  # the three modalities and the alternative multimodal fusions
REDUCTION = {"text": "c-tf-idf", "image": "embeddings", **{m: "embeddings" for m in FUSIONS}}


def sampling_dir(out_dir):
    return Path(out_dir) / "sampling"


def coins_table(out_dir, data_dir=DATA_DIR):
    """``sampling/coins.parquet``: one row per token with a document, its launch month and whether it
    graduated."""
    path = sampling_dir(out_dir) / "coins.parquet"
    if path.exists():
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(data_dir, threads=16)
    flags = con.execute("""
        SELECT DISTINCT t.mint, strftime(date_trunc('month', to_timestamp(t.created_at)), '%Y-%m') AS month,
               g.mint IS NOT NULL AS graduated
        FROM tokens t LEFT JOIN (SELECT DISTINCT mint FROM graduations) g USING (mint)
    """).df().drop_duplicates("mint")
    tok = pd.read_parquet(Path(out_dir) / "docs_token.parquet", columns=["mint", "sha256", "text_key"])
    coins = tok.merge(flags, on="mint", how="inner")
    coins.to_parquet(path, index=False)
    print(f"coins.parquet: {len(coins):,} coins, {coins.graduated.sum():,} graduated", flush=True)
    return coins


def document_flags(coins, modality):
    """Month (of the first coin) and graduated (any coin) of every document: DataFrame indexed by key."""
    key = {"text": coins.text_key, "image": coins.sha256}.get(modality, coins.sha256 + ":" + coins.text_key)
    return coins.assign(key=key).dropna(subset=["key"]).groupby("key").agg(month=("month", "min"),
                                                                           graduated=("graduated", "any"))


def month_quotas(coins, n):
    """Documents per month proportional to its coins, rounded by largest remainder to sum to ``n``."""
    share = coins.month.value_counts(normalize=True).sort_index() * n
    quota = np.floor(share).astype(int)
    quota[(share - quota).sort_values(ascending=False).index[:n - quota.sum()]] += 1
    return quota


def draw_sample(keys, flags, coins, scheme, n=SAMPLE_SIZE, seed=SEED):
    """Rows of ``keys`` in the sample of ``scheme``, sorted."""
    rng = np.random.default_rng(seed)
    if scheme == "random":
        return np.sort(rng.choice(len(keys), size=n, replace=False))
    doc = flags.reindex(keys).reset_index(drop=True)
    rows = []
    for month, quota in month_quotas(coins, n).items():
        for graduated, k in [(True, quota // 2), (False, quota - quota // 2)]:
            pool = np.flatnonzero((doc.month == month).to_numpy() & (doc.graduated == graduated).to_numpy())
            if len(pool) < k:
                raise ValueError(f"{month}: {len(pool)} documents with graduated={graduated}, {k} needed")
            rows.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(rows))


def fit_and_assign(X, rows, config):
    """Fit UMAP + HDBSCAN with ``config`` on ``X[rows]`` and assign every row of ``X``: the sample keeps
    its HDBSCAN labels, the others get ``approximate_predict`` on their ``UMAP.transform``."""
    from cuml.cluster import HDBSCAN
    from cuml.cluster.hdbscan import approximate_predict
    from cuml.manifold import UMAP

    umap = UMAP(n_neighbors=config["n_neighbors"], n_components=config["n_components"], min_dist=0.0,
                metric="cosine", random_state=SEED).fit(X[rows].astype(np.float32))
    reduced = _host(umap.embedding_)
    clusterer = HDBSCAN(min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"],
                        cluster_selection_method="eom", prediction_data=True).fit(reduced)
    labels = np.empty(len(X), dtype=np.int32)
    for s in range(0, len(X), CHUNK):
        part = _host(umap.transform(X[s:s + CHUNK].astype(np.float32)))
        labels[s:s + CHUNK] = _host(approximate_predict(clusterer, part)[0])
    labels[rows] = _host(clusterer.labels_)
    return labels


def reduce_by_ctfidf(model, renumbered, docs, labels):
    """BERTopic's c-TF-IDF ``reduce_outliers`` (default threshold) on the outliers of ``labels``, in
    chunks; ``model``/``renumbered`` from ``fit_labels`` on the sample."""
    back = {int(new): int(t) for t, new in renumbered.items() if t != -1}
    out = labels.copy()
    outliers = np.flatnonzero(labels == -1)
    for s in range(0, len(outliers), CHUNK):
        rows = outliers[s:s + CHUNK]
        new = model.reduce_outliers([docs[i] for i in rows], [-1] * len(rows), strategy="c-tf-idf")
        out[rows] = [back.get(int(t), -1) for t in new]
    return out


def reduce_by_embeddings(X, labels, rows):
    """BERTopic's ``embeddings`` strategy of ``reduce_outliers`` (default threshold 0): every outlier
    goes to the topic whose mean embedding over the sample ``rows`` is the most cosine-similar, unless
    no similarity is positive."""
    import torch

    lab = labels[rows]
    keep = lab != -1
    C = torch.zeros(int(lab.max()) + 1, X.shape[1], device="cuda")
    C.index_add_(0, torch.from_numpy(lab[keep].astype(np.int64)).cuda(), torch.from_numpy(X[rows[keep]]).cuda().float())
    C = torch.nn.functional.normalize(C, dim=1)
    out = labels.copy()
    outliers = np.flatnonzero(labels == -1)
    for s in range(0, len(outliers), CHUNK):
        part = torch.nn.functional.normalize(torch.from_numpy(X[outliers[s:s + CHUNK]]).cuda().float(), dim=1)
        best = (part @ C.T).max(dim=1)
        out[outliers[s:s + CHUNK]] = np.where(best.values.cpu().numpy() > 0, best.indices.cpu().numpy(), -1)
    return out


def reduce_outliers(modality, docs, X, labels, rows):
    """Outliers of ``labels`` reassigned with the modality's ``REDUCTION`` strategy, the topics being
    those of the sample ``rows``."""
    if REDUCTION[modality] == "embeddings":
        return reduce_by_embeddings(X, labels, rows)
    return reduce_by_ctfidf(*fit_labels([docs[i] for i in rows], labels[rows]), docs, labels)


def fit_scheme(out_dir, scheme, modality, clip_dir, n=SAMPLE_SIZE):
    """Sample, grid, refit, assignment and outlier reduction of one scheme and modality."""
    out = sampling_dir(out_dir) / scheme / modality
    out.mkdir(parents=True, exist_ok=True)
    coins = coins_table(out_dir)
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    flags = document_flags(coins, modality)
    rows = draw_sample(keys, flags, coins, scheme, n)
    sample = flags.reindex(keys[rows]).rename_axis("key").reset_index()
    sample.to_parquet(out / "sample.parquet", index=False)
    print(f"{scheme}/{modality}: {len(rows):,} of {len(keys):,} documents, {sample.graduated.mean():.1%} graduated",
          flush=True)
    ref = reference(docs)
    config = grid_search(keys[rows], [docs[i] for i in rows], X[rows], ref, out / "grid", f"{scheme}/{modality}")
    start = time.time()
    labels = fit_and_assign(X, rows, config)
    sample_docs, sample_labels = [docs[i] for i in rows], labels[rows]
    model, renumbered = fit_labels(sample_docs, sample_labels)
    words = {int(t): [w for w, _ in model.get_topic(int(renumbered[t])) if w] for t in renumbered.index if t != -1}
    reduced = reduce_outliers(modality, docs, X, labels, rows)
    in_sample = np.zeros(len(keys), dtype=bool)
    in_sample[rows] = True
    pd.DataFrame({"key": keys, "topic": labels, "topic_reduced": reduced, "in_sample": in_sample}).to_parquet(
        out / "topics.parquet", index=False)
    (out / "words.json").write_text(json.dumps(words))
    info = pd.DataFrame({"topic": sorted(words)})
    for col, lab in [("sample_count", sample_labels), ("count", labels), ("count_reduced", reduced)]:
        info[col] = info.topic.map(pd.Series(lab).value_counts()).fillna(0).astype(int)
    info["words"] = [" ".join(words[t][:10]) for t in info.topic]
    info.to_parquet(out / "info.parquet", index=False)
    reps = representatives(X, labels)
    reps["key"] = keys[reps.row]
    reps["doc"] = [docs[i][:300] for i in reps.row]
    reps.drop(columns="row").to_parquet(out / "representatives.parquet", index=False)
    row = {"scheme": scheme, "modality": modality, "run": config["run"], "sample_docs": len(rows), "docs": len(keys),
           **score_topics(list(words.values()), *ref, len(rows), int((sample_labels == -1).sum())),
           "outlier_rate_all": float((labels == -1).mean()), "reduction": REDUCTION[modality],
           "outlier_rate_reduced": float((reduced == -1).mean()), "minutes": (time.time() - start) / 60}
    row["outlier_rate_sample"] = row.pop("outlier_rate")
    path = sampling_dir(out_dir) / "results.parquet"
    if path.exists():  # a refit replaces the earlier row of the same scheme and modality
        old = pd.read_parquet(path)
        old[(old.scheme != scheme) | (old.modality != modality)].to_parquet(path, index=False)
    _append(path, row)
    print(row, flush=True)
    return row


def reduce_full(out_dir, modality, clip_dir):
    """Outlier reduction of the full-corpus model of ``modality`` (``topic_search.run_full``) with the
    modality's ``REDUCTION`` strategy, the topics being represented by all their members: adds
    ``topic_reduced`` to ``<out>/<modality>/full_topics.parquet``. Topics and their members are
    unchanged, so the narratives built on them stay valid."""
    path = Path(out_dir) / modality / "full_topics.parquet"
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    topics = pd.read_parquet(path)
    assert (topics.key.to_numpy() == keys).all()
    labels = topics.topic.to_numpy()
    topics["topic_reduced"] = reduce_outliers(modality, docs, X, labels, np.arange(len(labels)))
    topics.to_parquet(path, index=False)
    print(f"full/{modality}: {REDUCTION[modality]} reduction, outliers {(labels == -1).mean():.1%} -> "
          f"{(topics.topic_reduced == -1).mean():.1%}", flush=True)


def refit_reduction(out_dir, scheme, modality, clip_dir):
    """Redo the outlier reduction of a finished fit with the modality's ``REDUCTION`` strategy,
    updating ``topics.parquet``, ``info.parquet`` and ``results.parquet``."""
    out = sampling_dir(out_dir) / scheme / modality
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    topics = pd.read_parquet(out / "topics.parquet")
    assert (topics.key.to_numpy() == keys).all()
    labels, rows = topics.topic.to_numpy(), np.flatnonzero(topics.in_sample.to_numpy())
    topics["topic_reduced"] = reduce_outliers(modality, docs, X, labels, rows)
    topics.to_parquet(out / "topics.parquet", index=False)
    info = pd.read_parquet(out / "info.parquet")
    info["count_reduced"] = info.topic.map(topics.topic_reduced.value_counts()).fillna(0).astype(int)
    info.to_parquet(out / "info.parquet", index=False)
    path = sampling_dir(out_dir) / "results.parquet"
    results = pd.read_parquet(path)
    at = (results.scheme == scheme) & (results.modality == modality)
    results.loc[at, "reduction"] = REDUCTION[modality]
    results.loc[at, "outlier_rate_reduced"] = float((topics.topic_reduced == -1).mean())
    results.to_parquet(path, index=False)
    print(f"{scheme}/{modality}: {REDUCTION[modality]} reduction, "
          f"{(topics.topic_reduced == -1).mean():.1%} outliers left", flush=True)


# --------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------

def scheme_token_topics(out_dir, scheme):
    """``<scheme>/token_topics.parquet``: every coin with its topic in every modality, before
    (``<modality>``) and after (``<modality>_reduced``) outlier reduction; -1 when the coin's document
    is an outlier or has no embedding."""
    base = sampling_dir(out_dir)
    tok = pd.read_parquet(base / "coins.parquet")
    for modality in [m for m in MODELS if (base / scheme / m / "topics.parquet").exists()]:
        key = {"text": tok.text_key, "image": tok.sha256}.get(modality, tok.sha256 + ":" + tok.text_key)
        topics = pd.read_parquet(base / scheme / modality / "topics.parquet").set_index("key")
        for col, name in [("topic", modality), ("topic_reduced", f"{modality}_reduced")]:
            tok[name] = topics[col].reindex(key).fillna(-1).astype(np.int32).to_numpy()
    tok.to_parquet(base / scheme / "token_topics.parquet", index=False)
    return tok


def coverage(tok, suffix=""):
    """Share of coins in a topic of each combination of modalities (all coins and graduated ones)."""
    code = sum((tok[f"{m}{suffix}"].to_numpy() != -1) << k for k, m in enumerate(MODALITIES))
    names = ["+".join(m for k, m in enumerate(MODALITIES) if c >> k & 1) or "none" for c in range(2 ** len(MODALITIES))]
    pattern = pd.Series(np.asarray(names)[code])
    out = pd.DataFrame({"all": pattern.value_counts(normalize=True),
                        "graduated": pattern[tok.graduated.to_numpy()].value_counts(normalize=True)}).fillna(0)
    return out.rename_axis("modalities").reset_index()


def theils_u(x, y):
    """Theil's U(y|x), the uncertainty coefficient I(x; y) / H(y): how well ``x`` predicts ``y``, from 0
    (not at all) to 1 (``y`` is a function of ``x``, e.g. ``x`` is a refinement of ``y``)."""
    from scipy.stats import entropy
    from sklearn.metrics import mutual_info_score

    h_y = entropy(np.unique(y, return_counts=True)[1])
    return mutual_info_score(x, y) / h_y if h_y > 0 else 1.0


def pair_stats(a, b, a_name, b_name, shuffles=3, seed=SEED):
    """Agreement of two coin-level partitions on the coins in a topic in both (-1: outlier): symmetry
    ratio (each direction), AMI and Theil's U (each direction), the latter also with the source's labels
    shuffled (same topic sizes), the baseline a finer source reaches by granularity alone."""
    from sklearn.metrics import adjusted_mutual_info_score

    both = (a != -1) & (b != -1)
    x, y = a[both].to_numpy(), b[both].to_numpy()
    rng = np.random.default_rng(seed)
    shared = {"ami": adjusted_mutual_info_score(x, y), "coins_in_both": both.mean()}
    rows = []
    for (src, s, tgt, t, name_s, name_t) in [(a, x, b, y, a_name, b_name), (b, y, a, x, b_name, a_name)]:
        rows.append({"source": name_s, "target": name_t, "source_topics": len(np.unique(s)),
                     "target_topics": len(np.unique(t)), "symmetry_ratio": symmetry_ratio(src, tgt)[0],
                     "theils_u": theils_u(s, t),
                     "theils_u_shuffled": np.mean([theils_u(rng.permutation(s), t) for _ in range(shuffles)]),
                     **shared})
    return rows


def previous_models(out_dir):
    """Coin-level topics of the full-corpus models (``token_topics.parquet``) and narratives of the
    published-definition runs (text and multimodal), -1 for outliers."""
    from crypto_analysis.narratives import coin_narratives

    tok = pd.read_parquet(Path(out_dir) / "token_topics.parquet")
    out = pd.DataFrame({m: tok[f"{m}_topic"].fillna(-1).astype(np.int32) for m in MODALITIES})
    runs = {"text_published": ("text", "published"), "multimodal_published": ("multimodal", "published")}
    nar = coin_narratives(out_dir, runs)
    for run, (modality, _) in runs.items():
        out[f"{modality}_narrative"] = nar[run].fillna(-1).astype(np.int32).to_numpy()
    return out


def compare(out_dir):
    """Per scheme: coverage of coins by every combination of the three modalities and by every fitted
    model (``MODELS``); symmetry ratio, AMI and Theil's U of every pair of models; per model the same
    between the schemes; before and after outlier reduction. Also the pairs of the previous full-corpus
    models, at topic and (text and multimodal) narrative level."""
    base = sampling_dir(out_dir)
    (base / "comparison").mkdir(parents=True, exist_ok=True)
    tokens = {s: scheme_token_topics(out_dir, s) for s in SCHEMES}
    fitted = [m for m in MODELS if all(m in tok for tok in tokens.values())]
    cover, single, pairs = [], [], []
    for suffix, assignment in [("", "hdbscan"), ("_reduced", "reduced")]:
        for scheme, tok in tokens.items():
            cover.append(coverage(tok, suffix).assign(scheme=scheme, assignment=assignment))
            single += [{"scheme": scheme, "assignment": assignment, "model": m,
                        "all": (tok[f"{m}{suffix}"] != -1).mean(),
                        "graduated": (tok.loc[tok.graduated, f"{m}{suffix}"] != -1).mean()} for m in fitted]
            for a, b in combinations(fitted, 2):
                rows = pair_stats(tok[f"{a}{suffix}"], tok[f"{b}{suffix}"], a, b)
                pairs += [dict(r, scheme=scheme, assignment=assignment, level="topic") for r in rows]
        for m in fitted:
            rows = pair_stats(*(tokens[s][f"{m}{suffix}"] for s in SCHEMES), f"{m} random", f"{m} temporal")
            pairs += [dict(r, scheme="random vs temporal", assignment=assignment, level="topic") for r in rows]
    prev = previous_models(out_dir)
    for level, models in [("topic", MODALITIES), ("narrative", ["text_narrative", "multimodal_narrative"])]:
        for a, b in combinations(models, 2):
            rows = pair_stats(prev[a], prev[b], a.split("_")[0], b.split("_")[0])
            pairs += [dict(r, scheme="full (previous)", assignment="hdbscan", level=level) for r in rows]
    cover, single, pairs = pd.concat(cover, ignore_index=True), pd.DataFrame(single), pd.DataFrame(pairs)
    cover.to_parquet(base / "comparison" / "coverage.parquet", index=False)
    single.to_parquet(base / "comparison" / "coverage_models.parquet", index=False)
    pairs.to_parquet(base / "comparison" / "pairs.parquet", index=False)
    results = pd.read_parquet(base / "results.parquet")
    cols = ["scheme", "modality", "run", "n_topics", "npmi", "diversity", "outlier_rate_sample", "outlier_rate_all",
            "reduction", "outlier_rate_reduced"]
    print(results[cols].sort_values(["modality", "scheme"]).round(3).to_string(index=False))
    print(cover.pivot_table(index="modalities", columns=["assignment", "scheme"], values=["all", "graduated"])
          .round(3).to_string())
    print(single.pivot_table(index="model", columns=["assignment", "scheme"], values=["all", "graduated"])
          .round(3).to_string())
    cols = ["scheme", "assignment", "level", "source", "target", "source_topics", "target_topics", "coins_in_both",
            "symmetry_ratio", "ami", "theils_u", "theils_u_shuffled"]
    print(pairs[cols].round(3).to_string(index=False))
    return cover, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["coins", "fit", "reduce", "reduce-full", "compare"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=Path("data/clip"))
    parser.add_argument("--scheme", choices=SCHEMES + ["all"], default="all")
    parser.add_argument("--modality", choices=MODELS + ["all"], default="all")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()
    if args.step == "coins":
        coins_table(args.out_dir)
    elif args.step == "fit":
        for modality in MODELS if args.modality == "all" else [args.modality]:
            for scheme in SCHEMES if args.scheme == "all" else [args.scheme]:
                fit_scheme(args.out_dir, scheme, modality, args.clip_dir, args.sample)
    elif args.step == "reduce":
        for modality in MODELS if args.modality == "all" else [args.modality]:
            for scheme in SCHEMES if args.scheme == "all" else [args.scheme]:
                refit_reduction(args.out_dir, scheme, modality, args.clip_dir)
    elif args.step == "reduce-full":
        for modality in MODALITIES if args.modality == "all" else [args.modality]:
            reduce_full(args.out_dir, modality, args.clip_dir)
    else:
        compare(args.out_dir)


if __name__ == "__main__":
    main()
