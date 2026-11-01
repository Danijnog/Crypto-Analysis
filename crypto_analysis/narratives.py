"""Narratives of pump.fun tokens, adapting narrative modelling (Gerard et al., ICWSM 2025) to our topics.

The story clusters are the BERTopic text topics (``topic_search``), each represented by the centroid
of its texts' embeddings with the multilingual MPNet model used by Gerard et al. (``text_mpnet.npy``,
from ``topic_search.embed_text_docs``). Narratives are the clusters of an average-linkage
agglomerative clustering of those centroids under cosine distance, cut at a similarity threshold
rather than at a number of clusters. Gerard et al. join story clusters into a narrative when they lie
within 0.7-0.8 cosine similarity of it, so ``sweep`` reports the partition across thresholds before
one is chosen.

The geometry alone shows no stable cut, and our topics are already as loose as Gerard et al.'s
narratives (median document-to-centroid similarity 0.68, against 0.935 for their stories), so the
threshold is calibrated as they calibrated theirs, against same/different judgments: ``pairs`` draws
topic pairs stratified by their cophenetic similarity and asks an LLM (Qwen3.5-9B, greedy, both
orders) whether the two topics belong to the same narrative (``NARRATIVE`` gives the definition);
``calibrate`` picks the threshold whose partition best agrees with those judgments (F1 of "same",
pairs weighted back to the population of topic pairs). A human subset validates the LLM judge.

No threshold agrees with the judgments (best F1 0.15): the closest topics in the tree are the ones
without a theme, as short meaningless names embed close together. So ``graph`` lets the geometry only
propose candidates (each topic's nearest topics) and the LLM decide: narratives are the communities
of the graph of topics judged to share a narrative.

Writes to ``<out>/narratives/``: ``centroids.npy``, ``sizes.npy``, ``linkage.npy``,
``sweep.parquet``, ``pairs.parquet``, ``calibration.parquet``, ``candidates.parquet`` and
``topics.parquet``. Usage::

    python -m crypto_analysis.narratives sweep     --out-dir data/topics
    python -m crypto_analysis.narratives pairs     --out-dir data/topics
    python -m crypto_analysis.narratives calibrate --out-dir data/topics
    python -m crypto_analysis.narratives graph     --out-dir data/topics
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

MPNET_FILE = "text_mpnet.npy"
THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.05), 2)
CALIBRATION_THRESHOLDS = np.round(np.arange(0.40, 0.951, 0.025), 3)
PAIR_BINS = np.round(np.arange(0.35, 1.001, 0.05), 2)  # cophenetic similarity strata
PAIRS_PER_BIN = 35
PAIR_SAMPLES = 8
NEIGHBOURS = 5
SEED = 42
NARRATIVE = (
    "A narrative is a recurring story, cultural reference or point of view that coins exploit to attract "
    'attention, for example "Trump and US politics", "dog coins", "AI agents", "sexual humor" or '
    '"antisemitism". Two topics belong to the same narrative when a pump.fun trader would recognise them '
    'as the same market "narrative".'
)
PAIR_PROMPT = (
    "You are helping researchers study the narratives of the memecoins launched on pump.fun, a Solana "
    "launchpad where anyone can create a token with a name, a description and an image. {definition}\n\n"
    "Below are two topics of tokens, each with its keywords and token texts (name, then description) "
    "sampled across the whole topic.\n\n{a}\n\n{b}\n\n"
    "Do topics A and B belong to the same narrative? Answer with a JSON object only, with these keys:\n"
    '- "same": true or false\n'
    '- "narrative": if same, the shared narrative in at most 5 words, otherwise ""\n'
    '- "reason": at most 15 words'
)


def topic_centroids(out_dir, chunk=1_000_000):
    """L2-normalised MPNet centroid and size (distinct texts) of every text topic."""
    import torch

    out = Path(out_dir)
    keys = pd.read_parquet(out / "docs_text.parquet", columns=["text_key"]).text_key
    X = np.load(out / MPNET_FILE, mmap_mode="r")
    labels = pd.read_parquet(out / "text" / "full_topics.parquet").set_index("key").topic
    labels = labels.reindex(keys).fillna(-1).astype(np.int64).to_numpy()
    rows = np.flatnonzero(labels != -1)
    C = torch.zeros(int(labels.max()) + 1, X.shape[1], device="cuda")
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        C.index_add_(0, torch.from_numpy(labels[part]).cuda(), torch.from_numpy(np.asarray(X[part])).cuda().float())
    return torch.nn.functional.normalize(C, dim=1).cpu().numpy(), np.bincount(labels[rows], minlength=len(C))


def sweep(out_dir, thresholds=THRESHOLDS):
    """Narratives at every similarity threshold: count, size distribution and agreement with the
    next threshold (adjusted Rand index over topics)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import adjusted_rand_score

    out = Path(out_dir) / "narratives"
    out.mkdir(parents=True, exist_ok=True)
    C, sizes = topic_centroids(out_dir)
    Z = linkage(C, method="average", metric="cosine")
    np.save(out / "centroids.npy", C)
    np.save(out / "sizes.npy", sizes)
    np.save(out / "linkage.npy", Z)
    total, rows, previous = sizes.sum(), [], None
    for thr in thresholds:
        labels = fcluster(Z, t=1 - thr, criterion="distance")
        docs = pd.Series(sizes).groupby(labels).sum()
        topics = pd.Series(labels).value_counts()
        rows.append({"threshold": thr, "narratives": len(docs), "singletons": int((topics == 1).sum()),
                     "largest_share": docs.max() / total,
                     "share_in_multi_topic": docs[topics[topics > 1].index].sum() / total,
                     "ari_with_previous": adjusted_rand_score(previous, labels) if previous is not None else None})
        previous = labels
    df = pd.DataFrame(rows)
    df.to_parquet(out / "sweep.parquet", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    return df


def cophenetic_similarity(Z):
    """Square matrix of 1 - cophenetic distance: the similarity at which two topics join."""
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform

    return 1 - squareform(cophenet(Z))


def sample_pairs(S, bins=PAIR_BINS, per_bin=PAIRS_PER_BIN, seed=0):
    """Topic pairs stratified by cophenetic similarity, each with the weight (pairs in its stratum /
    pairs drawn from it) that maps the sample back to all topic pairs."""
    rng = np.random.default_rng(seed)
    a, b = np.triu_indices(len(S), k=1)
    sim = S[a, b]
    stratum = np.digitize(sim, bins)
    rows = []
    for s in np.unique(stratum):
        members = np.flatnonzero(stratum == s)
        drawn = rng.choice(members, min(per_bin, len(members)), replace=False)
        rows += [{"a": int(a[i]), "b": int(b[i]), "cophenetic": float(sim[i]), "stratum": int(s),
                  "weight": len(members) / len(drawn)} for i in drawn]
    return pd.DataFrame(rows)


def _topic_text(label, cards, topic, k=PAIR_SAMPLES):
    grp = cards[cards.topic == topic].sort_values("rank").head(k)
    samples = "\n".join(f"  {i + 1}. {d[:120]}" for i, d in enumerate(grp.doc))
    return f"Topic {label}\nKeywords: {grp.words.iloc[0]}\nSamples:\n{samples}"


def ask_pairs(out_dir, pairs):
    """LLM same-narrative judgments of the topic pairs (columns ``a``, ``b``), asked in both orders;
    ``same`` is the answer when both orders agree, missing otherwise."""
    from crypto_analysis.topic_taxonomy import Asker

    cards = pd.read_parquet(Path(out_dir) / "annotation" / "text" / "cards.parquet")
    ask, start = Asker(), time.time()
    pairs = pairs.reset_index(drop=True)
    for order, (first, second) in {"ab": ("a", "b"), "ba": ("b", "a")}.items():
        answers = []
        for row in pairs.itertuples():
            prompt = PAIR_PROMPT.format(definition=NARRATIVE, a=_topic_text("A", cards, getattr(row, first)),
                                        b=_topic_text("B", cards, getattr(row, second)))
            raw, answer = ask(prompt, max_new_tokens=120)
            same = answer.get("same")
            answers.append({f"same_{order}": same if isinstance(same, bool) else None,
                            f"narrative_{order}": str(answer.get("narrative", "")),
                            f"reason_{order}": str(answer.get("reason", "")), f"raw_{order}": raw})
            if len(answers) % 250 == 0:
                print(f"  order {order}: {len(answers)}/{len(pairs)} ({(time.time() - start) / 60:.1f} min)", flush=True)
        pairs = pd.concat([pairs, pd.DataFrame(answers)], axis=1)
    consistent = pairs.same_ab.notna() & (pairs.same_ab == pairs.same_ba)
    pairs["same"] = pairs.same_ab.where(consistent)
    print(f"{len(pairs)} pairs; order-consistent {consistent.mean():.1%}; judged same "
          f"{pairs.same.mean():.1%} of the consistent ones ({(time.time() - start) / 60:.1f} min)", flush=True)
    return pairs


def judge_pairs(out_dir):
    """Judgments of topic pairs stratified by cophenetic similarity (``pairs.parquet``)."""
    out = Path(out_dir) / "narratives"
    pairs = ask_pairs(out_dir, sample_pairs(cophenetic_similarity(np.load(out / "linkage.npy"))))
    pairs.to_parquet(out / "pairs.parquet", index=False)
    return pairs


def neighbour_pairs(C, k=NEIGHBOURS):
    """Unordered pairs of each topic with its ``k`` nearest topics (cosine of the MPNet centroids)."""
    S = C @ C.T
    np.fill_diagonal(S, -np.inf)
    near = np.argsort(-S, axis=1)[:, :k]
    pairs = {(min(i, j), max(i, j)) for i in range(len(C)) for j in near[i]}
    return pd.DataFrame([{"a": a, "b": b, "similarity": float(S[a, b])} for a, b in sorted(pairs)])


def narrative_graph(out_dir, k=NEIGHBOURS, seed=SEED):
    """Narratives as communities of the graph of topics the LLM judged to share a narrative.

    The geometry only proposes the candidates (each topic with its ``k`` nearest topics); an edge is
    kept when the LLM says "same narrative" in both orders (as ClusterLLM, Zhang et al. 2023, lets an
    LLM's pair judgments guide the clustering). Louvain communities of that graph are the narratives,
    which keeps chains A~B~C from joining A and C when the graph does not support it; topics without
    an edge are narratives of their own. Writes ``candidates.parquet`` and ``topics.parquet``
    (topic, narrative, component)."""
    import networkx as nx

    out = Path(out_dir) / "narratives"
    cached = out / "candidates.parquet"
    if cached.exists():
        pairs = pd.read_parquet(cached)
    else:
        pairs = ask_pairs(out_dir, neighbour_pairs(np.load(out / "centroids.npy"), k))
        pairs.to_parquet(cached, index=False)
    sizes = np.load(out / "sizes.npy")
    G = nx.Graph()
    G.add_nodes_from(range(len(sizes)))
    G.add_edges_from(pairs.loc[pairs.same.fillna(False).astype(bool), ["a", "b"]].itertuples(index=False))
    component = {t: i for i, comp in enumerate(nx.connected_components(G)) for t in comp}
    communities = sorted(nx.community.louvain_communities(G, seed=seed), key=lambda c: -sizes[list(c)].sum())
    narrative = {t: i for i, comm in enumerate(communities) for t in comm}
    topics = pd.DataFrame({"topic": range(len(sizes)), "docs": sizes,
                           "narrative": [narrative[t] for t in range(len(sizes))],
                           "component": [component[t] for t in range(len(sizes))]})
    topics.to_parquet(out / "topics.parquet", index=False)
    multi = topics.groupby("narrative").filter(lambda g: len(g) > 1)
    print(f"{G.number_of_edges()} same-narrative edges among {len(pairs)} candidate pairs; "
          f"{nx.number_connected_components(G)} components, {len(communities)} narratives; "
          f"{multi.narrative.nunique()} with 2+ topics cover {multi.docs.sum() / sizes.sum():.1%} of the documents",
          flush=True)
    return topics


def calibrate(out_dir, thresholds=CALIBRATION_THRESHOLDS):
    """Agreement of each threshold's partition with the order-consistent LLM judgments: weighted
    precision, recall and F1 of "same narrative", and weighted accuracy (``calibration.parquet``)."""
    from scipy.cluster.hierarchy import fcluster

    out = Path(out_dir) / "narratives"
    pairs = pd.read_parquet(out / "pairs.parquet").dropna(subset=["same"])
    y, w = pairs.same.astype(bool).to_numpy(), pairs.weight.to_numpy()
    Z = np.load(out / "linkage.npy")
    rows = []
    for thr in thresholds:
        pred = pairs.cophenetic.to_numpy() >= thr - 1e-9
        tp, fp, fn = w[pred & y].sum(), w[pred & ~y].sum(), w[~pred & y].sum()
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        rows.append({"threshold": thr, "narratives": len(np.unique(fcluster(Z, t=1 - thr, criterion="distance"))),
                     "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
                     "accuracy": w[pred == y].sum() / w.sum()})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "calibration.parquet", index=False)
    print(df.round(3).to_string(index=False))
    best = df.loc[df.f1.idxmax()]
    print(f"best F1 {best.f1:.3f} at similarity {best.threshold} ({int(best.narratives)} narratives)", flush=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sweep", "pairs", "calibrate", "graph"])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    {"sweep": sweep, "pairs": judge_pairs, "calibrate": calibrate, "graph": narrative_graph}[args.step](args.out_dir)


if __name__ == "__main__":
    main()
