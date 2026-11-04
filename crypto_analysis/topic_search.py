"""BERTopic per modality, as in Steffen (ICWSM 2025): hyperparameters chosen by a grid search on a
100k-document sample, then the chosen configuration fitted on the full database.

Three modalities, each deduplicated on its own content (the paper's modality-specific deduplication):

- ``text``: one document per distinct token text ("name. description", URLs kept out of the words),
  embedded with the paper's text model ``paraphrase-multilingual-MiniLM-L12-v2``;
- ``image``: one document per distinct image (sha256), clustered on its CLIP image embedding; its
  words are the distinct names of the tokens using it (plus, for the paper's labels, the captions of
  the captioned images in each topic);
- ``multimodal``: one document per distinct (image, text) pair, clustered on the normalised average
  of the CLIP image and CLIP text embeddings; its words are the text.

Every run is scored with NPMI coherence (top-10 words, on a 200k-document reference sample of the
modality's full corpus, so sample and full-database runs are comparable), topic diversity (unique
share of the top-25 words), number of topics and outlier rate (Grootendorst, 2022).

The grid (cuML UMAP x HDBSCAN on the GPU) explores several granularities, but the metrics do not
settle granularity (NPMI favours few broad topics, diversity many narrow ones), so it is fixed by the
paper: ``min_cluster_size`` closest to its min_topic_size share of the corpus (20 / 36,717, i.e. ~54 per
100k documents). Among those runs, the one with the highest topic quality, normalised NPMI
((NPMI + 1) / 2, as NPMI is usually negative on short texts) x diversity, with at most 50% outliers and
at least 20 topics is written to ``<out>/<modality>/grid/selected.json`` (edit that file to override
the choice). The full run keeps
UMAP's parameters and ``min_samples`` and scales ``min_cluster_size`` by N_full / N_sample, so a
topic keeps the same minimum share of the corpus. Usage::

    python -m crypto_analysis.topic_search docs       --out-dir data/topics
    python -m crypto_analysis.topic_search embed-text --out-dir data/topics
    python -m crypto_analysis.topic_search grid       --out-dir data/topics --modality text
    python -m crypto_analysis.topic_search full       --out-dir data/topics --modality text
    python -m crypto_analysis.topic_search hierarchy  --out-dir data/topics --modality text
    python -m crypto_analysis.topic_search tokens     --out-dir data/topics
    python -m crypto_analysis.topic_search compare    --out-dir data/topics
    python -m crypto_analysis.topic_search lda        --out-dir data/topics --modality image
"""

import argparse
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd

from crypto_analysis.clip_embeddings import load_embeddings, token_text, topic_sample
from crypto_analysis.datalake import DATA_DIR, connect
from crypto_analysis.topic_alignment import symmetry_ratio

SEED = 42
MODALITIES = ["text", "image", "multimodal"]
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SAMPLE_DOCS = 100_000
REFERENCE_DOCS = 200_000
UMAP_GRID = {"n_neighbors": [15, 50], "n_components": [5, 10]}
HDBSCAN_GRID = {"min_cluster_size": [25, 50, 100, 200], "min_samples": [5, 25]}
MAX_OUTLIERS, MIN_TOPICS = 0.5, 20
PAPER_TOPIC_SHARE = 20 / 36_717  # min_topic_size / text subcorpus in Steffen (2025)
LDA_GRID = [50, 100, 200]
KNN_BUILD = {"knn_n_clusters": 4, "knn_overlap_factor": 2}
LARGE = 1_000_000  # above this many points the kNN graph is built in partitions (nn-descent)

_URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|io|xyz|gg|fun|org|net)\S*")
_WORD = re.compile(r"[a-z][a-z0-9]{1,24}")
_STOP = None


def stop_words():
    global _STOP
    if _STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOP = frozenset(ENGLISH_STOP_WORDS) | {"https", "http", "www", "com"}
    return _STOP


def tokenize(text):
    """Lower-case word tokens (2-25 chars, starting with a letter), URLs and stop words removed."""
    stop = stop_words()
    return [w for w in _WORD.findall(_URL.sub(" ", str(text).lower())) if w not in stop]


# --------------------------------------------------------------------------------------------
# Documents and embeddings
# --------------------------------------------------------------------------------------------

def build_documents(out_dir, data_dir=DATA_DIR):
    """Write the documents of every modality.

    - ``docs_image.parquet`` (sha256, doc): the distinct names of the tokens using the image;
    - ``docs_token.parquet`` (mint, sha256, text_key, doc): one row per token, ``text_key`` being
      the md5 of its text (None when the token has no text);
    - ``docs_text.parquet`` (text_key, doc): the distinct texts, sorted by key.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = connect(data_dir, threads=8)
    if not (out / "docs_image.parquet").exists():
        con.execute(f"""
            COPY (
                SELECT i.sha256,
                       array_to_string(list_slice(list(DISTINCT m.meta_name), 1, 20), ' ') AS doc
                FROM token_images i LEFT JOIN token_metadata m USING (mint)
                WHERE i.status = 'ok'
                GROUP BY 1
            ) TO '{out / "docs_image.parquet"}' (FORMAT parquet)
        """)
        print("docs_image.parquet written", flush=True)
    if not (out / "docs_token.parquet").exists():
        df = con.execute("""
            SELECT i.mint, i.sha256, m.meta_name, m.description
            FROM token_images i JOIN token_metadata m USING (mint)
            WHERE i.status = 'ok'
        """).df().drop_duplicates("mint")
        df["doc"] = [token_text(n, d, max_chars=600) for n, d in zip(df.meta_name, df.description)]
        df["text_key"] = [hashlib.md5(d.encode()).hexdigest() if d else None for d in df.doc]
        df[["mint", "sha256", "text_key", "doc"]].to_parquet(out / "docs_token.parquet", index=False)
        print("docs_token.parquet written", flush=True)
    if not (out / "docs_text.parquet").exists():
        df = pd.read_parquet(out / "docs_token.parquet", columns=["text_key", "doc"])
        df = df.dropna().drop_duplicates("text_key").sort_values("text_key")
        df.to_parquet(out / "docs_text.parquet", index=False)
        print(f"docs_text.parquet written: {len(df):,} distinct texts", flush=True)


def embed_text_docs(out_dir, batch_size=1024, model_name=TEXT_MODEL, file_name="text_minilm.npy"):
    """``file_name``: L2-normalised float16 embeddings of ``docs_text.parquet`` by ``model_name``, row-aligned."""
    from sentence_transformers import SentenceTransformer

    out = Path(out_dir)
    path = out / file_name
    if path.exists():
        return
    docs = pd.read_parquet(out / "docs_text.parquet").doc.tolist()
    model = SentenceTransformer(model_name, device="cuda").half()
    start = time.time()
    emb = model.encode(docs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(path, emb.astype(np.float16))
    print(f"{len(docs):,} texts embedded with {model_name} ({(time.time() - start) / 60:.1f} min)", flush=True)


def load_modality(out_dir, modality, clip_dir, embeddings=True):
    """Documents and embeddings of one modality, row-aligned: ``(keys, docs, X)``.

    Keys: ``text_key`` (text), ``sha256`` (image), ``"<sha256>:<text_key>"`` (multimodal). Only
    documents with an embedding are kept; ``embeddings=False`` still applies that filter but returns
    ``X = None``.
    """
    out = Path(out_dir)
    if modality == "text":
        docs = pd.read_parquet(out / "docs_text.parquet")
        X = np.load(out / "text_minilm.npy", mmap_mode="r") if embeddings else None
        return docs.text_key.to_numpy(), docs.doc.tolist(), None if X is None else np.asarray(X)
    img_keys, img_emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    img_index = pd.Index(img_keys)
    if modality == "image":
        docs = pd.read_parquet(out / "docs_image.parquet")
        rows = img_index.get_indexer(docs.sha256)
        docs = docs[rows >= 0]
        X = img_emb[rows[rows >= 0]] if embeddings else None
        return docs.sha256.to_numpy(), docs.doc.fillna("").tolist(), X
    txt_keys, txt_emb = load_embeddings(Path(clip_dir) / "texts", "mint")
    docs = pd.read_parquet(out / "docs_token.parquet").dropna(subset=["text_key"])
    docs = docs.drop_duplicates(["sha256", "text_key"])  # one document per distinct (image, text) pair
    ir, tr = img_index.get_indexer(docs.sha256), pd.Index(txt_keys).get_indexer(docs.mint)
    ok = (ir >= 0) & (tr >= 0)
    docs, ir, tr = docs[ok], ir[ok], tr[ok]
    keys = (docs.sha256 + ":" + docs.text_key).to_numpy()
    if not embeddings:
        return keys, docs.doc.tolist(), None
    X = np.empty((len(docs), img_emb.shape[1]), dtype=np.float16)
    for s in range(0, len(docs), 1_000_000):  # chunked: the float32 sum of 5M rows would need 10 GB
        part = img_emb[ir[s:s + 1_000_000]].astype(np.float32) + txt_emb[tr[s:s + 1_000_000]].astype(np.float32)
        X[s:s + 1_000_000] = part / np.linalg.norm(part, axis=1, keepdims=True)
    return keys, docs.doc.tolist(), X


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def reference(docs, dictionary=None, n=REFERENCE_DOCS, seed=SEED):
    """Tokenized reference sample and its gensim dictionary, for NPMI coherence."""
    from gensim.corpora import Dictionary

    rng = np.random.default_rng(seed)
    sample = [tokenize(docs[i]) for i in rng.choice(len(docs), size=min(n, len(docs)), replace=False)]
    sample = [t for t in sample if t]
    return sample, dictionary or Dictionary(sample)


def score_topics(topic_words, ref_texts, ref_dict, n_docs, n_outliers):
    """NPMI coherence (top 10), diversity (top 25), number of topics and outlier rate."""
    from gensim.models.coherencemodel import CoherenceModel

    known = [[w for w in words if w in ref_dict.token2id][:10] for words in topic_words]
    known = [k for k in known if len(k) >= 3]
    npmi = CoherenceModel(topics=known, texts=ref_texts, dictionary=ref_dict, coherence="c_npmi",
                          topn=10, processes=8).get_coherence() if known else np.nan
    top25 = [w for words in topic_words for w in words[:25]]
    return {
        "npmi": float(npmi),
        "diversity": len(set(top25)) / max(len(top25), 1),
        "n_topics": len(topic_words),
        "outlier_rate": n_outliers / n_docs,
    }


# --------------------------------------------------------------------------------------------
# LDA (CPU)
# --------------------------------------------------------------------------------------------

def run_lda(out_dir, modality, clip_dir, grid=LDA_GRID, workers=8):
    from gensim import utils
    from gensim.corpora import Dictionary, MmCorpus
    from gensim.models import LdaMulticore

    out = Path(out_dir) / modality
    out.mkdir(parents=True, exist_ok=True)
    keys, docs, _ = load_modality(out_dir, modality, clip_dir, embeddings=False)
    corpus_path, dict_path = out / "lda_corpus.mm", out / "lda_dictionary.dict"
    if not corpus_path.exists():
        dictionary = Dictionary(tokenize(d) for d in docs)
        dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=100_000)
        dictionary.save(str(dict_path))
        MmCorpus.serialize(str(corpus_path), (dictionary.doc2bow(tokenize(d)) for d in docs))
        print(f"corpus: {len(docs):,} docs, {len(dictionary):,} words", flush=True)
    dictionary, corpus = Dictionary.load(str(dict_path)), MmCorpus(str(corpus_path))
    ref_texts, ref_dict = reference(docs)
    results_path = out / "results_lda.parquet"
    done = set(pd.read_parquet(results_path).run) if results_path.exists() else set()
    for k in grid:
        run = f"lda_k{k}"
        if run in done:
            continue
        start = time.time()
        lda = LdaMulticore(corpus, id2word=dictionary, num_topics=k, passes=2, chunksize=20_000,
                           workers=workers, random_state=SEED, eval_every=None)
        labels = np.full(len(docs), -1, dtype=np.int32)
        pos = 0
        for chunk in utils.grouper(corpus, 20_000):  # dominant topic per document, batched inference
            gamma, _ = lda.inference(chunk)
            empty = np.array([len(bow) == 0 for bow in chunk])
            labels[pos:pos + len(chunk)] = np.where(empty, -1, gamma.argmax(axis=1))
            pos += len(chunk)
        words = [[w for w, _ in lda.show_topic(t, topn=25)] for t in range(k)]
        row = {"run": run, "model": "LDA", "k": k,
               **score_topics(words, ref_texts, ref_dict, len(docs), int((labels == -1).sum())),
               "minutes": (time.time() - start) / 60}
        pd.DataFrame({"key": keys, "topic": labels}).to_parquet(out / f"{run}.parquet", index=False)
        (out / f"{run}_words.json").write_text(json.dumps(words))
        lda.save(str(out / f"{run}.model"))
        _append(results_path, row)
        print(row, flush=True)


# --------------------------------------------------------------------------------------------
# BERTopic (GPU, cuML)
# --------------------------------------------------------------------------------------------

def gpu_free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def wait_for_gpu(min_free_gb, stable_minutes=10, poll_seconds=60):
    """Block until at least ``min_free_gb`` of GPU memory has been free for ``stable_minutes``."""
    free_since = None
    while True:
        free = gpu_free_mb() / 1024
        now = time.time()
        if free >= min_free_gb:
            free_since = free_since or now
            if now - free_since >= stable_minutes * 60:
                print(f"GPU free ({free:.1f} GB) for {stable_minutes} min: starting", flush=True)
                return
        else:
            free_since = None
        time.sleep(poll_seconds)


def _host(array):
    import cupy as cp

    return cp.asnumpy(array) if hasattr(array, "__cuda_array_interface__") else np.asarray(array)


def _knn_kwds(n):
    # nn-descent kNN graph built over 4 overlapping partitions of the host data, so millions of
    # 512-d points fit on one GPU
    return {"build_algo": "nn_descent", "build_kwds": KNN_BUILD} if n > LARGE else {}


def umap_reduce(X, n_neighbors, n_components):
    from cuml.manifold import UMAP

    return _host(UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric="cosine",
                      random_state=SEED, **_knn_kwds(len(X))).fit_transform(X.astype(np.float32)))


def hdbscan_labels(reduced, min_cluster_size, min_samples):
    from cuml.cluster import HDBSCAN

    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method="eom",
                     **_knn_kwds(len(reduced))).fit_predict(reduced)
    return _host(labels).astype(np.int32)


def fit_labels(docs, labels, n=25):
    """BERTopic with the clustering given (no embedding, UMAP or HDBSCAN step), and the Series mapping
    each given label to BERTopic's topic id (BERTopic renumbers topics by size)."""
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sklearn.feature_extraction.text import CountVectorizer

    # no min_df: BERTopic fits the vectorizer on one document per topic, so min_df would count topics
    # and drop the words specific to a few topics
    model = BERTopic(umap_model=BaseDimensionalityReduction(), hdbscan_model=BaseCluster(),
                     vectorizer_model=CountVectorizer(analyzer=tokenize), top_n_words=n)
    model.fit(docs, embeddings=np.zeros((len(docs), 1)), y=labels)
    return model, pd.Series(model.topics_).groupby(np.asarray(labels)).first()


def bertopic_words(docs, labels, n=25):
    """c-TF-IDF words of each cluster via BERTopic with the clustering given: ``{label: words}``."""
    model, renumbered = fit_labels(docs, labels, n)
    return {int(t): [w for w, _ in model.get_topic(int(renumbered[t])) if w] for t in renumbered.index if t != -1}


def representatives(X, labels, k=9, chunk=1_000_000):
    """The ``k`` documents of each topic closest (cosine) to the topic's mean embedding:
    DataFrame (topic, row, similarity, rank), ``row`` indexing ``X``."""
    import torch

    rows = np.flatnonzero(labels != -1)
    lab = labels[rows].astype(np.int64)
    C = torch.zeros(int(labels.max()) + 1, X.shape[1], device="cuda")
    for s in range(0, len(rows), chunk):
        C.index_add_(0, torch.from_numpy(lab[s:s + chunk]).cuda(),
                     torch.from_numpy(X[rows[s:s + chunk]]).cuda().float())
    C = torch.nn.functional.normalize(C, dim=1)
    sim = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        part = torch.from_numpy(X[rows[s:s + chunk]]).cuda().float()
        sim[s:s + chunk] = (part * C[torch.from_numpy(lab[s:s + chunk]).cuda()]).sum(dim=1).cpu().numpy()
    df = pd.DataFrame({"topic": lab, "row": rows, "similarity": sim})
    df = df.sort_values(["topic", "similarity"], ascending=[True, False]).groupby("topic").head(k)
    df["rank"] = df.groupby("topic").cumcount()
    return df.reset_index(drop=True)


def _grid_runs(nn, nc):
    return [(f"bertopic_nn{nn}_nc{nc}_mcs{m}_ms{s}", m, s)
            for m, s in product(HDBSCAN_GRID["min_cluster_size"], HDBSCAN_GRID["min_samples"])]


def run_grid(out_dir, modality, clip_dir, sample=SAMPLE_DOCS):
    """UMAP x HDBSCAN grid on a sample of ``sample`` documents, then ``select_config``.

    The image sample is ``clip_embeddings.topic_sample``, i.e. the captioned images.
    """
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    ref = reference(docs)  # from the full corpus, shared with the full run
    rows = np.sort(pd.Index(keys).get_indexer(topic_sample(keys, sample, SEED)))
    return grid_search(keys[rows], [docs[i] for i in rows], X[rows], ref, Path(out_dir) / modality / "grid", modality)


def grid_search(keys, docs, X, ref, out, label):
    """Run the grid on the given documents, writing every run and ``results.parquet`` to ``out``, and
    return ``select_config(out)``. ``ref``: the ``reference`` sample of the modality's full corpus."""
    out.mkdir(parents=True, exist_ok=True)
    ref_texts, ref_dict = ref
    print(f"{label}: grid on {len(docs):,} sampled documents", flush=True)
    results_path = out / "results.parquet"
    done = set(pd.read_parquet(results_path).run) if results_path.exists() else set()
    for nn, nc in product(UMAP_GRID["n_neighbors"], UMAP_GRID["n_components"]):
        runs = [r for r in _grid_runs(nn, nc) if r[0] not in done]
        if not runs:
            continue
        start = time.time()
        reduced = umap_reduce(X, nn, nc)
        umap_minutes = (time.time() - start) / 60
        for run, mcs, ms in runs:
            start = time.time()
            labels = hdbscan_labels(reduced, mcs, ms)
            words = bertopic_words(docs, labels)
            row = {"run": run, "model": "BERTopic", "n_neighbors": nn, "n_components": nc,
                   "min_cluster_size": mcs, "min_samples": ms, "sample_docs": len(docs),
                   **score_topics(list(words.values()), ref_texts, ref_dict, len(docs), int((labels == -1).sum())),
                   "umap_minutes": umap_minutes, "minutes": (time.time() - start) / 60}
            pd.DataFrame({"key": keys, "topic": labels}).to_parquet(out / f"{run}.parquet", index=False)
            (out / f"{run}_words.json").write_text(json.dumps(words))
            _append(results_path, row)
            print(row, flush=True)
    return select_config(out)


def select_config(out):
    """Write ``<out>/selected.json`` (``out``: a grid directory): among the runs at the paper's
    granularity, the one with the highest normalised NPMI x diversity, with at most ``MAX_OUTLIERS``
    outliers and ``MIN_TOPICS`` topics (all runs at that granularity if none qualifies)."""
    results = pd.read_parquet(out / "results.parquet")
    results["quality"] = (results.npmi + 1) / 2 * results.diversity
    target = PAPER_TOPIC_SHARE * results.sample_docs.iloc[0]
    paper_mcs = results.min_cluster_size.iloc[(results.min_cluster_size - target).abs().argmin()]
    results = results[results.min_cluster_size == paper_mcs]
    eligible = results[(results.outlier_rate <= MAX_OUTLIERS) & (results.n_topics >= MIN_TOPICS)]
    best = (eligible if len(eligible) else results).sort_values("quality", ascending=False).iloc[0]
    keep = ["run", "n_neighbors", "n_components", "min_cluster_size", "min_samples", "sample_docs",
            "npmi", "diversity", "quality", "n_topics", "outlier_rate"]
    selected = {k: best[k].item() if hasattr(best[k], "item") else best[k] for k in keep}
    selected["eligible"] = bool(len(eligible))
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    cols = ["run", "npmi", "diversity", "quality", "n_topics", "outlier_rate"]
    print(results.sort_values("quality", ascending=False)[cols].head(10).to_string(index=False))
    print(f"selected: {selected}", flush=True)
    return selected


def run_full(out_dir, modality, clip_dir, wait_gpu_gb=None):
    """Fit the selected configuration on the full corpus of ``modality``.

    Writes to ``<out>/<modality>/``: ``full_topics.parquet`` (key, topic), ``full_words.json``,
    ``full_info.parquet`` (topic, count, words and, for images, caption words),
    ``full_representatives.parquet`` (9 most central documents per topic) and a row of
    ``results_full.parquet`` with the same metrics as the grid.
    """
    if wait_gpu_gb:
        wait_for_gpu(wait_gpu_gb)
    out = Path(out_dir) / modality
    selected = json.loads((out / "grid" / "selected.json").read_text())
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    ref_texts, ref_dict = reference(docs)
    nn, nc, ms = selected["n_neighbors"], selected["n_components"], selected["min_samples"]
    mcs = max(selected["min_cluster_size"], round(selected["min_cluster_size"] * len(docs) / selected["sample_docs"]))
    print(f"{modality}: {len(docs):,} documents; UMAP nn={nn} nc={nc}; HDBSCAN min_cluster_size={mcs} "
          f"(grid {selected['min_cluster_size']} x {len(docs) / selected['sample_docs']:.1f}), min_samples={ms}",
          flush=True)
    reduced_path = out / f"umap_nn{nn}_nc{nc}.npy"
    start = time.time()
    if reduced_path.exists():
        reduced = np.load(reduced_path)
    else:
        reduced = umap_reduce(X, nn, nc)
        np.save(reduced_path, reduced)
        print(f"UMAP: {(time.time() - start) / 60:.1f} min", flush=True)
    labels = hdbscan_labels(reduced, mcs, ms)
    words = bertopic_words(docs, labels)
    row = {"run": f"full_nn{nn}_nc{nc}_mcs{mcs}_ms{ms}", "selected": selected["run"], "model": "BERTopic",
           "n_neighbors": nn, "n_components": nc, "min_cluster_size": mcs, "min_samples": ms, "docs": len(docs),
           **score_topics(list(words.values()), ref_texts, ref_dict, len(docs), int((labels == -1).sum())),
           "minutes": (time.time() - start) / 60}
    pd.DataFrame({"key": keys, "topic": labels}).to_parquet(out / "full_topics.parquet", index=False)
    (out / "full_words.json").write_text(json.dumps(words))
    counts = pd.Series(labels).value_counts()
    info = pd.DataFrame({"topic": counts.index, "count": counts.to_numpy()})
    info["words"] = [" ".join(words.get(t, [])[:10]) for t in info.topic]
    if modality == "image" and (Path(clip_dir) / "captions.parquet").exists():
        captions = pd.read_parquet(Path(clip_dir) / "captions.parquet").drop_duplicates("sha256").set_index("sha256")
        caption = captions.caption.reindex(keys).to_numpy()
        has = pd.notna(caption)
        caption_words = bertopic_words(caption[has].tolist(), labels[has])
        info["caption_words"] = [" ".join(caption_words.get(t, [])[:10]) for t in info.topic]
    info.to_parquet(out / "full_info.parquet", index=False)
    reps = representatives(X, labels)
    reps["key"] = keys[reps.row]
    reps["doc"] = [docs[i][:300] for i in reps.row]
    reps.drop(columns="row").to_parquet(out / "full_representatives.parquet", index=False)
    _append(out / "results_full.parquet", row)
    print(row, flush=True)


def topic_hierarchy(out_dir, modality, clip_dir):
    """BERTopic's hierarchy of the full-database topics (Ward linkage on their c-TF-IDF vectors).

    Writes ``full_hierarchy.parquet`` (one row per merge; leaf ids and ``Topics`` are the ids of
    ``full_topics.parquet``, merges are numbered from the number of topics up) and the interactive
    dendrogram ``full_hierarchy.html``.
    """
    out = Path(out_dir) / modality
    keys, docs, _ = load_modality(out_dir, modality, clip_dir, embeddings=False)
    labels = pd.read_parquet(out / "full_topics.parquet").set_index("key").topic.reindex(keys).to_numpy()
    model, renumbered = fit_labels(docs, labels)
    original = {int(new): int(t) for t, new in renumbered.items() if t != -1}
    words = json.loads((out / "full_words.json").read_text())
    model.set_topic_labels({new: f"{t}: {' '.join(words[str(t)][:4])}" for new, t in original.items()})
    hierarchy = model.hierarchical_topics(docs)
    fig = model.visualize_hierarchy(hierarchical_topics=hierarchy, custom_labels=True,
                                    width=1200, height=max(800, 18 * len(original)))
    fig.write_html(out / "full_hierarchy.html")
    for col in ["Child_Left_ID", "Child_Right_ID"]:
        hierarchy[col] = [original.get(int(c), int(c)) for c in hierarchy[col]]
    hierarchy["Topics"] = [[original[t] for t in topics] for topics in hierarchy.Topics]
    hierarchy.to_parquet(out / "full_hierarchy.parquet", index=False)
    print(f"{modality}: hierarchy of {len(original)} topics ({len(hierarchy)} merges) written", flush=True)
    return hierarchy


# --------------------------------------------------------------------------------------------
# Tokens across modalities
# --------------------------------------------------------------------------------------------

def token_topics(out_dir):
    """``token_topics.parquet``: one row per token with its topic in every modality fitted so far
    (missing when the token has no document in that modality)."""
    out = Path(out_dir)
    tok = pd.read_parquet(out / "docs_token.parquet", columns=["mint", "sha256", "text_key"])
    key = {"text": tok.text_key, "image": tok.sha256, "multimodal": tok.sha256 + ":" + tok.text_key}
    for modality in MODALITIES:
        path = out / modality / "full_topics.parquet"
        if path.exists():
            topics = pd.read_parquet(path).set_index("key").topic
            tok[f"{modality}_topic"] = topics.reindex(key[modality]).astype("Int32").to_numpy()
    tok.to_parquet(out / "token_topics.parquet", index=False)
    print(f"token_topics.parquet: {len(tok):,} tokens, columns {list(tok.columns)}", flush=True)
    return tok


def compare_modalities(out_dir):
    """Symmetry ratio of every pair of modalities over the tokens (the paper's Table 6)."""
    out = Path(out_dir)
    tok = pd.read_parquet(out / "token_topics.parquet").set_index("mint")
    fitted = [m for m in MODALITIES if f"{m}_topic" in tok]
    rows = []
    for a, b in combinations(fitted, 2):
        ratio_ab, _ = symmetry_ratio(tok[f"{a}_topic"], tok[f"{b}_topic"])
        ratio_ba, _ = symmetry_ratio(tok[f"{b}_topic"], tok[f"{a}_topic"])
        rows += [{"source": a, "target": b, "symmetry_ratio": ratio_ab},
                 {"source": b, "target": a, "symmetry_ratio": ratio_ba}]
    df = pd.DataFrame(rows)
    df.to_parquet(out / "symmetry.parquet", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def _append(path, row):
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["docs", "embed-text", "grid", "full", "hierarchy", "tokens", "compare", "lda"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=Path("data/clip"))
    parser.add_argument("--modality", choices=MODALITIES, default="text")
    parser.add_argument("--sample", type=int, default=SAMPLE_DOCS, help="grid: documents in the sample")
    parser.add_argument("--wait-gpu-gb", type=float, default=None,
                        help="full: wait until this much GPU memory has been free for 10 min")
    args = parser.parse_args()
    if args.step == "docs":
        build_documents(args.out_dir)
    elif args.step == "embed-text":
        embed_text_docs(args.out_dir)
    elif args.step == "grid":
        run_grid(args.out_dir, args.modality, args.clip_dir, args.sample)
    elif args.step == "full":
        run_full(args.out_dir, args.modality, args.clip_dir, args.wait_gpu_gb)
    elif args.step == "hierarchy":
        topic_hierarchy(args.out_dir, args.modality, args.clip_dir)
    elif args.step == "tokens":
        token_topics(args.out_dir)
    elif args.step == "compare":
        compare_modalities(args.out_dir)
    else:
        run_lda(args.out_dir, args.modality, args.clip_dir)


if __name__ == "__main__":
    main()
