"""Topic-model selection on the full database: BERTopic on CLIP embeddings (GPU) vs LDA (CPU).

Two modalities, as in Steffen (ICWSM 2025):

- ``image``: one document per distinct image (2.7M); BERTopic clusters its CLIP image embedding.
- ``token``: one document per token (5.1M); BERTopic clusters the *multimodal* embedding, the
  normalised average of the token's CLIP image and text embeddings.

The words of every model come from the same documents and tokenizer, so coherence is comparable:
an image's document is the distinct names of the tokens using it; a token's document is its
"name. description" with URLs removed.

Every run is scored with NPMI coherence (top-10 words, on a 200k-document reference sample), topic
diversity (unique share of the top-25 words), number of topics and outlier rate, as in the BERTopic
paper (Grootendorst, 2022). Results and per-run topic labels go to ``<out>/<modality>/``, so the
grid is resumable and the chosen run is already a full-database model. Usage::

    python -m crypto_analysis.topic_search docs     --out-dir data/topics
    python -m crypto_analysis.topic_search lda      --out-dir data/topics --modality image
    python -m crypto_analysis.topic_search bertopic --out-dir data/topics --modality image --wait-gpu-gb 26
"""

import argparse
from itertools import product
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd

from crypto_analysis.clip_embeddings import load_embeddings, token_text
from crypto_analysis.datalake import DATA_DIR, connect

SEED = 42
REFERENCE_DOCS = 200_000
UMAP_GRID = {"n_neighbors": [15, 50], "n_components": [5, 10]}
HDBSCAN_GRID = {"min_cluster_size": [50, 200, 1000], "min_samples": [None, 10]}
LDA_GRID = [50, 100, 200]
KNN_BUILD = {"knn_n_clusters": 4, "knn_overlap_factor": 2}

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
# Documents
# --------------------------------------------------------------------------------------------

def build_documents(out_dir, data_dir=DATA_DIR):
    """Write ``docs_image.parquet`` (sha256, doc) and ``docs_token.parquet`` (mint, sha256, doc)."""
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
        df[["mint", "sha256", "doc"]].to_parquet(out / "docs_token.parquet", index=False)
        print("docs_token.parquet written", flush=True)


def load_modality(out_dir, modality, clip_dir, embeddings=True):
    """Documents and CLIP embeddings of one modality, row-aligned: ``(keys, docs, X)``.

    Only documents with an embedding are kept; ``embeddings=False`` still applies that filter but
    returns ``X = None``.
    """
    out = Path(out_dir)
    img_keys, img_emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    img_index = pd.Index(img_keys)
    if modality == "image":
        docs = pd.read_parquet(out / "docs_image.parquet")
        rows = img_index.get_indexer(docs.sha256)
        docs = docs[rows >= 0]
        X = img_emb[rows[rows >= 0]] if embeddings else None
        return docs.sha256.to_numpy(), docs.doc.fillna("").tolist(), X
    txt_keys, txt_emb = load_embeddings(Path(clip_dir) / "texts", "mint")
    docs = pd.read_parquet(out / "docs_token.parquet")
    ir, tr = img_index.get_indexer(docs.sha256), pd.Index(txt_keys).get_indexer(docs.mint)
    ok = (ir >= 0) & (tr >= 0)
    docs, ir, tr = docs[ok], ir[ok], tr[ok]
    if not embeddings:
        return docs.mint.to_numpy(), docs.doc.fillna("").tolist(), None
    X = img_emb[ir].astype(np.float32) + txt_emb[tr].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return docs.mint.to_numpy(), docs.doc.fillna("").tolist(), X.astype(np.float16)


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


def bertopic_words(docs, labels, n=25):
    """c-TF-IDF words of each cluster via BERTopic with the clustering given (manual BERTopic)."""
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sklearn.feature_extraction.text import CountVectorizer

    model = BERTopic(umap_model=BaseDimensionalityReduction(), hdbscan_model=BaseCluster(),
                     vectorizer_model=CountVectorizer(analyzer=tokenize, min_df=5), top_n_words=n)
    model.fit(docs, embeddings=np.zeros((len(docs), 1)), y=labels)
    topics = sorted(t for t in model.get_topics() if t != -1)
    return topics, [[w for w, _ in model.get_topic(t)] for t in topics]


def run_bertopic(out_dir, modality, clip_dir, wait_gpu_gb=None):
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    import cupy as cp

    if wait_gpu_gb:
        wait_for_gpu(wait_gpu_gb)
    out = Path(out_dir) / modality
    out.mkdir(parents=True, exist_ok=True)
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    ref_texts, ref_dict = reference(docs)
    results_path = out / "results_bertopic.parquet"
    done = set(pd.read_parquet(results_path).run) if results_path.exists() else set()
    for nn, nc in product(UMAP_GRID["n_neighbors"], UMAP_GRID["n_components"]):
        reduced_path = out / f"umap_nn{nn}_nc{nc}.npy"
        runs = [f"bertopic_nn{nn}_nc{nc}_mcs{m}_ms{s or 'd'}"
                for m, s in product(HDBSCAN_GRID["min_cluster_size"], HDBSCAN_GRID["min_samples"])]
        if all(r in done for r in runs):
            continue
        if reduced_path.exists():
            reduced = np.load(reduced_path)
        else:
            start = time.time()
            # nn-descent kNN graph built over 4 overlapping partitions of the host data, so millions of
            # 512-d points fit on one GPU
            reduced = UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine", random_state=SEED,
                           build_algo="nn_descent", build_kwds=KNN_BUILD).fit_transform(X.astype(np.float32))
            reduced = cp.asnumpy(reduced) if hasattr(reduced, "__cuda_array_interface__") else np.asarray(reduced)
            np.save(reduced_path, reduced)
            print(f"UMAP nn={nn} nc={nc}: {(time.time() - start) / 60:.1f} min", flush=True)
        for (mcs, ms), run in zip(product(HDBSCAN_GRID["min_cluster_size"], HDBSCAN_GRID["min_samples"]), runs):
            if run in done:
                continue
            start = time.time()
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="eom",
                             build_algo="nn_descent", build_kwds=KNN_BUILD).fit_predict(reduced)
            labels = cp.asnumpy(labels) if hasattr(labels, "__cuda_array_interface__") else np.asarray(labels)
            topics, words = bertopic_words(docs, labels)
            row = {"run": run, "model": "BERTopic", "n_neighbors": nn, "n_components": nc,
                   "min_cluster_size": mcs, "min_samples": ms or mcs,
                   **score_topics(words, ref_texts, ref_dict, len(docs), int((labels == -1).sum())),
                   "minutes": (time.time() - start) / 60}
            pd.DataFrame({"key": keys, "topic": labels.astype(np.int32)}).to_parquet(out / f"{run}.parquet",
                                                                                      index=False)
            (out / f"{run}_words.json").write_text(json.dumps(dict(zip(map(int, topics), words))))
            _append(results_path, row)
            print(row, flush=True)


def _append(path, row):
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["docs", "lda", "bertopic"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=Path("data/clip"))
    parser.add_argument("--modality", choices=["image", "token"], default="image")
    parser.add_argument("--wait-gpu-gb", type=float, default=None,
                        help="bertopic: wait until this much GPU memory has been free for 10 min")
    args = parser.parse_args()
    if args.step == "docs":
        build_documents(args.out_dir)
    elif args.step == "lda":
        run_lda(args.out_dir, args.modality, args.clip_dir)
    else:
        run_bertopic(args.out_dir, args.modality, args.clip_dir, args.wait_gpu_gb)


if __name__ == "__main__":
    main()
