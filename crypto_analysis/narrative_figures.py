"""Figures of the checkpoint-1 narratives and the checkpoint-2 pilot, for the group doc and the paper.

Writes to ``reports/figures/narratives/``:

* ``distribution.csv``: share of texts in topics per narrative, by rank, with the cumulative share, and
  ``narrative_coverage.pdf``/``.png``: that cumulative share as a curve, in the project's LaTeX style
  (``crypto_analysis.plotting``);
* ``top.csv``: the 32 largest narratives (half of the texts in topics) with size, topics and their most
  distinctive words against all other texts in topics, by the weighted log-odds ratio with an
  informative Dirichlet prior (Monroe, Colaresi and Quinn, 2008), slurs masked for display;
* ``images_<first>-<last>.png``: for each top narrative, images that cover its region of the CLIP image
  space (k-means medoids weighted by coins, the coverage sampling of the topic cards), screened out
  when CLIP zero-shot prompts read them as nudity, gore or hate symbols;
* ``pilot.csv``: attribute prevalence in the pilot's uniform stratum, from text alone and from text with
  the image (checkpoint 2, LLM labels).

Usage::

    python -m crypto_analysis.narrative_figures --out-dir data/topics --clip-dir data/clip
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

CHECKPOINT = "checkpoints/2026-11-01-narratives"
FIGURES = Path("reports/figures/narratives")
TOP = 32
WORDS = 8
PRIOR = 10_000  # total pseudo-count of the Dirichlet prior, spread by corpus frequency
IMAGES = 6
IMAGE_SAMPLE = 20_000
SEED = 42

STOP = {"coin", "coins", "token", "tokens", "sol", "solana", "pump", "fun", "pumpfun", "memecoin", "meme", "memes",
        "community", "ca", "https", "http", "www", "com", "the", "of", "and", "to", "is", "in", "on", "for", "it",
        "this", "that", "with", "you", "we", "are", "be", "will", "my", "your", "our", "an", "at", "by", "as",
        "from", "all", "just", "so", "not", "no", "me", "us", "its", "it's", "i'm", "first", "new", "official"}
SLURS = re.compile(r"\b(nigg\w*|fag\w*|retard\w*|kike\w*|chink\w*|spic|spics|tranny\w*)\b", re.IGNORECASE)

# CLIP zero-shot screen: an image whose probability mass falls on the first group is left out of the grids
SCREEN_OUT = ["a photo of a naked person", "pornography", "explicit sexual content", "a nude cartoon",
              "a swastika", "a nazi flag", "a bloody gore image", "a dead body", "an antisemitic caricature",
              "a racist caricature"]
SCREEN_IN = ["a cartoon character", "a photo of an animal", "a meme", "a logo", "a photo of a person",
             "text on a plain background", "a drawing", "a photo of food", "a crypto coin icon"]


def _mask(word):
    return SLURS.sub(lambda m: m.group()[0] + "*" * (len(m.group()) - 2) + m.group()[-1], word)


def narratives_table(out_dir):
    """Checkpoint-1 narratives by rank; ``name`` is the checkpoint-2 label name, else the provisional one."""
    from crypto_analysis.coin_labels import NARRATIVES

    ranked = pd.read_parquet(Path(out_dir) / CHECKPOINT / "narratives_ranked.parquet")
    ranked["name"] = [NARRATIVES.get(r, p) for r, p in zip(ranked["rank"], ranked.provisional_name)]
    ranked["share_pct"] = 100 * ranked.share
    ranked["cumulative_pct"] = 100 * ranked.cumulative
    return ranked


def distribution(ranked):
    return ranked[["rank", "topics", "docs", "share_pct", "cumulative_pct"]].round(4)


def coverage_figure(ranked, path, marks=(0.5, 0.8)):
    """Cumulative share of texts in topics against the number of narratives, largest first, with the
    number of narratives needed to reach each share in ``marks``."""
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import (
        COLORS,
        FIG_4x3,
        count_label,
        pct_label,
        percent_axis,
        save_figure,
        set_latex_style,
        tex_escape,
    )

    set_latex_style()
    fig, ax = plt.subplots(figsize=FIG_4x3)
    ax.step(ranked["rank"], ranked.cumulative, where="post", color=COLORS["blue"], lw=1.3)
    for share in marks:
        hit = ranked[ranked.cumulative >= share].iloc[0]
        ax.plot([1, hit["rank"], hit["rank"]], [hit.cumulative, hit.cumulative, 0], color=COLORS["muted"],
                lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.plot(hit["rank"], hit.cumulative, "o", color=COLORS["blue"], ms=4, zorder=3)
        ax.annotate(tex_escape(f"{hit['rank']} narratives: {hit.cumulative:.1%}"), (hit["rank"], hit.cumulative),
                    xytext=(6, -11), textcoords="offset points", fontsize=7.5, color=COLORS["ink"])
    ax.set_xlim(1, len(ranked))
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 50, 100, 150, 200, 250, len(ranked)])
    ax.set_xlabel(tex_escape(f"{count_label('narratives')}, largest first"))
    ax.set_ylabel(pct_label("Texts in topics covered"))
    percent_axis(ax)
    save_figure(fig, path)
    plt.close(fig)
    print(f"{path}.pdf / .png")


def distinctive_words(out_dir, ranked, k=WORDS, prior=PRIOR):
    """Top ``k`` words of each top narrative by the z-scored log-odds ratio against all other texts in topics."""
    from sklearn.feature_extraction.text import CountVectorizer

    root = Path(out_dir)
    topics = pd.read_parquet(root / "text" / "full_topics.parquet")
    docs = pd.read_parquet(root / "docs_text.parquet").merge(topics, left_on="text_key", right_on="key")
    docs = docs[docs.topic >= 0]
    narrative = {t: r for r, ts in zip(ranked["rank"], ranked.topic_ids) for t in ts}
    docs["rank"] = docs.topic.map(narrative)
    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[^\W\d_][\w']+\b", min_df=20, binary=True,
                          stop_words=sorted(STOP))
    X = vec.fit_transform(docs.doc.fillna(""))
    vocab = np.array(vec.get_feature_names_out())
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    alpha = prior * total / total.sum()
    words = {}
    for rank in range(1, TOP + 1):
        mask = (docs["rank"] == rank).to_numpy()
        y_i = np.asarray(X[mask].sum(axis=0)).ravel().astype(float)
        y_j = total - y_i
        n_i, n_j = y_i.sum(), y_j.sum()
        delta = (np.log((y_i + alpha) / (n_i + prior - y_i - alpha))
                 - np.log((y_j + alpha) / (n_j + prior - y_j - alpha)))
        z = delta / np.sqrt(1 / (y_i + alpha) + 1 / (y_j + alpha))
        words[rank] = ", ".join(_mask(w) for w in vocab[np.argsort(-z)[:k]])
    return pd.Series(words, name="words")


def _screen(img_emb):
    """True where CLIP zero-shot prompts put most probability on nudity, gore or hate symbols."""
    from sentence_transformers import SentenceTransformer

    text = SentenceTransformer("clip-ViT-B-32").encode(SCREEN_OUT + SCREEN_IN, normalize_embeddings=True)
    logits = 100 * img_emb @ text.T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p[:, :len(SCREEN_OUT)].sum(axis=1) > 0.5


def representative_images(out_dir, clip_dir, ranked, k=IMAGES, seed=SEED):
    """{rank: [sha256, ...]}: ``k`` k-means medoids of each top narrative's coin images (weighted by coins),
    largest cluster first, skipping images the screen leaves out."""
    from sklearn.cluster import KMeans

    from crypto_analysis.clip_embeddings import load_embeddings

    rng = np.random.default_rng(seed)
    keys, emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    index = pd.Index(keys)
    coins = pd.read_parquet(Path(out_dir) / "token_topics.parquet", columns=["sha256", "text_topic"]).dropna()
    narrative = {t: r for r, ts in zip(ranked["rank"], ranked.topic_ids) for t in ts}
    coins["rank"] = coins.text_topic.astype(int).map(narrative)
    picked = {}
    for rank in range(1, TOP + 1):
        sha = coins.sha256[coins["rank"] == rank].to_numpy()
        sha = sha[rng.choice(len(sha), min(len(sha), IMAGE_SAMPLE), replace=False)]
        counts = pd.Series(sha).value_counts()
        rows = index.get_indexer(counts.index)
        counts, rows = counts[rows >= 0], rows[rows >= 0]
        X = emb[rows].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        keep = ~_screen(X)
        counts, X = counts[keep], X[keep]
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X, sample_weight=counts.to_numpy())
        weight = np.bincount(km.labels_, weights=counts.to_numpy(), minlength=k)
        medoids = []
        for c in np.argsort(-weight):
            members = np.flatnonzero(km.labels_ == c)
            centre = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
            medoids.append(counts.index[members[np.argmax(X[members] @ centre)]])
        picked[rank] = medoids
        print(f"  narrative {rank}: {len(counts):,} distinct images, {int((~keep).sum())} screened out", flush=True)
    return picked


def image_grid(picked, ranked, first, last, path, tile=128, cols=4):
    """One panel per narrative (title + 3 x 2 images), ``cols`` panels per row."""
    from PIL import Image, ImageDraw, ImageFont

    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import _rgb

    shas = [s for r in range(first, last + 1) for s in picked[r]]
    images = load_images(connect(DATA_DIR, threads=8), shas, DATA_DIR / "images")
    pad, head = 12, 44
    panel_w, panel_h = 3 * tile + 2 * 4, head + 2 * tile + 4
    n = last - first + 1
    rows = -(-n // cols)
    canvas = Image.new("RGB", (cols * panel_w + (cols + 1) * pad, rows * panel_h + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    bold, small = ImageFont.load_default(size=15), ImageFont.load_default(size=13)
    names = ranked.set_index("rank")
    for i, rank in enumerate(range(first, last + 1)):
        x0, y0 = pad + (i % cols) * (panel_w + pad), pad + (i // cols) * (panel_h + pad)
        name = names.name[rank]
        name = name if len(name) <= 44 else name[:42] + "…"
        draw.text((x0, y0), f"{rank}. {name}", fill="#1d1d1b", font=bold)
        draw.text((x0, y0 + 20), f"{names.share_pct[rank]:.1f}% of texts · {names.topics[rank]} topic(s)",
                  fill="#6b6b66", font=small)
        for j, sha in enumerate(picked[rank]):
            tx, ty = x0 + (j % 3) * (tile + 4), y0 + head + (j // 3) * (tile + 4)
            img = images.get(sha)
            if img is None:
                draw.rectangle([tx, ty, tx + tile, ty + tile], outline="#dcdcd6")
                continue
            img = _rgb(img)
            img.thumbnail((tile, tile))
            canvas.paste(img, (tx + (tile - img.width) // 2, ty + (tile - img.height) // 2))
    canvas.save(path, optimize=True)
    print(f"{path} ({canvas.width}x{canvas.height})")


def pilot_table(out_dir):
    """Uniform-stratum prevalence of each attribute from text alone and from text with the image."""
    from crypto_analysis.coin_labels import ATTRIBUTES, RACE_TARGETS, _wilson

    pilot = Path(out_dir) / "coin_labels" / "pilot"
    sample = pd.read_parquet(pilot / "sample.parquet")
    uniform = sample.mint[sample.stratum == "uniform"]
    rows = []
    for condition, label in (("text", "text only"), ("multimodal", "text + image")):
        labels = pd.read_parquet(pilot / f"labels_{condition}.parquet").set_index("mint").loc[uniform]
        labels["racist"] = labels.hate & labels.hate_targets.map(lambda t: any(x in t for x in RACE_TARGETS))
        labels["antisemitic"] = labels.hate & labels.antisemitism.notna()
        for attr in [*ATTRIBUTES, "racist", "antisemitic"]:
            k, n = int(labels[attr].sum()), len(labels)
            lo, hi = _wilson(k, n)
            rows.append({"attribute": attr, "condition": label, "coins": k, "n": n, "pct": round(100 * k / n, 2),
                         "ci_low": round(100 * lo, 2), "ci_high": round(100 * hi, 2)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--coverage-only", action="store_true", help="only the distribution CSV and figure")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ranked = narratives_table(args.out_dir)
    distribution(ranked).to_csv(FIGURES / "distribution.csv", index=False)
    coverage_figure(ranked, FIGURES / "narrative_coverage")
    if args.coverage_only:
        return
    top = ranked.head(TOP)[["rank", "name", "topics", "docs", "share_pct", "cumulative_pct"]].copy()
    top["words"] = top["rank"].map(distinctive_words(args.out_dir, ranked))
    top.round(3).to_csv(FIGURES / "top.csv", index=False)
    pilot_table(args.out_dir).to_csv(FIGURES / "pilot.csv", index=False)
    print(top[["rank", "name", "words"]].to_string(index=False))
    if not args.skip_images:
        picked = representative_images(args.out_dir, args.clip_dir, ranked)
        pd.DataFrame([{"rank": r, "order": i, "sha256": s} for r, ss in picked.items() for i, s in enumerate(ss)]
                     ).to_csv(FIGURES / "images.csv", index=False)
        image_grid(picked, ranked, 1, 16, FIGURES / "images_01-16.png")
        image_grid(picked, ranked, 17, 32, FIGURES / "images_17-32.png")


if __name__ == "__main__":
    main()
