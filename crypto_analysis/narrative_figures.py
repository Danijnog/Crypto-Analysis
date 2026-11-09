"""Figures of the multimodal narratives (published definition) and of the coin-label pilot, for the group
doc and the paper. Coins are placed in a narrative by their multimodal topic after outlier reduction
(``topic_sampling reduce-full``), so practically every coin is counted.

Writes to ``reports/figures/narratives/``:

* ``distribution.csv``: share of coins per narrative, by rank, with the cumulative share, and
  ``narrative_coverage.pdf``/``.png``: that cumulative share as a curve, in the project's LaTeX style
  (``crypto_analysis.plotting``);
* ``top.csv``: the head narratives (covering half of the coins) with name, size, topics and their most
  distinctive words against all other texts, by the weighted log-odds ratio with an informative Dirichlet
  prior (Monroe, Colaresi and Quinn, 2008), on the texts of the distinct (image, text) pairs, slurs masked
  for display. A name is the coin-label name (``coin_labels.NARRATIVES``) when the narrative is listed,
  else its provisional name: the LLM's shared narrative for merged topics, or the leaf theme(s) of its
  topics (``topic_taxonomy leaves``), "(no theme)" when none has one;
* ``images_<first>-<last>.png``: for each head narrative, images that cover its region of the CLIP image
  space (k-means medoids weighted by coins, the coverage sampling of the topic cards), each with the
  most frequent name of its coins (the share of the narrative in its region is in ``images.csv``), screened out
  when CLIP zero-shot prompts read them as nudity, gore or hate symbols (``images.csv`` lists them);
  with ``--blur-faces`` also ``images_<first>-<last>_blurred.png``, faces of non-public people blurred;
* ``pilot.csv``: attribute prevalence in the pilot's uniform stratum, from text alone and from text with
  the image (LLM labels).

Usage::

    python -m crypto_analysis.narrative_figures --out-dir data/topics --clip-dir data/clip
"""

import argparse
from functools import cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

FIGURES = Path("reports/figures/narratives")
HEAD = 0.5  # the head narratives cover this share of the coins
GRID = 16  # narratives per image grid
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
# narratives (coin-label ids) about people who are not public figures: their faces are pixelated in the
# ``--blur-faces`` grids, and a photo where no face is found is blurred whole
FACE_NARRATIVES = {7, 11, 43}
# images the screen misses (text slurs drawn in the image), left out of the grids by hand
EXCLUDED_IMAGES = {"93c3f7378046cd507a1c9755c7da169875abf2e70b8c5b10b7da59b0bf9c0bde"}
SCREEN_IN = ["a cartoon character", "a photo of an animal", "a meme", "a logo", "a photo of a person",
             "text on a plain background", "a drawing", "a photo of food", "a crypto coin icon"]


def _mask(word):
    return SLURS.sub(lambda m: m.group()[0] + "*" * (len(m.group()) - 2) + m.group()[-1], word)


def provisional_names(out_dir, ranked):
    """{narrative: name}: for merged topics the shared narrative the LLM named most often in its
    same-narrative pairs, else the distinct leaf themes of its topics, "(no theme)" when none has one."""
    from crypto_analysis.narratives import narrative_dir

    root = Path(out_dir)
    themes = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    pairs = pd.read_parquet(narrative_dir(root, "multimodal", "published") / "candidates.parquet")
    pairs = pairs[pairs.same.fillna(False).astype(bool)]
    topic_narrative = {t: n for n, ts in zip(ranked.narrative, ranked.topic_ids) for t in ts}
    # both orders of a same-narrative pair name the narrative its topics share
    owner = pd.Series(np.concatenate([pairs.a.to_numpy()] * 2)).map(topic_narrative)
    label = pd.Series(np.concatenate([pairs.narrative_ab.to_numpy(), pairs.narrative_ba.to_numpy()]))
    label = label.fillna("").astype(str).str.strip()
    keep = label != ""
    shared = label[keep].groupby(owner[keep]).agg(lambda x: x.value_counts().index[0])
    names = {}
    for narrative, topics in zip(ranked.narrative, ranked.topic_ids):
        found = list(dict.fromkeys(t for t in themes.reindex(topics) if isinstance(t, str)))
        names[narrative] = shared.get(narrative) or (" / ".join(found) if found else "(no theme)")
    return names


def narratives_table(out_dir):
    """Multimodal narratives ranked by coins; ``name`` is the coin-label name, else the provisional one."""
    from crypto_analysis.coin_labels import NARRATIVES
    from crypto_analysis.narratives import ranked_narratives

    ranked = ranked_narratives(out_dir)
    provisional = provisional_names(out_dir, ranked)
    ranked["provisional_name"] = ranked.narrative.map(provisional)
    ranked["name"] = [NARRATIVES.get(r, p) for r, p in zip(ranked["rank"], ranked.provisional_name)]
    ranked["share_pct"] = 100 * ranked.share
    ranked["cumulative_pct"] = 100 * ranked.cumulative
    return ranked


def head_size(ranked, share=HEAD):
    return int((ranked.cumulative < share).sum()) + 1


def distribution(ranked):
    return ranked[["rank", "narrative", "topics", "coins", "share_pct", "cumulative_pct", "provisional_name"]].round(4)


def coverage_figure(ranked, path, marks=(0.5, 0.8)):
    """Cumulative share of coins against the number of narratives, largest first, with the number of
    narratives needed to reach each share in ``marks``."""
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
    ax.set_ylabel(pct_label("Coins covered"))
    percent_axis(ax)
    save_figure(fig, path)
    plt.close(fig)
    print(f"{path}.pdf / .png")


def distinctive_words(out_dir, clip_dir, ranked, top, k=WORDS, prior=PRIOR):
    """Top ``k`` words of each of the ``top`` narratives by the z-scored log-odds ratio against all other
    texts of (image, text) pairs in a topic after outlier reduction."""
    from sklearn.feature_extraction.text import CountVectorizer

    from crypto_analysis.topic_search import load_modality

    root = Path(out_dir)
    keys, texts, _ = load_modality(root, "multimodal", clip_dir, embeddings=False)
    topics = pd.read_parquet(root / "multimodal" / "full_topics.parquet").set_index("key").topic_reduced
    docs = pd.DataFrame({"doc": texts, "topic": topics.reindex(keys).to_numpy()})
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
    for rank in range(1, top + 1):
        mask = (docs["rank"] == rank).to_numpy()
        y_i = np.asarray(X[mask].sum(axis=0)).ravel().astype(float)
        y_j = total - y_i
        n_i, n_j = y_i.sum(), y_j.sum()
        delta = (np.log((y_i + alpha) / (n_i + prior - y_i - alpha))
                 - np.log((y_j + alpha) / (n_j + prior - y_j - alpha)))
        z = delta / np.sqrt(1 / (y_i + alpha) + 1 / (y_j + alpha))
        words[rank] = ", ".join(_mask(w) for w in vocab[np.argsort(-z)[:k]])
    return pd.Series(words, name="words")


@cache
def _screen_prompts():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("clip-ViT-B-32").encode(SCREEN_OUT + SCREEN_IN, normalize_embeddings=True)


def _screen(img_emb):
    """True where CLIP zero-shot prompts put most probability on nudity, gore or hate symbols."""
    text = _screen_prompts()
    logits = 100 * img_emb @ text.T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p[:, :len(SCREEN_OUT)].sum(axis=1) > 0.5


def representative_images(out_dir, clip_dir, ranked, top, k=IMAGES, seed=SEED):
    """{rank: [(sha256, share), ...]}: ``k`` k-means medoids of the coin images of each of the ``top``
    narratives (weighted by coins), largest region first, with the share of the narrative's coins in each
    region, skipping images the screen leaves out. Each image stands for one region of the narrative
    (coverage), not for its centre."""
    from sklearn.cluster import KMeans

    from crypto_analysis.clip_embeddings import load_embeddings

    rng = np.random.default_rng(seed)
    keys, emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    index = pd.Index(keys)
    coins = pd.read_parquet(Path(out_dir) / "token_topics.parquet", columns=["sha256", "multimodal_topic_reduced"])
    coins = coins.dropna()
    narrative = {t: r for r, ts in zip(ranked["rank"], ranked.topic_ids) for t in ts}
    coins["rank"] = coins.multimodal_topic_reduced.astype(int).map(narrative)
    picked = {}
    for rank in range(1, top + 1):
        sha = coins.sha256[coins["rank"] == rank].to_numpy()
        sha = sha[rng.choice(len(sha), min(len(sha), IMAGE_SAMPLE), replace=False)]
        counts = pd.Series(sha).value_counts()
        rows = index.get_indexer(counts.index)
        counts, rows = counts[rows >= 0], rows[rows >= 0]
        X = emb[rows].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        keep = ~_screen(X) & ~counts.index.isin(EXCLUDED_IMAGES)
        counts, X = counts[keep], X[keep]
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X, sample_weight=counts.to_numpy())
        weight = np.bincount(km.labels_, weights=counts.to_numpy(), minlength=k)
        medoids = []
        for c in np.argsort(-weight):
            members = np.flatnonzero(km.labels_ == c)
            centre = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
            medoids.append((counts.index[members[np.argmax(X[members] @ centre)]], weight[c] / weight.sum()))
        picked[rank] = medoids
        print(f"  narrative {rank}: {len(counts):,} distinct images, {int((~keep).sum())} screened out", flush=True)
    return picked


def image_names(shas):
    """{sha256: name}: the most frequent name of the coins using each image, slurs masked."""
    from crypto_analysis.datalake import DATA_DIR, connect

    con = connect(DATA_DIR, threads=8)
    con.register("wanted", pd.DataFrame({"sha256": list(shas)}))
    names = con.execute("""
        SELECT i.sha256, trim(m.meta_name) AS name, count(*) AS n
        FROM wanted w JOIN token_images i USING (sha256) JOIN token_metadata m USING (mint)
        WHERE i.status = 'ok' AND trim(coalesce(m.meta_name, '')) <> ''
        GROUP BY 1, 2
    """).df()
    names = names.sort_values(["n", "name"], ascending=[False, True]).drop_duplicates("sha256")
    return {sha: _mask(name) for sha, name in zip(names.sha256, names.name)}


def blur_faces(img):
    """``img`` with every frontal face (OpenCV Haar cascade) pixelated, or blurred whole when none is found."""
    import cv2
    from PIL import Image, ImageFilter

    a = np.asarray(img)
    side = min(a.shape[:2])
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5,
                                      minSize=(max(20, side // 12),) * 2)
    if len(faces) == 0:
        return img.filter(ImageFilter.GaussianBlur(radius=max(4, side // 25)))
    out = img.copy()
    for x, y, w, h in faces:
        m = w // 4
        box = (max(0, x - m), max(0, y - m), min(a.shape[1], x + w + m), min(a.shape[0], y + h + m))
        region = out.crop(box)
        out.paste(region.resize((8, 8)).resize(region.size, Image.NEAREST), box)
    return out


def image_grid(picked, names, ranked, first, last, path, tile=150, cols=4, blur=()):
    """One panel per narrative (title + 3 x 2 images), ``cols`` panels per row; under each image the
    most frequent name of its coins. Faces are blurred (``blur_faces``) in the narratives whose rank is in
    ``blur``."""
    from PIL import Image, ImageDraw, ImageFont

    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import _rgb

    shas = [s for r in range(first, last + 1) for s, _ in picked[r]]
    images = load_images(connect(DATA_DIR, threads=8), shas, DATA_DIR / "images")
    pad, head, gap, caption = 14, 46, 6, 22
    cell_h = tile + caption
    panel_w, panel_h = 3 * tile + 2 * gap, head + 2 * cell_h + gap
    n = last - first + 1
    rows = -(-n // cols)
    canvas = Image.new("RGB", (cols * panel_w + (cols + 1) * pad, rows * panel_h + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    bold, small = ImageFont.load_default(size=15), ImageFont.load_default(size=13)
    label = ImageFont.load_default(size=12)
    table = ranked.set_index("rank")

    def fit(text, font, width):
        while text and draw.textlength(text, font=font) > width:
            text = text[:-2] + "…"
        return text

    for i, rank in enumerate(range(first, last + 1)):
        x0, y0 = pad + (i % cols) * (panel_w + pad), pad + (i // cols) * (panel_h + pad)
        draw.text((x0, y0), fit(f"{rank}. {table.name[rank]}", bold, panel_w), fill="#1d1d1b", font=bold)
        draw.text((x0, y0 + 20), f"{table.share_pct[rank]:.1f}% of coins · {table.topics[rank]} topic(s)",
                  fill="#6b6b66", font=small)
        for j, (sha, _) in enumerate(picked[rank]):
            tx, ty = x0 + (j % 3) * (tile + gap), y0 + head + (j // 3) * (cell_h + gap)
            img = images.get(sha)
            if img is None:
                draw.rectangle([tx, ty, tx + tile, ty + tile], outline="#dcdcd6")
            else:
                img = _rgb(img)
                img = blur_faces(img) if rank in blur else img
                img.thumbnail((tile, tile))
                canvas.paste(img, (tx + (tile - img.width) // 2, ty + (tile - img.height) // 2))
            draw.text((tx, ty + tile + 4), fit(names.get(sha, "(no name)"), label, tile), fill="#1d1d1b", font=label)
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
    parser.add_argument("--blur-faces", action="store_true",
                        help="also write the grids with the faces of FACE_NARRATIVES blurred (for the paper)")
    parser.add_argument("--coverage-only", action="store_true", help="only the distribution CSV and figure")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ranked = narratives_table(args.out_dir)
    distribution(ranked).to_csv(FIGURES / "distribution.csv", index=False)
    coverage_figure(ranked, FIGURES / "narrative_coverage")
    if args.coverage_only:
        return
    n = head_size(ranked)
    top = ranked.head(n)[["rank", "narrative", "name", "topics", "coins", "share_pct", "cumulative_pct"]].copy()
    top["words"] = top["rank"].map(distinctive_words(args.out_dir, args.clip_dir, ranked, n))
    top.round(3).to_csv(FIGURES / "top.csv", index=False)
    if (Path(args.out_dir) / "coin_labels" / "pilot" / "labels_multimodal.parquet").exists():
        pilot_table(args.out_dir).to_csv(FIGURES / "pilot.csv", index=False)
    print(top[["rank", "name", "words"]].to_string(index=False))
    if not args.skip_images:
        picked = representative_images(args.out_dir, args.clip_dir, ranked, n)
        names = image_names(s for ss in picked.values() for s, _ in ss)
        pd.DataFrame([{"rank": r, "order": i, "sha256": s, "region_share": round(share, 4), "name": names.get(s)}
                      for r, ss in picked.items() for i, (s, share) in enumerate(ss)]
                     ).to_csv(FIGURES / "images.csv", index=False)
        for first in range(1, n + 1, GRID):
            last = min(first + GRID - 1, n)
            image_grid(picked, names, ranked, first, last, FIGURES / f"images_{first:02d}-{last:02d}.png")
            if args.blur_faces:
                image_grid(picked, names, ranked, first, last, FIGURES / f"images_{first:02d}-{last:02d}_blurred.png",
                           blur=FACE_NARRATIVES)


if __name__ == "__main__":
    main()
