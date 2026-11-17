"""Topic cards for annotating the full-database topics of ``topic_search`` (Steffen, ICWSM 2025).

A card is what both the human annotators and the LLM see for one topic: its c-TF-IDF words, a sample
of its documents and, for image and multimodal topics, a numbered collage of the sampled images.

The sample covers the topic's region of the embedding space instead of its centre: the topic's
documents are split into ``k`` k-means clusters in the modality's embedding space and the document
closest to each cluster centre is kept, ordered by cluster size and stored with that cluster's share
of the topic. The paper's representative documents (the most central ones) make every topic look
coherent; this sample also shows the variety, and the parts that make a topic incoherent.

Writes ``<out>/annotation/<modality>/cards.parquet`` (topic, count, words, rank, key, doc, share) and,
for image and multimodal, ``collages/<topic>.jpg``. ``pilot`` then builds the open-coding page from
which the annotation guide is derived, and ``open`` asks the LLM (Qwen3.5-9B, greedy, no thinking)
the pilot's open questions about every topic. Usage::

    python -m crypto_analysis.topic_annotation cards --out-dir data/topics --modality text
    python -m crypto_analysis.topic_annotation pilot --out-dir data/topics
    python -m crypto_analysis.topic_annotation open  --out-dir data/topics --all-modalities
"""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from crypto_analysis.datalake import DATA_DIR, connect
from crypto_analysis.gallery import load_images
from crypto_analysis.topic_search import MODALITIES, SEED, load_modality

CARD_SAMPLES = 12
PILOT_TOPICS = 20
MAX_FIT = 20_000  # k-means is fitted on a uniform subsample of larger topics
TILE, COLLAGE_COLS = 256, 4
LLM = "/home/datalake/models/Qwen3.5-9B"

# Open coding: the questions of the human pilot page, no categories yet
OPEN_INTRO = (
    "You are helping researchers annotate the results of topic modeling on pump.fun, a Solana launchpad "
    "where anyone can create a memecoin by giving it a name, an optional description and an image. "
    "BERTopic grouped the tokens into topics. You see one topic. Its {k} samples are not its most typical "
    "members: the topic was split into {k} regions of its embedding space and each sample is the member "
    "closest to the centre of one region, shown with the share of the topic in that region. Every sample is "
    "a pump.fun memecoin, so being a memecoin, on Solana or on pump.fun is never a theme or a link; the "
    "collage layout and its numbers were added by the researchers."
)
OPEN_INPUT = {
    "text": "The topic was built from the token texts (name, then description). Topic keywords: {words}\n"
            "Samples (share of the topic in brackets):\n{samples}",
    "image": "The topic was built from the token images only, so judge the images only. The collage shows the "
             "{k} samples, numbered in their top-left corner. Share of the topic in each sample's region: {shares}",
    "multimodal": "The topic was built from the token image and text together. The collage shows the {k} sample "
                  "images, numbered in their top-left corner; the texts below have the same numbers. "
                  "Topic keywords: {words}\nSamples (share of the topic in brackets):\n{samples}",
}
OPEN_QUESTIONS = """Answer with a JSON object only, with these keys, in this order:
- "samples": a list of {k} short descriptions, one per sample, of what it shows or is about
- "theme": the main theme, i.e. the one shared by the most samples even if some samples do not fit; list the themes if there are several, or say there is none if no theme is shared
- "name": a short topic name in English, at most 5 words
- "definition": one sentence defining the topic
- "fit": how many of the {k} samples fit the main theme (integer 0-{k}; 0 if there is no theme)
- "other_link": what else links the samples besides the theme (e.g. the form of the name, a visual style, a template, spam), or "" if nothing
{images}- "notes": what makes this topic hard to annotate, and which category you would create for it"""
OPEN_IMAGES = '- "images": what the images are like (type and style: photo, meme, drawing, screenshot, logo, text, AI-generated...)\n'


def coverage_sample(X, labels, k=CARD_SAMPLES, max_fit=MAX_FIT, seed=SEED):
    """k-means medoids of each topic: DataFrame (topic, row, share, rank), ``row`` indexing ``X``,
    ``share`` the fraction of the topic in that medoid's k-means cluster, ranked by share."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    out = []
    for topic, rows in pd.Series(np.arange(len(labels))).groupby(labels).indices.items():
        if topic == -1:
            continue
        fit = rows if len(rows) <= max_fit else np.sort(rng.choice(rows, max_fit, replace=False))
        Z = np.asarray(X[fit], dtype=np.float32)
        km = KMeans(n_clusters=min(k, len(fit)), n_init=3, random_state=seed).fit(Z)
        dist = km.transform(Z)
        share = np.bincount(km.labels_, minlength=km.n_clusters) / len(fit)
        for rank, c in enumerate(np.argsort(-share, kind="stable")):
            members = np.flatnonzero(km.labels_ == c)
            out.append((int(topic), int(fit[members[dist[members, c].argmin()]]), float(share[c]), rank))
    return pd.DataFrame(out, columns=["topic", "row", "share", "rank"])


def _rgb(img):
    img.seek(0)  # first frame of animated images
    img = img.convert("RGBA")
    canvas = Image.new("RGB", img.size, "white")
    canvas.paste(img, mask=img.getchannel("A"))
    return canvas


def collage(images, ncols=COLLAGE_COLS, tile=TILE):
    """Grid of ``images`` (None leaves a blank tile), each tile numbered from 1 in its corner."""
    nrows = -(-len(images) // ncols)
    canvas = Image.new("RGB", (ncols * tile, nrows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=tile // 9)
    for i, img in enumerate(images):
        x0, y0 = (i % ncols) * tile, (i // ncols) * tile
        if img is not None:
            img = _rgb(img)
            img.thumbnail((tile, tile))
            canvas.paste(img, (x0 + (tile - img.width) // 2, y0 + (tile - img.height) // 2))
        draw.rectangle([x0, y0, x0 + tile // 6, y0 + tile // 8], fill="black")
        draw.text((x0 + 4, y0 + 2), str(i + 1), fill="white", font=font)
        draw.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], outline="#bbbbbb")
    return canvas


def build_cards(out_dir, modality, clip_dir, k=CARD_SAMPLES):
    # other sample counts (sensitivity runs) go to their own folder, <modality>_k<k>
    out = Path(out_dir) / "annotation" / (modality if k == CARD_SAMPLES else f"{modality}_k{k}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    keys, docs, X = load_modality(out_dir, modality, clip_dir)
    labels = pd.read_parquet(Path(out_dir) / modality / "full_topics.parquet").set_index("key").topic
    labels = labels.reindex(keys).fillna(-1).astype(np.int32).to_numpy()
    cards = coverage_sample(X, labels, k)
    print(f"{modality}: {k} k-means medoids for {cards.topic.nunique()} topics "
          f"({(time.time() - start) / 60:.1f} min)", flush=True)
    cards["key"] = keys[cards.row]
    cards["doc"] = [docs[i][:300] for i in cards.row]
    info = pd.read_parquet(Path(out_dir) / modality / "full_info.parquet")
    cards = info[["topic", "count", "words"]].merge(cards.drop(columns="row"), on="topic")
    cards = cards.sort_values(["topic", "rank"]).reset_index(drop=True)
    cards.to_parquet(out / "cards.parquet", index=False)
    if modality != "text":
        sha = cards.key.str.split(":").str[0]
        images = load_images(connect(DATA_DIR, threads=8), sha, DATA_DIR / "images")
        (out / "collages").mkdir(exist_ok=True)
        for topic, grp in sha.groupby(cards.topic):
            collage([images.get(s) for s in grp]).save(out / "collages" / f"{topic}.jpg", quality=90)
        missing = sha.map(lambda s: s not in images).sum()
        print(f"{modality}: {cards.topic.nunique()} collages ({missing} images missing)", flush=True)
    print(f"{modality}: cards written to {out} ({(time.time() - start) / 60:.1f} min)", flush=True)
    return cards


def stratified_topics(cards, n, exclude=(), bins=5, seed=SEED):
    """``n`` topics drawn evenly across ``bins`` size quantiles, leaving out ``exclude``."""
    sizes = cards.drop_duplicates("topic").set_index("topic")["count"].drop(list(exclude), errors="ignore")
    strata = pd.qcut(sizes.rank(method="first"), bins, labels=False)
    per = np.diff(np.linspace(0, n, bins + 1).round().astype(int))
    picked = [sizes[strata == b].sample(k, random_state=seed).index for b, k in enumerate(per)]
    return sorted(int(t) for idx in picked for t in idx)


def _jpeg_base64(path, width=768):
    import base64
    import io

    img = Image.open(path)
    img.thumbnail((width, width))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def build_pilot(out_dir, n=PILOT_TOPICS):
    """Open-coding pilot: ``n`` size-stratified topics per modality written to
    ``annotation/pilot_topics.parquet`` (to keep them out of the evaluation sample) and a
    self-contained annotation page ``annotation/open_coding.html`` (images embedded, answers kept in
    the browser and exported as CSV)."""
    root = Path(out_dir) / "annotation"
    data, picked = [], []
    for modality in MODALITIES:
        cards = pd.read_parquet(root / modality / "cards.parquet")
        topics = stratified_topics(cards, n)
        picked += [{"modality": modality, "topic": t} for t in topics]
        for topic in topics:
            grp = cards[cards.topic == topic].sort_values("rank")
            collage_path = root / modality / "collages" / f"{topic}.jpg"
            data.append({"modality": modality, "topic": topic, "count": int(grp["count"].iloc[0]),
                         "words": grp.words.iloc[0],
                         "samples": [{"doc": d, "share": round(float(s), 4)} for d, s in zip(grp.doc, grp.share)],
                         "collage": _jpeg_base64(collage_path) if modality != "text" else None})
    pd.DataFrame(picked).to_parquet(root / "pilot_topics.parquet", index=False)
    template = (Path(__file__).parent / "templates" / "open_coding.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = root / "open_coding.html"
    page.write_text(template.replace("/*DATA*/null", payload))
    print(f"pilot: {len(data)} cards, {page} ({page.stat().st_size / 1e6:.1f} MB)", flush=True)


def open_prompt(modality, words, docs, shares):
    k = len(docs)
    samples = "\n".join(f"{i + 1}. [{s:.0%}] {d}" for i, (d, s) in enumerate(zip(docs, shares)))
    body = OPEN_INPUT[modality].format(k=k, words=words, samples=samples,
                                       shares=", ".join(f"{i + 1}: {s:.0%}" for i, s in enumerate(shares)))
    questions = OPEN_QUESTIONS.format(k=k, images=OPEN_IMAGES if modality != "text" else "")
    return f"{OPEN_INTRO.format(k=k)}\n\n{body}\n\n{questions}"


def parse_json(text):
    """The first JSON object in ``text``, values flattened to strings (lists joined by "; ") and
    ``fit`` to an int; ``{}`` when there is none or it does not parse."""
    start, end = text.find("{"), text.rfind("}")
    try:
        answer = json.loads(text[start:end + 1]) if start >= 0 < end else {}
    except json.JSONDecodeError:
        return {}
    flat = {k: "; ".join(map(str, v)) if isinstance(v, list) else str(v) for k, v in answer.items()}
    fit = re.search(r"\d+", flat.get("fit", ""))
    flat["fit"] = int(fit.group()) if fit else None
    return flat


def load_llm(path=LLM, quantized=False):
    """The LLM in bf16 on the GPU, or 4-bit (bitsandbytes NF4) when ``quantized``, for a shared GPU with
    little free memory; labelling runs use bf16, only the image safety screen may use 4-bit."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "left"
    if quantized:
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(path, quantization_config=config, device_map={"": 0})
        return model.eval(), processor
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    return model, processor


def generate(model, processor, conversations, max_new_tokens=600, thinking=False):
    """Greedy answers to a batch of chat conversations, without the model's reasoning unless ``thinking``."""
    import torch

    inputs = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", padding=True,
                                           enable_thinking=thinking).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def annotate_open(out_dir, modalities=MODALITIES, pilot_only=False, batch=1, limit=None):
    """Open-coding answers of the LLM for every topic (or the pilot topics), one parquet per modality
    in ``annotation/llm/`` (kept apart from the annotators' page so they annotate blind)."""
    root = Path(out_dir) / "annotation"
    (root / "llm").mkdir(exist_ok=True)
    model, processor = load_llm()
    pilot = pd.read_parquet(root / "pilot_topics.parquet")
    for modality in modalities:
        cards = pd.read_parquet(root / modality / "cards.parquet").sort_values(["topic", "rank"])
        topics = pilot[pilot.modality == modality].topic.tolist() if pilot_only else sorted(cards.topic.unique())
        topics = topics[:limit] if limit else topics
        rows, start = [], time.time()
        for s in range(0, len(topics), batch):
            conversations = []
            for topic in topics[s:s + batch]:
                grp = cards[cards.topic == topic]
                docs = grp.doc.tolist() if modality != "image" else [""] * len(grp)
                content = [{"type": "text", "text": open_prompt(modality, grp.words.iloc[0], docs, grp.share)}]
                if modality != "text":
                    content.insert(0, {"type": "image", "image": Image.open(root / modality / "collages" / f"{topic}.jpg")})
                conversations.append([{"role": "user", "content": content}])
            for topic, raw in zip(topics[s:s + batch], generate(model, processor, conversations)):
                rows.append({"topic": int(topic), "raw": raw, **parse_json(raw)})
            print(f"{modality}: {len(rows)}/{len(topics)} topics ({(time.time() - start) / 60:.1f} min)", flush=True)
        df = pd.DataFrame(rows)
        name = f"open_{modality}{'_pilot' if pilot_only else ''}.parquet"
        df.to_parquet(root / "llm" / name, index=False)
        failed = df["name"].isna().sum() if "name" in df else len(df)
        print(f"{modality}: {name} written ({failed} unparsed answers)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["cards", "pilot", "open"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=Path("data/clip"))
    parser.add_argument("--modality", choices=MODALITIES, default="text")
    parser.add_argument("--samples", type=int, default=CARD_SAMPLES)
    parser.add_argument("--topics", type=int, default=PILOT_TOPICS, help="pilot: topics per modality")
    parser.add_argument("--all-modalities", action="store_true", help="open: every modality")
    parser.add_argument("--pilot-only", action="store_true", help="open: only the pilot topics")
    parser.add_argument("--limit", type=int, default=None, help="open: first topics only (smoke test)")
    parser.add_argument("--batch", type=int, default=1, help="open: >1 is faster but padded batches change image answers")
    args = parser.parse_args()
    if args.step == "cards":
        build_cards(args.out_dir, args.modality, args.clip_dir, args.samples)
    elif args.step == "pilot":
        build_pilot(args.out_dir, args.topics)
    else:
        annotate_open(args.out_dir, MODALITIES if args.all_modalities else [args.modality], args.pilot_only,
                      args.batch, args.limit)


if __name__ == "__main__":
    main()
