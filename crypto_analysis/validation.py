"""Human validation decks for the topic-level pipeline: same-narrative pairs and content attributes.

Two self-contained annotation pages, each filled independently by every annotator and exported as CSV:

* ``pairs``: ``PAIR_SAMPLE`` candidate topic pairs, each shown as two topic cards (the LLM-written topic
  name, c-TF-IDF terms and the ``CARD_SAMPLES`` coverage samples with image and text). Question: do the two
  topics belong to the same narrative? The candidates are the union of each topic's ``NEIGHBOURS`` nearest
  topics in the multimodal space (the topic model's fused CLIP centroid) and in the text space (MiniLM
  centroid of the topic's texts), as the LLM judge will see them. The sample is stratified so that rare
  same-narrative pairs are well represented: pairs whose topic names share a content word, pairs of close
  neighbours (rank ``CLOSE`` or better in either space) and the remaining candidates; each pair keeps its
  inclusion probability to weight estimates back to all candidates.
* ``names``: ``NAME_SAMPLE`` topics drawn uniformly, each with the name and summary the LLM wrote for it
  (``topic_taxonomy leaves``). Questions: does the name describe the topic (yes, partly, no; a better name
  if not yes), and is the topic coherent (most samples share a subject)? Most narratives are single topics
  and keep their topic's name, so this also validates most narrative names.
* ``attributes``: ``TOPIC_SAMPLE`` topics, half drawn from topics with a sign of harmful content (samples the
  LLM marked in the leaf descriptions, or terms matching ``SENSITIVE``) and half from the others. Every
  coverage sample is labelled with the content attributes (hate with racist, antisemitic or other target;
  sexual; violence; controlled substances), "none" or "other" (named). A topic's share of an attribute is
  the region-weighted share of its samples that show it.

Before any image goes into a page, Qwen3.5-9B screens it for sexual content involving minors; flagged images
are withheld (the coin text stays). Definitions shown are those of ``coin_labels`` (sources in
``reports/references/definitions.md``). Outputs in ``<out>/validation/``. Usage::

    python -m crypto_analysis.validation sample --out-dir data/topics
    python -m crypto_analysis.validation screen --out-dir data/topics
    python -m crypto_analysis.validation pages  --out-dir data/topics
    python -m crypto_analysis.validation names-sample --out-dir data/topics
    python -m crypto_analysis.validation pages  --out-dir data/topics --deck names
"""

import argparse
import base64
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from crypto_analysis.topic_search import SEED

NEIGHBOURS = 20
CLOSE = 3
PAIR_SAMPLE = 150
TOPIC_SAMPLE = 100
NAME_SAMPLE = 100
TEST_SAMPLE = 50
TEST_TOPIC_CAP = 2  # at most this many test pairs per topic, so the many near-identical generic topics do not fill it
DECKS = ["pairs", "attributes", "names", "macro_pairs", "adjudication", "macro_test", "macro_names"]
CARD_SAMPLES = 12
GENERIC = {"meme", "memes", "memecoin", "memecoins", "coin", "coins", "token", "tokens", "crypto", "themed",
           "theme", "culture", "internet", "characters", "character", "and", "the", "of", "with", "related",
           "based", "random", "various", "popular", "humor", "humorous", "names", "name", "collection"}
SENSITIVE = re.compile(r"nazi|hitler|swastika|holocaust|jew|antisemit|israel|racis|nigg|slur|hate|kkk|porn|sex|nud|"
                       r"nsfw|boob|tits|penis|genital|onlyfans|drug|weed|cannabis|cocaine|meth|fentanyl|pill|"
                       r"alcohol|beer|cigar|kill|murder|shoot|gun|weapon|war\b|terror|violen|blood|gore", re.IGNORECASE)
TEMPLATES = Path(__file__).parent / "templates"


def validation_dir(out_dir):
    return Path(out_dir) / "validation"


# --------------------------------------------------------------------------------------------
# Topics and candidate pairs
# --------------------------------------------------------------------------------------------

def topic_frame(out_dir):
    """Every multimodal topic with its LLM-written name and summary, c-TF-IDF terms and harm signal."""
    root = Path(out_dir)
    info = pd.read_parquet(root / "multimodal" / "full_info.parquet").query("topic >= 0").set_index("topic")
    leaves = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic")
    topics = info[["count", "words"]].join(leaves[["theme", "summary"]])
    marked = pd.Series(False, index=leaves.index)
    for col in [c for c in leaves.columns if c.endswith("_share")]:
        marked |= leaves[col].fillna(0) > 0
    text = topics.theme.fillna("") + " " + topics.summary.fillna("") + " " + topics.words.fillna("")
    topics["sensitive"] = marked.reindex(topics.index, fill_value=False) | text.str.contains(SENSITIVE)
    return topics.sort_index()


def text_centroids(out_dir):
    """L2-normalised MiniLM centroid of the texts of every multimodal topic (cached)."""
    root = Path(out_dir)
    path = validation_dir(root) / "text_centroids.npy"
    if path.exists():
        return np.load(path)
    topics = pd.read_parquet(root / "multimodal" / "full_topics.parquet")
    topics = topics[topics.topic >= 0]
    text_key = topics.key.str.split(":").str[1]
    rows = pd.Index(pd.read_parquet(root / "docs_text.parquet", columns=["text_key"]).text_key).get_indexer(text_key)
    emb = np.load(root / "text_minilm.npy", mmap_mode="r")
    labels, keep = topics.topic.to_numpy(), rows >= 0
    C = np.zeros((int(labels.max()) + 1, emb.shape[1]), dtype=np.float64)
    for s in range(0, len(rows), 1_000_000):
        r, lab, k = rows[s:s + 1_000_000], labels[s:s + 1_000_000], keep[s:s + 1_000_000]
        np.add.at(C, lab[k], np.asarray(emb[r[k]], dtype=np.float32))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, C)
    return C


def neighbour_ranks(C, k):
    """{(a, b): (rank, similarity)} for a < b, the best rank of either topic among the other's ``k`` nearest."""
    S = C @ C.T
    np.fill_diagonal(S, -np.inf)
    out = {}
    for a in range(len(C)):
        for rank, b in enumerate(np.argsort(-S[a])[:k], 1):
            key = (min(a, int(b)), max(a, int(b)))
            if key not in out or rank < out[key][0]:
                out[key] = (rank, float(S[a, b]))
    return out


def candidate_pairs(out_dir, k=NEIGHBOURS):
    """Union of the ``k``-nearest-topic pairs in the multimodal and the text space, with the rank and
    similarity of each pair in each space (rank 0: not a neighbour there)."""
    from crypto_analysis.narratives import narrative_dir

    mm = neighbour_ranks(np.load(narrative_dir(out_dir, "multimodal", "published") / "centroids.npy"), k)
    tx = neighbour_ranks(text_centroids(out_dir), k)
    rows = [{"a": a, "b": b, "rank_mm": mm.get((a, b), (0, np.nan))[0], "sim_mm": mm.get((a, b), (0, np.nan))[1],
             "rank_text": tx.get((a, b), (0, np.nan))[0], "sim_text": tx.get((a, b), (0, np.nan))[1]}
            for a, b in sorted(set(mm) | set(tx))]
    return pd.DataFrame(rows)


def name_words(name):
    return {w for w in re.findall(r"[a-z][a-z0-9']{2,}", str(name).lower()) if w not in GENERIC}


def pair_strata(cands, topics):
    """``cands`` with its stratum: names share a content word; close neighbours; the rest."""
    words = {t: name_words(name) for t, name in topics.theme.items()}
    shared = np.array([bool(words.get(a, set()) & words.get(b, set())) for a, b in zip(cands.a, cands.b)])
    best = np.where(cands.rank_mm > 0, cands.rank_mm, 99).clip(max=99)
    best = np.minimum(best, np.where(cands.rank_text > 0, cands.rank_text, 99))
    stratum = np.where(shared, "shared name word", np.where(best <= CLOSE, "close neighbours", "other candidates"))
    return cands.assign(stratum=stratum)


def sample_pairs(cands, topics, n=PAIR_SAMPLE, seed=SEED):
    """``n`` candidate pairs in three equal strata (``pair_strata``), each with its inclusion probability."""
    rng = np.random.default_rng(seed)
    cands = pair_strata(cands, topics)
    picked = []
    for name, grp in cands.groupby("stratum"):
        take = min(len(grp), n // 3)
        rows = grp.iloc[rng.choice(len(grp), take, replace=False)].assign(inclusion=take / len(grp))
        picked.append(rows)
    sample = pd.concat(picked)
    short = n - len(sample)
    if short > 0:  # a small stratum leaves room: fill from the remaining candidates
        rest = cands.drop(sample.index)
        extra = rest.iloc[rng.choice(len(rest), short, replace=False)]
        sample = pd.concat([sample, extra.assign(inclusion=np.nan)])
        for name, grp in sample.groupby("stratum"):  # inclusion probability within the final stratum sizes
            sample.loc[grp.index, "inclusion"] = len(grp) / int((cands.stratum == name).sum())
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "pair_id", [f"P{i + 1:03d}" for i in range(len(sample))])
    return sample


def gold_pairs(out_dir):
    """The development reference (``pairs_gold.parquet``): the macro-topic pairs both annotators labelled alike,
    plus the joint decision on those they did not (``adjudicated_pairs.csv``)."""
    out = validation_dir(out_dir)
    labels = pd.concat([pd.read_csv(f) for f in sorted(out.glob("macro_pairs_*.csv"))])
    adjudicated = pd.read_csv(out / "adjudicated_pairs.csv").set_index("pair_id")
    votes = labels.groupby("pair_id").same.agg(["nunique", "first"])
    split = set(votes.index[votes["nunique"] > 1])
    assert split == set(adjudicated.index), split ^ set(adjudicated.index)
    gold = pd.read_parquet(out / "pairs_sample.parquet")[["pair_id", "a", "b", "stratum", "inclusion"]]
    gold["source"] = np.where(gold.pair_id.isin(split), "adjudicated", "agreed")
    gold["same"] = gold.pair_id.map(adjudicated.same.combine_first(votes["first"])).eq("yes")
    subject = labels.dropna(subset=["narrative"]).groupby("pair_id").narrative.first()
    gold["subject"] = gold.pair_id.map(adjudicated.subject.combine_first(subject)).where(gold.same)
    gold.to_parquet(out / "pairs_gold.parquet", index=False)
    print(f"{len(gold)} pairs, {int(gold.same.sum())} same; " + gold.source.value_counts().to_string().replace("\n", ", "))
    return gold


def sample_test_pairs(out_dir, n=TEST_SAMPLE, seed=SEED + 1):
    """``n`` held-out candidate pairs in the strata of ``sample_pairs`` (``macro_test_sample.parquet``): none is a
    development pair, none touches a topic of the few-shot examples, and no topic appears more than
    ``TEST_TOPIC_CAP`` times. Its ``inclusion`` is the stratum's sampling fraction, ignoring that cap."""
    from crypto_analysis.narrative_overlap import FEW_SHOT

    out = validation_dir(out_dir)
    dev = pd.read_parquet(out / "pairs_sample.parquet")
    shown = dev[dev.pair_id.isin([e[0] for e in FEW_SHOT])]
    held = set(shown.a) | set(shown.b)
    cands = pair_strata(pd.read_parquet(out / "candidates.parquet"), topic_frame(out_dir))
    cands = cands.merge(dev[["a", "b"]].assign(dev=True), on=["a", "b"], how="left")
    cands = cands[cands.dev.isna() & ~cands.a.isin(held) & ~cands.b.isin(held)].drop(columns="dev")
    rng, used, picked = np.random.default_rng(seed), {}, []
    for i, (name, grp) in enumerate(cands.groupby("stratum")):
        quota, taken = n // 3 + (i < n % 3), []
        for row in rng.permutation(len(grp)):
            r = grp.iloc[row]
            if len(taken) < quota and used.get(r.a, 0) < TEST_TOPIC_CAP and used.get(r.b, 0) < TEST_TOPIC_CAP:
                used[r.a], used[r.b] = used.get(r.a, 0) + 1, used.get(r.b, 0) + 1
                taken.append(r)
        picked.append(pd.DataFrame(taken).assign(inclusion=quota / len(grp)))
    sample = pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "pair_id", [f"T{i + 1:03d}" for i in range(len(sample))])
    sample.to_parquet(out / "macro_test_sample.parquet", index=False)
    print(f"{len(sample)} test pairs, {len(set(sample.a) | set(sample.b))} topics, "
          f"{len((set(sample.a) | set(sample.b)) - set(dev.a) - set(dev.b))} topics in no development pair\n"
          + sample.stratum.value_counts().to_string(), flush=True)
    return sample


def sample_topics(topics, n=TOPIC_SAMPLE, seed=SEED):
    """``n`` topics, half among those with a harm signal and half among the others, with inclusion
    probabilities."""
    rng = np.random.default_rng(seed)
    picked = []
    for flag, take in [(True, n // 2), (False, n - n // 2)]:
        pool = topics[topics.sensitive == flag]
        take = min(take, len(pool))
        picked.append(pool.iloc[rng.choice(len(pool), take, replace=False)].assign(
            stratum="harm signal" if flag else "no harm signal", inclusion=take / len(pool)))
    sample = pd.concat(picked).sample(frac=1, random_state=seed).rename_axis("topic").reset_index()
    sample.insert(0, "item_id", [f"T{i + 1:03d}" for i in range(len(sample))])
    return sample


def sample_names(out_dir, n=NAME_SAMPLE, seed=SEED):
    """``n`` topics drawn uniformly for the topic-name deck (``names_sample.parquet``, kept once written)."""
    path = validation_dir(out_dir) / "names_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    topics = topic_frame(out_dir)
    rng = np.random.default_rng(seed)
    sample = topics.iloc[rng.choice(len(topics), n, replace=False)].rename_axis("topic").reset_index()
    sample.insert(0, "item_id", [f"N{i + 1:03d}" for i in range(len(sample))])
    sample["inclusion"] = n / len(topics)
    sample.to_parquet(path, index=False)
    print(f"{len(sample)} topics for the name deck", flush=True)
    return sample


def build_samples(out_dir):
    out = validation_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    topics = topic_frame(out_dir)
    cands = candidate_pairs(out_dir)
    cands.to_parquet(out / "candidates.parquet", index=False)
    pairs = sample_pairs(cands, topics)
    pairs.to_parquet(out / "pairs_sample.parquet", index=False)
    attrs = sample_topics(topics)
    attrs.to_parquet(out / "attributes_sample.parquet", index=False)
    print(f"{len(cands):,} candidate pairs (multimodal {(cands.rank_mm > 0).sum():,}, text {(cands.rank_text > 0).sum():,},"
          f" both {((cands.rank_mm > 0) & (cands.rank_text > 0)).sum():,})")
    print(pairs.stratum.value_counts().to_string())
    print(attrs.stratum.value_counts().to_string(), flush=True)


# --------------------------------------------------------------------------------------------
# Image screen and pages
# --------------------------------------------------------------------------------------------

def macro_name_items(out_dir):
    """Members of every macro-topic the LLM named, largest topic first, each with the key of its first coverage
    sample (the medoid of its largest region); None before the macro-topics exist."""
    from crypto_analysis.narrative_overlap import MACRO_TOPICS, overlap_dir

    path = overlap_dir(out_dir) / MACRO_TOPICS
    if not path.exists():
        return None
    macro = pd.read_parquet(path)
    macro = macro[macro.named_by == "llm"].explode("topics").rename(columns={"topics": "topic"})
    cards = pd.read_parquet(Path(out_dir) / "annotation" / "multimodal" / "cards.parquet")
    first = cards.sort_values("rank").drop_duplicates("topic").set_index("topic")
    macro["key"] = macro.topic.map(first.key)
    macro["count"] = macro.topic.map(first["count"])
    return macro.sort_values(["macro_topic", "count"], ascending=[True, False])


def deck_cards(out_dir):
    """Coverage samples of every topic in any deck: topic, rank, share, sha256, doc."""
    out = validation_dir(out_dir)
    pairs = pd.read_parquet(out / "pairs_sample.parquet")
    attrs = pd.read_parquet(out / "attributes_sample.parquet")
    wanted = set(pairs.a) | set(pairs.b) | set(attrs.topic)
    if (out / "names_sample.parquet").exists():
        wanted |= set(pd.read_parquet(out / "names_sample.parquet").topic)
    if (out / "macro_test_sample.parquet").exists():
        test = pd.read_parquet(out / "macro_test_sample.parquet")
        wanted |= set(test.a) | set(test.b)
    cards = pd.read_parquet(Path(out_dir) / "annotation" / "multimodal" / "cards.parquet")
    macro = macro_name_items(out_dir)
    if macro is not None:  # the macro-topic name deck shows one sample of every member topic
        cards = cards[cards.topic.isin(wanted) | cards.key.isin(set(macro.key))]
        return cards.sort_values(["topic", "rank"]).assign(sha256=cards.key.str.split(":").str[0])
    cards = cards[cards.topic.isin(wanted)].sort_values(["topic", "rank"])
    return cards.assign(sha256=cards.key.str.split(":").str[0])


SCREEN_PROMPT = ("You check images before researchers show them to human annotators. Does this image show sexual "
                 "content involving a minor (anyone who appears to be under 18), including sexualised drawings? "
                 'Answer with a JSON object only: {"minor_sexual": true or false}')


def read_flag(raw, answer):
    """The screen's answer as a bool: the JSON field, also when written as a string or as a bare true/false;
    anything else counts as flagged, so the image is withheld."""
    flag = answer.get("minor_sexual", raw.strip().strip("`").strip())
    flag = str(flag).strip().lower() if not isinstance(flag, bool) else flag
    return flag if isinstance(flag, bool) else {"true": True, "false": False}.get(flag, True)


def screen(out_dir, quantized=False):
    """Qwen3.5-9B check of every deck image for sexual content involving minors (``screen.parquet``);
    ``quantized`` loads it in 4-bit when the shared GPU lacks memory for bf16."""
    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import _rgb
    from crypto_analysis.topic_taxonomy import Asker

    out = validation_dir(out_dir)
    shas = deck_cards(out_dir).sha256.unique().tolist()
    done = pd.read_parquet(out / "screen.parquet") if (out / "screen.parquet").exists() else None
    todo = [s for s in shas if done is None or s not in set(done.sha256)]
    images = load_images(connect(DATA_DIR, threads=8), todo, DATA_DIR / "images")
    ask, rows = Asker(quantized), []
    for i, sha in enumerate(todo, 1):
        img = images.get(sha)
        if img is None:
            rows.append({"sha256": sha, "minor_sexual": None, "raw": "missing"})
            continue
        img = _rgb(img)
        img.thumbnail((512, 512))
        raw, answer = ask(SCREEN_PROMPT, img, max_new_tokens=20)
        rows.append({"sha256": sha, "minor_sexual": read_flag(raw, answer), "raw": raw})
        if i % 250 == 0 or i == len(todo):  # saved as it goes, so a restart resumes where it stopped
            result = pd.concat([done, pd.DataFrame(rows)]) if done is not None else pd.DataFrame(rows)
            result.to_parquet(out / "screen.parquet", index=False)
            print(f"  screened {i}/{len(todo)}", flush=True)
    result = pd.read_parquet(out / "screen.parquet")
    print(f"{len(result):,} images screened, {int(result.minor_sexual.fillna(False).sum())} withheld "
          "(unparsed answers count as withheld)", flush=True)


def thumbs(shas, size):
    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import _rgb

    out = {}
    for sha, img in load_images(connect(DATA_DIR, threads=8), shas, DATA_DIR / "images").items():
        img = _rgb(img)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        out[sha] = base64.b64encode(buf.getvalue()).decode()
    return out


def definitions():
    """The definitions shown to annotators, with their sources (as given to the LLM)."""
    from crypto_analysis.coin_labels import ANTISEMITISM, ATTRIBUTES, NARRATIVE

    return {
        "narrative": NARRATIVE.replace("collection of coins", "collection of tokens"),
        "narrative_reference": "“collections of documents that seek to address the same event or issue” — Hanley, "
                               "Kumar & Durumeric, IEEE S&P 2024, §3.1 (also Gerard et al., ICWSM 2026). The "
                               "parenthesis is our context for pump.fun, not part of the definition.",
        "attributes": {a: ATTRIBUTES[a] for a in ("hate", "sexual", "violence", "controlled_substances")},
        "racist": "hate whose target is race, ethnicity, nationality or immigration status (targets from Kiela et "
                  "al.'s list of protected characteristics, recorded as in ElSherief et al., ICWSM 2018)",
        "antisemitic": "hate targeting Jews: “Antisemitism is a certain perception of Jews, which may be expressed "
                       "as hatred toward Jews…” (IHRA working definition as quoted by Chandra et al., WebSci 2021); "
                       "types: " + "; ".join(f"{k}: {v}" for k, v in ANTISEMITISM.items()),
        "rules": "Mark an attribute only when its definition is met. Tone, intent and whether it is a joke do not "
                 "matter. An object associated with an attribute (a weapon, a pill, a flag) is not enough by itself: "
                 "what the sample shows or says must meet the definition. Do not guess what an unfamiliar symbol "
                 "means; read a hidden meaning only when it is a known code (e.g. 1488). Being a memecoin is never "
                 "an attribute by itself.",
        "sources": {
            "hate": "Kiela et al., NeurIPS 2020 (§2.1); images: Qu et al., ACM CCS 2025 (Table 9, “Hate”)",
            "sexual": "Markov et al., AAAI 2023 (§2, category S); images: Qu et al., ACM CCS 2025 (Table 9)",
            "violence": "Markov et al., AAAI 2023 (§2, category V); images: Qu et al., ACM CCS 2025 (Table 9)",
            "controlled_substances": "Ghosh et al. (Aegis 2.0), NAACL 2025 (Appendix A.13.3, S5)",
        },
    }


def coin_texts(out_dir, text_keys):
    """{text_key: {coin, symbol, description}}: the name, symbol and full description of a coin with that
    text, cleaned as the models saw them (their text is "name. description")."""
    from crypto_analysis.clip_embeddings import clean_text
    from crypto_analysis.datalake import DATA_DIR, connect

    tok = pd.read_parquet(Path(out_dir) / "docs_token.parquet", columns=["mint", "text_key"])
    tok = tok[tok.text_key.isin(set(text_keys))].drop_duplicates("text_key")
    con = connect(DATA_DIR, threads=8)
    con.register("wanted", tok)
    rows = con.execute("""
        SELECT w.text_key, any_value(m.meta_name) AS name, any_value(m.meta_symbol) AS symbol,
               any_value(m.description) AS description
        FROM wanted w JOIN token_metadata m USING (mint) GROUP BY 1
    """).df()

    def text(v):
        return clean_text(str(v)) if isinstance(v, str) else ""

    return {k: {"coin": text(n), "symbol": text(sym), "description": text(d)}
            for k, n, sym, d in zip(rows.text_key, rows.name, rows.symbol, rows.description)}


def page(template, data, path):
    html = (TEMPLATES / template).read_text()
    path.write_text(html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def build_pages(out_dir, decks=DECKS):
    out = validation_dir(out_dir)
    topics = topic_frame(out_dir)
    cards = deck_cards(out_dir)
    withheld = set(pd.read_parquet(out / "screen.parquet").query("minor_sexual == True").sha256)
    pilot = Path(out_dir) / "coin_labels"
    for path in pilot.glob("*/labels_*.parquet"):  # images the coin-level runs flagged stay withheld
        labels = pd.read_parquet(path)
        sample = pd.read_parquet(path.parent / "sample.parquet")
        withheld |= set(sample.sha256[sample.mint.isin(labels.mint[labels.sexual_minors])].dropna())

    meta = coin_texts(out_dir, cards.key.str.split(":").str[1])

    def topic_card(t, with_share):
        grp = cards[cards.topic == t]
        fallback = {"coin": "", "symbol": "", "description": ""}
        return {"topic": int(t), "name": str(topics.theme.get(t) or "(no name)"),
                "summary": str(topics.summary.get(t) or ""),
                "terms": " ".join(str(topics.words.get(t, "")).split()[:10]),
                "samples": [{"rank": int(r.rank), "sha": None if r.sha256 in withheld else r.sha256,
                             **meta.get(r.key.split(":")[1], dict(fallback, coin=str(r.doc)[:240])),
                             **({"share": round(float(r.share), 3)} if with_share else {})}
                            for r in grp.itertuples()]}

    defs = definitions()
    if "pairs" in decks:
        pairs = pd.read_parquet(out / "pairs_sample.parquet")
        items = [{"id": r.pair_id, "a": topic_card(r.a, False), "b": topic_card(r.b, False)} for r in pairs.itertuples()]
        shas = {s["sha"] for it in items for side in ("a", "b") for s in it[side]["samples"] if s["sha"]}
        page("pairs_annotation.html", {"deck": "pairs-v1", "items": items, "images": thumbs(shas, 150),
                                        "definitions": defs}, out / "pairs_annotation.html")
    if "macro_pairs" in decks:  # the same pairs, asked as a macro-topic classification
        from crypto_analysis.narrative_overlap import MACRO_CLASSES, MACRO_DEFINITION, MACRO_RULES

        pairs = pd.read_parquet(out / "pairs_sample.parquet")
        items = [{"id": r.pair_id, "a": topic_card(r.a, False), "b": topic_card(r.b, False)} for r in pairs.itertuples()]
        shas = {s["sha"] for it in items for side in ("a", "b") for s in it[side]["samples"] if s["sha"]}
        macro = {"definition": MACRO_DEFINITION, "classes": MACRO_CLASSES.split("\n"), "rules": MACRO_RULES,
                 "reference": "Macro-topics follow the topic groups of Steffen et al. (ICWSM 2025): topics grouped into "
                              "broader groups by their thematic similarity."}
        page("macro_pairs_annotation.html", {"deck": "macro-pairs-v1", "prefix": "macro_pairs", "items": items,
                                             "images": thumbs(shas, 150), "macro": macro}, out / "macro_pairs_annotation.html")
    if "macro_test" in decks:  # the held-out pairs, under the guideline fixed in the adjudication
        from crypto_analysis.narrative_overlap import (
            GUIDED_CLASSES,
            GUIDED_DEFINITION,
            GUIDED_RULES,
        )

        pairs = pd.read_parquet(out / "macro_test_sample.parquet")
        items = [{"id": r.pair_id, "a": topic_card(r.a, False), "b": topic_card(r.b, False)} for r in pairs.itertuples()]
        shas = {s["sha"] for it in items for side in ("a", "b") for s in it[side]["samples"] if s["sha"]}
        macro = {"definition": GUIDED_DEFINITION, "classes": GUIDED_CLASSES.split("\n"), "rules": GUIDED_RULES,
                 "reference": "Macro-topics follow the topic groups of Steffen et al. (ICWSM 2025): topics grouped into "
                              "broader groups by their thematic similarity."}
        page("macro_pairs_annotation.html", {"deck": "macro-test-v1", "prefix": "macro_test", "items": items,
                                             "images": thumbs(shas, 150), "macro": macro}, out / "macro_test_annotation.html")
    if "adjudication" in decks:  # the pairs the annotators of macro_pairs disagree on, with both answers
        from crypto_analysis.narrative_overlap import MACRO_CLASSES, MACRO_DEFINITION, MACRO_RULES

        labels = pd.concat([pd.read_csv(f) for f in sorted(out.glob("macro_pairs_*.csv"))])
        split = labels.groupby("pair_id").same.nunique().loc[lambda n: n > 1].index
        pairs = pd.read_parquet(out / "pairs_sample.parquet").set_index("pair_id").loc[sorted(split)]
        items = []
        for pair_id, r in pairs.iterrows():
            votes = [{"annotator": v.annotator, "same": v.same, "subject": v.narrative if isinstance(v.narrative, str) else "",
                      "comment": v.comment if isinstance(v.comment, str) else ""}
                     for v in labels[labels.pair_id == pair_id].itertuples()]
            items.append({"id": pair_id, "a": topic_card(r.a, False), "b": topic_card(r.b, False), "votes": votes})
        shas = {s["sha"] for it in items for side in ("a", "b") for s in it[side]["samples"] if s["sha"]}
        macro = {"definition": MACRO_DEFINITION, "classes": MACRO_CLASSES.split("\n"), "rules": MACRO_RULES,
                 "reference": "Macro-topics follow the topic groups of Steffen et al. (ICWSM 2025): topics grouped into "
                              "broader groups by their thematic similarity."}
        page("adjudication_annotation.html", {"deck": "adjudication-v1", "items": items, "images": thumbs(shas, 150),
                                              "macro": macro}, out / "adjudication_annotation.html")
        print(f"{len(items)} pairs to adjudicate")
    if "macro_names" in decks:  # the LLM-named macro-topics, one sample of every member topic
        from crypto_analysis.narrative_overlap import GUIDED_DEFINITION

        macro, items = macro_name_items(out_dir), []
        for i, (m, grp) in enumerate(sorted(macro.groupby("macro_topic"), key=lambda g: -len(g[1])), 1):
            samples = []
            for r in grp.itertuples():
                sha = r.key.split(":")[0]
                samples.append({"rank": 0, "sha": None if sha in withheld else sha, "topic_name": str(topics.theme.get(r.topic)),
                                **meta.get(r.key.split(":")[1], {"coin": "", "symbol": "", "description": ""})})
            items.append({"id": f"M{i:02d}", "topic": int(m), "name": grp.name.iloc[0], "summary": grp.description.iloc[0],
                          "terms": "; ".join(str(topics.theme.get(t)) for t in grp.topic), "samples": samples})
        shas = {s["sha"] for it in items for s in it["samples"] if s["sha"]}
        page("macro_names_annotation.html", {"deck": "macro-names-v1", "items": items, "images": thumbs(shas, 180),
                                             "definition": GUIDED_DEFINITION}, out / "macro_names_annotation.html")
    if "attributes" in decks:
        attrs = pd.read_parquet(out / "attributes_sample.parquet")
        items = [{"id": r.item_id, **topic_card(r.topic, True)} for r in attrs.itertuples()]
        shas = {s["sha"] for it in items for s in it["samples"] if s["sha"]}
        page("attributes_annotation.html", {"deck": "attributes-v1", "items": items, "images": thumbs(shas, 220),
                                             "definitions": defs}, out / "attributes_annotation.html")
    if "names" in decks:
        names = pd.read_parquet(out / "names_sample.parquet")
        items = [{"id": r.item_id, **topic_card(r.topic, False)} for r in names.itertuples()]
        shas = {s["sha"] for it in items for s in it["samples"] if s["sha"]}
        page("names_annotation.html", {"deck": "names-v1", "items": items, "images": thumbs(shas, 180)},
             out / "names_annotation.html")
    print(f"{len(withheld & set(cards.sha256))} deck images withheld")


# --------------------------------------------------------------------------------------------
# Agreement with the LLM
# --------------------------------------------------------------------------------------------

ATTRIBUTE_COLS = ["hate", "racist", "antisemitic", "sexual", "violence", "controlled_substances", "other", "none"]
# "other" has no published definition and the annotators' "other" is not reported; the LLM's stays in the results
REPORTED = [a for a in ATTRIBUTE_COLS if a != "other"]
HARMS = ["hate", "sexual", "violence", "controlled_substances"]


def without_other(frame):
    """``none`` recomputed as "none of the four attributes", so that "other" changes no reported figure."""
    return frame.assign(none=~frame[HARMS].fillna(0).astype(bool).any(axis=1))


def binary_agreement(human, llm, weights=None):
    """Cohen's kappa, accuracy, the LLM's precision and recall against the human labels, and both rates
    (weighted by ``weights`` when given, to undo a stratified sample)."""
    from sklearn.metrics import cohen_kappa_score

    h, m = np.asarray(human, dtype=bool), np.asarray(llm, dtype=bool)
    w = np.ones(len(h)) if weights is None else np.asarray(weights, dtype=float)
    tp, fp, fn = w[h & m].sum(), w[~h & m].sum(), w[h & ~m].sum()
    both_constant = len(set(h)) == 1 and len(set(m)) == 1
    return {"n": len(h), "human_rate": w[h].sum() / w.sum(), "llm_rate": w[m].sum() / w.sum(),
            "accuracy": w[h == m].sum() / w.sum(),
            "kappa": np.nan if both_constant else cohen_kappa_score(h, m, sample_weight=w),
            "llm_precision": tp / (tp + fp) if tp + fp else np.nan, "llm_recall": tp / (tp + fn) if tp + fn else np.nan}


def attribute_raters(out_dir, files):
    """Per attribute (``REPORTED``), over the samples of the attributes deck: Cohen's kappa between the first two
    annotators, Fleiss' kappa of all annotators with the LLM in each condition, and Cohen's kappa of the LLM
    against the samples on which every annotator agrees."""
    from sklearn.metrics import cohen_kappa_score

    from crypto_analysis.topic_attributes import CONDITIONS, attributes_dir

    labels = {}
    for f in files:
        h = without_other(pd.read_csv(f)).set_index(["topic", "sample_rank"]).sort_index()
        labels[str(h.annotator.iloc[0]).split()[0].capitalize()] = h
    names = list(labels)
    index = labels[names[0]].index
    llm = {}
    for c in CONDITIONS:
        m = pd.read_parquet(attributes_dir(out_dir) / f"samples_{c}.parquet").set_index(["topic", "rank"])
        m["racist"] &= m.hate
        m["antisemitic"] &= m.hate
        llm[c] = without_other(m).reindex(index)
    rows = []
    for a in REPORTED:
        H = np.column_stack([labels[n].loc[index, a].fillna(0).astype(bool) for n in names])
        agreed = H.min(axis=1) == H.max(axis=1)
        row = {"attribute": a, "n": len(H), **{f"{n} marked": int(H[:, i].sum()) for i, n in enumerate(names)},
               "all annotators marked": int(H.all(axis=1).sum()),
               f"cohen {names[0]} x {names[1]}": cohen_kappa_score(H[:, 0], H[:, 1])}
        for c in CONDITIONS:
            m = llm[c][a].astype(bool).to_numpy()
            row[f"LLM {c} marked"] = int(m.sum())
            row[f"fleiss annotators + LLM {c}"] = fleiss_kappa(np.column_stack([H, m]))
            row[f"cohen consensus x LLM {c}"] = cohen_kappa_score(H[agreed, 0], m[agreed])
        rows.append(row)
    return pd.DataFrame(rows)


def agreement(out_dir):
    """Each annotator's agreement with the LLM on the three decks (``agreement_<deck>.parquet``)."""
    from crypto_analysis.narrative_overlap import judged_pairs
    from crypto_analysis.topic_attributes import attributes_dir

    out = validation_dir(out_dir)
    results = {}
    files = {deck: sorted(out.glob(f"{deck}_*.csv")) for deck in DECKS}

    if files["pairs"]:
        sample = pd.read_parquet(out / "pairs_sample.parquet")
        llm = judged_pairs(out_dir, "judgments.parquet").dropna(subset=["ab", "ba"])[["a", "b", "same", "ab", "ba"]]
        rows = []
        for f in files["pairs"]:
            h = pd.read_csv(f).rename(columns={"same": "human"})
            h = h.merge(sample[["pair_id", "a", "b", "stratum", "inclusion"]], on="pair_id").merge(llm, on=["a", "b"])
            h = h[h.human.isin(["yes", "no"])]
            human = h.human.eq("yes")
            either = h.ab.eq(True) | h.ba.eq(True)
            for name, pred, w in [("yes in both orders (graph edge)", h.same, None),
                                  ("yes in either order", either, None),
                                  ("yes in both orders, weighted to all candidates", h.same, 1 / h.inclusion)]:
                rows.append({"annotator": h.annotator.iloc[0], "llm_rule": name, **binary_agreement(human, pred, w)})
            for stratum, g in h.groupby("stratum"):
                rows.append({"annotator": h.annotator.iloc[0], "llm_rule": f"both orders, stratum: {stratum}",
                             **binary_agreement(g.human.eq("yes"), g.same)})
        results["pairs"] = pd.DataFrame(rows)

    if files["names"]:
        rows = []
        for f in files["names"]:
            h = pd.read_csv(f)
            fits = h.name_fits.value_counts(normalize=True)
            rows.append({"annotator": h.annotator.iloc[0], "n": len(h), **{f"name_{k}": fits.get(k, 0.0)
                         for k in ("yes", "partly", "no")}, "coherent_yes": h.coherent.eq("yes").mean()})
        results["names"] = pd.DataFrame(rows)

    if files["attributes"]:  # the LLM in every condition against the annotators, who saw image and text
        from crypto_analysis.topic_attributes import CONDITIONS

        rows, topic_rows = [], []
        for condition, f in [(c, f) for c in CONDITIONS for f in files["attributes"]]:
            llm = pd.read_parquet(attributes_dir(out_dir) / f"samples_{condition}.parquet")
            llm["racist"] &= llm.hate
            llm["antisemitic"] &= llm.hate
            llm, h = without_other(llm), without_other(pd.read_csv(f))
            h = h.merge(llm, left_on=["topic", "sample_rank"], right_on=["topic", "rank"], suffixes=("_h", "_m"))
            for a in REPORTED:
                rows.append({"annotator": h.annotator.iloc[0], "condition": condition, "attribute": a, "unit": "sample",
                             **binary_agreement(h[f"{a}_h"].fillna(0).astype(bool), h[f"{a}_m"])})
                share = h.assign(hs=h[f"{a}_h"].fillna(0).astype(float) * h.share,
                                 ms=h[f"{a}_m"].astype(float) * h.share).groupby("topic")[["hs", "ms", "share"]].sum()
                hs, ms = share.hs / share.share, share.ms / share.share
                topic_rows.append({"annotator": h.annotator.iloc[0], "condition": condition, "attribute": a,
                                   "topics": len(share),
                                   "human_mean_share": hs.mean(), "llm_mean_share": ms.mean(),
                                   "mean_abs_diff": (hs - ms).abs().mean(),
                                   "spearman": hs.corr(ms, method="spearman") if hs.std() and ms.std() else np.nan,
                                   "topics_any_human": int((hs > 0).sum()), "topics_any_llm": int((ms > 0).sum())})
        results["attributes"] = pd.DataFrame(rows)
        results["attribute_shares"] = pd.DataFrame(topic_rows)
        if len(files["attributes"]) > 1:  # between the annotators, and the annotators with the LLM
            results["attribute_raters"] = attribute_raters(out_dir, files["attributes"])

    for name, table in results.items():
        table.to_parquet(out / f"agreement_{name}.parquet", index=False)
        print(f"\n== {name}\n" + table.round(3).to_string(index=False), flush=True)
    return results


def fleiss_kappa(labels):
    """Fleiss' kappa of a boolean (items x raters) array."""
    n = labels.shape[1]
    yes = labels.sum(axis=1)
    observed = ((yes * (yes - 1) + (n - yes) * (n - yes - 1)) / (n * (n - 1))).mean()
    chance = labels.mean() ** 2 + (1 - labels.mean()) ** 2
    return (observed - chance) / (1 - chance)


def krippendorff_alpha(labels):
    """Krippendorff's alpha of a boolean (items x raters) array with no missing labels."""
    n, total = labels.shape[1], labels.size
    yes = labels.sum(axis=1)
    observed = (2 * yes * (n - yes) / (n - 1)).sum() / total
    return 1 - observed / (2 * labels.mean() * (1 - labels.mean()) * total / (total - 1))


def bootstrap_ci(statistic, n_items, draws=5000, seed=SEED):
    """95% percentile interval of ``statistic(rows)`` over items resampled with replacement."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n_items, n_items)) for _ in range(draws)]
    return np.nanpercentile(values, [2.5, 97.5])


def test_agreement(out_dir, variants=("guided_few_k18", "guided_k18")):
    """Agreement on the held-out pairs (``agreement_test.parquet``): Cohen's kappa of every pair of raters, Fleiss'
    kappa and Krippendorff's alpha of the annotators with each LLM variant (a yes in both orders), with bootstrap
    intervals, and each variant's precision and recall on the pairs the annotators agree on."""
    from sklearn.metrics import cohen_kappa_score

    from crypto_analysis.narrative_overlap import overlap_dir

    out = validation_dir(out_dir)
    frame = pd.read_parquet(out / "macro_test_sample.parquet").set_index("pair_id")
    humans = []
    for f in sorted(out.glob("macro_test_*.csv")):
        labels = pd.read_csv(f).set_index("pair_id")
        humans.append(str(labels.annotator.iloc[0]).split()[0].capitalize())
        frame[humans[-1]] = labels.same.eq("yes")
    frame = frame.reset_index()
    for variant in variants:
        judged = pd.read_parquet(overlap_dir(out_dir) / "variants" / "test" / f"{variant}.parquet")
        wide = judged.pivot_table(index=["a", "b"], columns="order", values="same", aggfunc="first").reset_index()
        wide[variant] = wide.ab.eq(True) & wide.ba.eq(True)
        frame = frame.merge(wide[["a", "b", variant]], on=["a", "b"])
    rows = []

    def add(measure, raters, statistic, subset=frame):
        def on(items):
            return statistic(subset.iloc[items])
        low, high = bootstrap_ci(on, len(subset))
        rows.append({"measure": measure, "raters": " x ".join(raters), "n": len(subset),
                     "value": on(np.arange(len(subset))), "low": low, "high": high})

    names = humans + list(variants)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not (a in variants and b in variants):
                add("cohen", [a, b], lambda g, a=a, b=b: cohen_kappa_score(g[a], g[b]))
    for raters in [humans] + [humans + [v] for v in variants]:
        add("fleiss", raters, lambda g, r=raters: fleiss_kappa(g[r].to_numpy()))
        add("alpha", raters, lambda g, r=raters: krippendorff_alpha(g[r].to_numpy()))
    agreed = frame[frame[humans].nunique(axis=1) == 1]
    for variant in variants:
        add("cohen on agreed pairs", ["annotators", variant],
            lambda g, v=variant: cohen_kappa_score(g[humans[0]], g[v]), agreed)
        r = binary_agreement(agreed[humans[0]], agreed[variant])
        rows.append({"measure": "precision on agreed pairs", "raters": variant, "n": len(agreed), "value": r["llm_precision"]})
        rows.append({"measure": "recall on agreed pairs", "raters": variant, "n": len(agreed), "value": r["llm_recall"]})
    table = pd.DataFrame(rows)
    table.to_parquet(out / "agreement_test.parquet", index=False)
    print(frame[names].sum().to_string())
    print(table.round(3).to_string(index=False), flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sample", "names-sample", "gold", "test-sample", "screen", "pages", "agreement",
                                         "test-agreement"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quantized", action="store_true", help="screen: load the LLM in 4-bit")
    parser.add_argument("--deck", choices=DECKS + ["all"], default="all", help="pages: which deck to build")
    args = parser.parse_args()
    if args.step == "screen":
        screen(args.out_dir, args.quantized)
    elif args.step == "pages":
        build_pages(args.out_dir, DECKS if args.deck == "all" else [args.deck])
    else:
        {"sample": build_samples, "names-sample": sample_names, "gold": gold_pairs, "test-sample": sample_test_pairs,
         "agreement": agreement, "test-agreement": test_agreement}[args.step](args.out_dir)


if __name__ == "__main__":
    main()
