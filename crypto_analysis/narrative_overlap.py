"""Narratives as overlapping communities of the graph of topics an LLM judged to share a narrative.

1. Representation: every multimodal topic is shown by its name (the LLM-written leaf theme,
   ``topic_taxonomy leaves``), its c-TF-IDF terms and its ``CARD_SAMPLES`` coverage samples (texts, and the
   numbered collage of their images).
2. Candidates: the union of each topic's ``validation.NEIGHBOURS`` nearest topics in the multimodal and the
   text space (``validation.candidate_pairs``), plus ``FAR_PAIRS`` random pairs outside it, which estimate
   how many same-narrative pairs the neighbourhoods miss. The neighbour ranks are nested, so one run gives
   every smaller k as well.
3. Judgment: Qwen3.5-9B (greedy) classifies each pair as the same macro-topic or not, few-shot with images and
   text (``GUIDED_INTRO``, ``FEW_SHOT``), with the prompt and ``JUDGE_SAMPLES`` chosen on the development pairs;
   answers are saved as they come, so a run resumes.
4. Graph: the topics as nodes; an edge where both orders said yes.

Outputs in ``<out>/narratives/overlap/``: ``far_pairs.parquet``, ``judgments_guided.parquet`` (a, b, order, same,
narrative, reason, kinds, raw); ``judgments.parquet`` is the earlier zero-shot run. Usage::

    python -m crypto_analysis.narrative_overlap judge --out-dir data/topics
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crypto_analysis.narratives import DEFINITIONS, PAIR_IMAGES, PAIR_PROMPT, pair_collage
from crypto_analysis.topic_search import SEED

FAR_PAIRS = 1000
# Our operationalisation of "the same event or issue" for pump.fun topics (context for the task, not part of
# Hanley et al.'s definition): judge the shared subject, not sample by sample; generic meme or pop culture with
# no specific subject is one broad issue; a shared broad category alone is not. Its example (dog vs cat memes)
# is not among the human-labelled pairs.
OPERATIONALIZATION = (
    "Operational interpretation. Judge the topics at the level of their underlying subject, not sample by sample: "
    "do not require every sample of both topics to refer to the same entity or event. Two topics belong to the "
    "same narrative when they address the same specific issue: the same person, character, animal, brand, "
    "project, event or cultural phenomenon. Topics about generic internet, meme or pop culture, without a more "
    "specific subject, address the same broad issue and belong to the same narrative. Sharing only a broad "
    "category is not enough: topics about different things of one category (e.g. dog memes and cat memes) are "
    "different narratives. Base the decision on the topic names, keywords and samples together."
)
# Macro-topics: broader groups of topics by thematic similarity, the topic groups of Steffen (ICWSM 2025, "we
# manually grouped the topics into broader topic groups [...] analyzing thematic similarities"), here decided
# pair by pair as a zero-shot classification, with no labelled examples in the prompt.
MACRO_DEFINITION = (
    "A macro-topic is one recognisable subject: a person, an organisation, a country or conflict, a character or "
    "franchise, a kind of animal, a product or an activity. Topics with no subject more specific than internet "
    "memes or pop culture belong to one macro-topic: internet culture. A topic centred on one specific event or "
    "episode is a different macro-topic from a topic about its subject in general, and from topics about other "
    "events, even when the same person or entity appears in both."
)
MACRO_CLASSES = (
    "SAME: both topics are mainly about the same subject, at the same level of specificity.\n"
    "DIFFERENT: they are mainly about different subjects, or one is about a specific event and the other is not."
)
MACRO_RULES = (
    "Decide by what each topic is mainly about, using its name, keywords and samples together. Topics in one "
    "macro-topic can differ in their samples and in their style. Harmful or offensive content does not decide the "
    "class."
)
MACRO_PROMPT = (
    "You classify pairs of topics of memecoins launched on pump.fun, a Solana launchpad where anyone can create a "
    "token with a name, a description and an image. A topic is a group of similar tokens. Researchers group topics "
    "into macro-topics by their thematic similarity.\n\n" + MACRO_DEFINITION + "\n\n"
    "Classify the pair into exactly one class:\n" + MACRO_CLASSES + "\n\n" + MACRO_RULES + "{images}\n\n{a}\n\n{b}\n\n"
    "Answer with a JSON object only, with these keys:\n"
    '- "label": "SAME" or "DIFFERENT"\n'
    '- "macro_topic": if SAME, the shared subject in at most 4 words, otherwise ""\n'
    '- "reason": at most 15 words'
)
# The guideline fixed when the two annotators adjudicated their disagreements (2026-11-06): what decides a pair is
# the kind of each topic. It replaces the "same level of specificity" of the first macro-topic guideline.
GUIDED_DEFINITION = (
    "A macro-topic is a broader group of topics with a thematic similarity. Each topic is one of three kinds. A "
    "focused topic follows one well-defined content pattern: most of its tokens show the same person, organisation, "
    "country or conflict, character or franchise, kind of animal, product or activity. A broad topic has no such "
    "pattern: its tokens show many unrelated things, as in internet memes or pop culture in general. A name-only "
    "topic has no shared content: its tokens share only how they were created, such as names starting with one "
    "letter or generic coin names."
)
GUIDED_CLASSES = (
    "SAME: both topics are focused on the same subject.\n"
    "SAME: both topics are focused, on different subjects that one natural broader theme covers, such as elk and "
    "raccoons (animal memecoins).\n"
    "SAME: both topics are broad (internet culture).\n"
    "SAME: both topics are name-only (token creation pattern).\n"
    "DIFFERENT: one topic is broad and the other follows a well-defined pattern, even when that pattern could fit "
    "inside the broad topic, such as internet culture and SpongeBob.\n"
    "DIFFERENT: both topics are focused, on different subjects with no natural broader theme.\n"
    "DIFFERENT: one topic is centred on a specific event and the other on its subject in general or on another "
    "event, even when the same person or entity appears in both."
)
GUIDED_RULES = (
    "Decide the kind of each topic by what its samples and images show, not by its name alone. Topics in one "
    "macro-topic can differ in their samples and in their style. Harmful or offensive content does not decide the "
    "class."
)
GUIDED_INTRO = (
    "You classify pairs of topics of memecoins launched on pump.fun, a Solana launchpad where anyone can create a "
    "token with a name, a description and an image. A topic is a group of similar tokens. Researchers group topics "
    "into macro-topics by their thematic similarity.\n\n" + GUIDED_DEFINITION + "\n\n"
    "Classify the pair into exactly one class:\n" + GUIDED_CLASSES + "\n\n" + GUIDED_RULES
)
GUIDED_PAIR = (
    "{images}\n\n{a}\n\n{b}\n\n"
    "Answer with a JSON object only, with these keys:\n"
    '- "kind_a": "focused", "broad" or "name-only"\n'
    '- "kind_b": "focused", "broad" or "name-only"\n'
    '- "label": "SAME" or "DIFFERENT"\n'
    '- "macro_topic": if SAME, the shared subject in at most 4 words, otherwise ""\n'
    '- "reason": at most 15 words'
)
GUIDED_PROMPT = GUIDED_INTRO + GUIDED_PAIR
# Few-shot: 3 positive and 2 negative pairs of the human-labelled sample, one per kind of decision the rules leave
# hardest, each shown as the LLM sees a pair (its image and its text). They are left out of every evaluation and
# their topics out of the test sample.
FEW_SHOT = [
    ("P004", "focused", "focused", "SAME", "animal memecoins", "Two different animals, covered by the theme animals."),
    ("P043", "broad", "focused", "DIFFERENT", "", "A broad topic and a well-defined franchise pattern."),
    ("P024", "broad", "broad", "SAME", "internet culture", "Both show many unrelated memes with no pattern."),
    ("P064", "focused", "focused", "DIFFERENT", "", "Different subjects; only a loose region links them, no natural theme."),
    ("P066", "name-only", "name-only", "SAME", "token creation pattern", "Tokens share a name initial, not a content."),
]
# prompt variants compared on the human-labelled pairs: (operationalisation, reasoning)
VARIANTS = {"v1": (False, False), "v2": (True, False), "v3": (True, True), "macro": (None, False),
            "guided": (None, False), "guided_few": (None, False)}
SAVE_EVERY = 200  # a multiple of BATCH and JUDGE_BATCH
BATCH = 4
JUDGE_BATCH = 2  # six images per conversation
JUDGE_SAMPLES = 18  # coverage samples per topic, chosen on the development pairs
JUDGMENTS = "judgments_guided.parquet"
COMMUNITIES = "communities_guided.parquet"  # communities.parquet holds those of the earlier zero-shot run
COMMUNITY_STATS = "community_stats_guided.parquet"
DEFINITION = "published"


def overlap_dir(out_dir):
    return Path(out_dir) / "narratives" / "overlap"


def topic_text(label, cards, names, topic, k):
    grp = cards[cards.topic == topic].sort_values("rank").head(k)
    samples = "\n".join(f"  {i + 1}. {d[:120]}" for i, d in enumerate(grp.doc))
    return (f"Topic {label}\nName: {names.get(topic) or '(none)'}\nKeywords: {grp.words.iloc[0]}\n"
            f"Samples:\n{samples}")


def few_shot_content(collages, cards, names, pairs, k):
    """The opening of a few-shot conversation: the instructions, then every ``FEW_SHOT`` pair as the LLM sees a
    pair (its image, then its text with ``k`` samples, numbered like the tiles) with its answer."""
    pairs = pairs.set_index("pair_id")
    content = [{"type": "text", "text": GUIDED_INTRO + "\n\nExamples follow, each after its image." +
                PAIR_IMAGES["multimodal"]}]
    for i, (pair_id, kind_a, kind_b, label, macro_topic, reason) in enumerate(FEW_SHOT, 1):
        a, b = int(pairs.a[pair_id]), int(pairs.b[pair_id])
        answer = {"kind_a": kind_a, "kind_b": kind_b, "label": label, "macro_topic": macro_topic, "reason": reason}
        text = (f"Example {i}\n{topic_text('A', cards, names, a, k)}\n\n{topic_text('B', cards, names, b, k)}\n\n"
                f"Answer: {json.dumps(answer)}")
        content += [{"type": "image", "image": pair_collage(collages, a, b)}, {"type": "text", "text": text}]
    return content + [{"type": "text", "text": "Now the pair to classify, after its image."}]


def far_pairs(out_dir, n=FAR_PAIRS, seed=SEED):
    """``n`` random topic pairs that are not candidates (cached)."""
    path = overlap_dir(out_dir) / "far_pairs.parquet"
    if path.exists():
        return pd.read_parquet(path)
    cands = pd.read_parquet(Path(out_dir) / "validation" / "candidates.parquet")
    taken = set(zip(cands.a, cands.b))
    n_topics = int(max(cands.a.max(), cands.b.max())) + 1
    rng, rows = np.random.default_rng(seed), set()
    while len(rows) < n:
        a, b = sorted(rng.choice(n_topics, 2, replace=False).tolist())
        if (a, b) not in taken:
            rows.add((a, b))
    far = pd.DataFrame(sorted(rows), columns=["a", "b"])
    path.parent.mkdir(parents=True, exist_ok=True)
    far.to_parquet(path, index=False)
    return far


def judge(out_dir, k=JUDGE_SAMPLES, batch=JUDGE_BATCH, only_validation=False):
    """Few-shot macro-topic classification (``GUIDED_PAIR`` after ``few_shot_content``, the configuration fixed on
    the development pairs) of every candidate and far pair, saved every ``SAVE_EVERY`` answers, so a run resumes.
    All pairs are asked in the order ab; only those answered SAME are asked in the order ba, because an edge needs
    SAME in both. ``same`` is True for SAME."""
    from crypto_analysis.topic_annotation import generate
    from crypto_analysis.topic_taxonomy import Asker, parse

    root, out = Path(out_dir), overlap_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    annotation = root / "annotation" / ("multimodal" if k == 12 else f"multimodal_k{k}")
    cards = pd.read_parquet(annotation / "cards.parquet")
    names = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    names = names.where(names.notna(), None).to_dict()
    development = pd.read_parquet(root / "validation" / "pairs_sample.parquet")
    opening = few_shot_content(annotation / "collages", cards, names, development, k)
    cands = pd.read_parquet(root / "validation" / "candidates.parquet")[["a", "b"]].assign(source="candidate")
    pairs = pd.concat([cands, far_pairs(root).assign(source="far")], ignore_index=True)
    if only_validation:  # the human-labelled pairs first, so their agreement can be measured early
        pairs = pairs.merge(development[["a", "b"]], on=["a", "b"])
    path = out / JUDGMENTS
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["a", "b", "order", "same"])
    ask, start = Asker(), time.time()
    print(f"{len(pairs):,} pairs ({len(cands):,} candidates)", flush=True)
    for order in ("ab", "ba"):
        wanted = [(int(a), int(b)) for a, b in zip(pairs.a, pairs.b)]
        if order == "ba":
            yes = done[(done.order == "ab") & done.same.eq(True)]
            yes = set(zip(yes.a, yes.b))
            wanted = [p for p in wanted if p in yes]
        seen = set(zip(done.a[done.order == order], done.b[done.order == order]))
        jobs = [p for p in wanted if p not in seen]
        print(f"order {order}: {len(jobs):,} judgments to go", flush=True)
        rows = []
        for s in range(0, len(jobs), batch):  # greedy answers, several pairs per forward pass
            conversations = []
            for a, b in jobs[s:s + batch]:
                first, second = (a, b) if order == "ab" else (b, a)
                prompt = GUIDED_PAIR.format(images=PAIR_IMAGES["multimodal"], a=topic_text("A", cards, names, first, k),
                                            b=topic_text("B", cards, names, second, k))
                image = pair_collage(annotation / "collages", first, second)
                conversations.append([{"role": "user", "content": opening + [{"type": "image", "image": image},
                                                                             {"type": "text", "text": prompt}]}])
            for (a, b), raw in zip(jobs[s:s + batch], generate(ask.model, ask.processor, conversations, 160)):
                answer = parse(raw)
                label = str(answer.get("label", "")).strip().upper()
                rows.append({"a": a, "b": b, "order": order, "same": {"SAME": True, "DIFFERENT": False}.get(label),
                             "narrative": str(answer.get("macro_topic", "")), "reason": str(answer.get("reason", "")),
                             "kind_a": str(answer.get("kind_a", "")), "kind_b": str(answer.get("kind_b", "")),
                             "raw": raw})
            i = min(s + batch, len(jobs))
            if i % SAVE_EVERY == 0 or i == len(jobs):
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
                done.to_parquet(path, index=False)
                rows = []
                rate = (time.time() - start) / i
                print(f"  {order} {i:,}/{len(jobs):,} ({(time.time() - start) / 60:.0f} min, "
                      f"~{rate * (len(jobs) - i) / 3600:.1f} h left in this order)", flush=True)
        start = time.time()


# Overlapping community detection (CDlib) without a preset number of communities, with the parameter
# values each one's authors give as natural (kclique: triangles; lfm: alpha = 1) and neighbours of them;
# the choice among them is made on the human-labelled pairs (``validation``).
ALGORITHMS = {
    "kclique": [{"k": k} for k in (3, 4, 5, 6)],
    "slpa": [{"t": 21, "r": r} for r in (0.1, 0.2, 0.3)],
    "demon": [{"epsilon": e, "min_com_size": 2} for e in (0.25, 0.5)],
    "lfm": [{"alpha": a} for a in (0.8, 1.0, 1.2)],
}
KS = (5, 10, 15, 20)
GENERIC_NAME = r"(?i)pop culture|internet (?:meme|culture)|meme culture|memes and|and memes|random|various|mix\b"


def judged_pairs(out_dir, judgments=JUDGMENTS):
    """One row per judged pair: its answers, ``same`` (yes in both orders), source and neighbour ranks. In
    ``JUDGMENTS`` a pair answered no in the order ab has no answer in the order ba."""
    root = Path(out_dir)
    j = pd.read_parquet(overlap_dir(root) / judgments)
    wide = j.pivot_table(index=["a", "b"], columns="order", values="same", aggfunc="first").reset_index()
    wide["same"] = (wide.get("ab") == True) & (wide.get("ba") == True)
    cands = pd.read_parquet(root / "validation" / "candidates.parquet")
    wide = wide.merge(cands, on=["a", "b"], how="left")
    wide["source"] = np.where(wide.rank_mm.notna(), "candidate", "far")
    ranks = wide[["rank_mm", "rank_text"]].where(wide[["rank_mm", "rank_text"]] > 0)
    wide["rank"] = ranks.min(axis=1)
    return wide


def graph(pairs, n_topics, k):
    """Topics as nodes; an edge for every candidate pair within rank ``k`` judged the same in both orders."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n_topics))
    keep = pairs[(pairs.source == "candidate") & (pairs["rank"] <= k) & pairs.same]
    G.add_edges_from(zip(keep.a.astype(int), keep.b.astype(int)))
    return G


def detect(G, algorithm, params, seed=SEED):
    """Communities (lists of topics) of ``G``; topics left out of every community are narratives of their own."""
    import random

    from cdlib import algorithms

    random.seed(seed)
    np.random.seed(seed)
    core = G.subgraph([n for n in G if G.degree(n) > 0]).copy()
    found = getattr(algorithms, algorithm)(core, **params).communities if core.number_of_edges() else []
    covered = {t for c in found for t in c}
    return [sorted(int(t) for t in c) for c in found] + [[int(t)] for t in G if t not in covered]


def communities(out_dir):
    """Every algorithm, parameter value and k on the judged graph (``COMMUNITIES``: k, algorithm, params,
    community, topic) with summary statistics (``COMMUNITY_STATS``)."""
    out = overlap_dir(out_dir)
    pairs = judged_pairs(out_dir)
    cands = pd.read_parquet(Path(out_dir) / "validation" / "candidates.parquet")
    n_topics = int(max(cands.a.max(), cands.b.max())) + 1
    far = pairs[pairs.source == "far"]
    print(f"{len(pairs):,} judged pairs; same in both orders: candidates "
          f"{pairs[pairs.source == 'candidate'].same.mean():.2%}, far {far.same.mean() if len(far) else float('nan'):.2%}")
    rows, stats = [], []
    for k in KS:
        G = graph(pairs, n_topics, k)
        for algorithm, grid in ALGORITHMS.items():
            for params in grid:
                comms = detect(G, algorithm, params)
                label = json.dumps(params, sort_keys=True)
                rows += [{"k": k, "algorithm": algorithm, "params": label, "community": i, "topic": t}
                         for i, c in enumerate(comms) for t in c]
                member = pd.Series([t for c in comms for t in c]).value_counts()
                stats.append({"k": k, "algorithm": algorithm, "params": label, "edges": G.number_of_edges(),
                              "communities": len(comms), "multi_topic": sum(len(c) > 1 for c in comms),
                              "overlapping_topics": int((member > 1).sum()), "largest": max(len(c) for c in comms)})
    pd.DataFrame(rows).to_parquet(out / COMMUNITIES, index=False)
    stats = pd.DataFrame(stats)
    stats.to_parquet(out / COMMUNITY_STATS, index=False)
    print(stats.to_string(index=False), flush=True)
    return stats


def evaluate_communities(out_dir):
    """Every community configuration against the human labels (``community_evaluation_guided.parquet``): a pair is
    in one macro-topic when its topics share a community. Development: the reference of the 150 pairs without the
    ``FEW_SHOT`` ones; test: the pairs both annotators labelled alike, and each annotator. The judge's own edges
    (rank within k, SAME in both orders) are the baseline."""

    from crypto_analysis.validation import binary_agreement, validation_dir

    root, out = Path(out_dir), overlap_dir(out_dir)
    val = validation_dir(root)
    dev = pd.read_parquet(val / "pairs_gold.parquet")
    dev = dev[~dev.pair_id.isin([e[0] for e in FEW_SHOT])][["a", "b", "same"]]
    test = pd.read_parquet(val / "macro_test_sample.parquet")[["pair_id", "a", "b"]]
    annotators = []
    for f in sorted(val.glob("macro_test_*.csv")):
        labels = pd.read_csv(f).set_index("pair_id")
        annotators.append(str(labels.annotator.iloc[0]).split()[0].capitalize())
        test[annotators[-1]] = test.pair_id.map(labels.same.eq("yes"))
    agreed = test[test[annotators].nunique(axis=1) == 1]
    sets = {"development": (dev, "same"), "test, agreed pairs": (agreed, annotators[0])}
    sets.update({f"test, {name}": (test, name) for name in annotators})
    pairs = judged_pairs(out_dir).set_index(["a", "b"])
    comms = pd.read_parquet(out / COMMUNITIES)
    configs = [(k, "edges", "", None) for k in KS]
    configs += [(k, alg, params, grp) for (k, alg, params), grp in comms.groupby(["k", "algorithm", "params"])]
    rows = []
    for k, algorithm, params, grp in configs:
        if grp is None:  # the judge's edges
            edge = pairs.same & (pairs["rank"] <= k)

            def same(a, b, edge=edge):
                return bool(edge.get((min(a, b), max(a, b)), False))
        else:
            member = grp.groupby("topic").community.agg(set).to_dict()

            def same(a, b, member=member):
                return bool(member.get(a, set()) & member.get(b, set()))
        row = {"k": k, "algorithm": algorithm, "params": params}
        for name, (frame, column) in sets.items():
            pred = np.array([same(a, b) for a, b in zip(frame.a, frame.b)])
            r = binary_agreement(frame[column].astype(bool), pred)
            row.update({f"{name}: kappa": r["kappa"], f"{name}: precision": r["llm_precision"],
                        f"{name}: recall": r["llm_recall"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    stats = pd.read_parquet(out / COMMUNITY_STATS)
    table = table.merge(stats, on=["k", "algorithm", "params"], how="left")
    table.to_parquet(out / "community_evaluation_guided.parquet", index=False)
    return table


NAME_PROMPT = (
    "You name macro-topics of memecoins launched on pump.fun, a Solana launchpad where anyone can create a token "
    "with a name, a description and an image. A topic is a group of similar tokens. " + GUIDED_DEFINITION + "\n\n"
    "The {n} topics below were grouped into one macro-topic. Their names, largest topic first:\n{all_names}\n\n"
    "The {shown} largest are shown by their keywords and samples; the image shows each one's sample images, one "
    "block per topic in the same order, labelled by the researchers.\n\n{topics}\n\n"
    "Answer with a JSON object only, with these keys:\n"
    '- "name": the subject the topics share, in at most 5 words\n'
    '- "description": one sentence on what the tokens of this macro-topic are about'
)
# The configuration chosen on 2026-11-07: clique percolation over the judged edges of every candidate pair
MACRO_CONFIG = (20, "kclique", {"k": 3})
MACRO_TOPICS = "macro_topics_guided.parquet"


def stacked_collage(collages, topics, width=640):
    """The collages of ``topics`` one above the other, each under a bar with its label."""
    from PIL import Image, ImageDraw, ImageFont

    parts = []
    for i, topic in enumerate(topics):
        img = Image.open(collages / f"{topic}.jpg").convert("RGB")
        bar = Image.new("RGB", (width, 32), "black")
        ImageDraw.Draw(bar).text((10, 4), f"Topic {i + 1}", fill="white", font=ImageFont.load_default(size=22))
        parts += [bar, img.resize((width, round(img.height * width / img.width)))]
    canvas = Image.new("RGB", (width, sum(p.height for p in parts)), "white")
    y = 0
    for part in parts:
        canvas.paste(part, (0, y))
        y += part.height
    return canvas


def name_communities(out_dir, config=MACRO_CONFIG, max_topics=4, samples=4, listed=60):
    """Name and description of every community of the chosen configuration (``MACRO_TOPICS``: macro_topic,
    name, description, topics, named_by). A single-topic macro-topic keeps its topic's name; a larger one is
    named by the LLM from the names of its ``listed`` largest topics and the keywords, samples and images of
    its ``max_topics`` largest."""
    from crypto_analysis.topic_taxonomy import Asker

    k, algorithm, params = config
    root, out = Path(out_dir), overlap_dir(out_dir)
    annotation = root / "annotation" / "multimodal"
    cards = pd.read_parquet(annotation / "cards.parquet")
    sizes = cards.drop_duplicates("topic").set_index("topic")["count"]
    names = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    comms = pd.read_parquet(out / COMMUNITIES)
    comms = comms[(comms.k == k) & (comms.algorithm == algorithm) & (comms.params == json.dumps(params, sort_keys=True))]
    ask, rows = None, []
    for community, grp in comms.groupby("community"):
        topics = sorted(grp.topic.tolist(), key=lambda t: -sizes.get(t, 0))
        if len(topics) == 1:
            rows.append({"macro_topic": int(community), "name": names.get(topics[0]), "description": None,
                         "topics": topics, "named_by": "topic"})
            continue
        ask = ask or Asker()
        shown = topics[:max_topics]
        text = "\n\n".join(
            f"Topic {i + 1}\nName: {names.get(t) or '(none)'}\nKeywords: {cards[cards.topic == t].words.iloc[0]}\n"
            + "\n".join(f"  - {d[:120]}" for d in cards[cards.topic == t].sort_values('rank').doc.head(samples))
            for i, t in enumerate(shown))
        prompt = NAME_PROMPT.format(n=len(topics), all_names="; ".join(str(names.get(t)) for t in topics[:listed]),
                                    shown=len(shown), topics=text)
        raw, answer = ask(prompt, stacked_collage(annotation / "collages", shown), max_new_tokens=120)
        rows.append({"macro_topic": int(community), "name": str(answer.get("name", "")),
                     "description": str(answer.get("description", "")), "topics": topics, "named_by": "llm",
                     "raw": raw})
    named = pd.DataFrame(rows)
    named.to_parquet(out / MACRO_TOPICS, index=False)
    print(f"{len(named)} macro-topics ({(named.named_by == 'llm').sum()} named by the LLM)", flush=True)
    return named


def judge_validation(out_dir, variant, k=12, batch=BATCH, test=False):
    """Both-order judgments of the human-labelled pairs with one prompt ``VARIANTS`` entry and ``k`` coverage
    samples per topic (``variants/<variant>[_k<k>].parquet``); ``v1`` is the earlier narrative prompt."""
    from crypto_analysis.topic_annotation import generate
    from crypto_analysis.topic_taxonomy import Asker, parse

    root, out = Path(out_dir), overlap_dir(out_dir) / "variants"
    out.mkdir(parents=True, exist_ok=True)
    operational, thinking = VARIANTS[variant]
    definition = DEFINITIONS[DEFINITION] + ("\n\n" + OPERATIONALIZATION if operational else "")
    suffix = "" if k == 12 else f"_k{k}"
    annotation = root / "annotation" / f"multimodal{suffix}"
    cards = pd.read_parquet(annotation / "cards.parquet")
    names = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    names = names.where(names.notna(), None).to_dict()
    pairs = pd.read_parquet(root / "validation" / "pairs_sample.parquet")
    few_shot = variant == "guided_few"
    opening = few_shot_content(annotation / "collages", cards, names, pairs, k) if few_shot else []
    if test:  # the held-out pairs instead of the development ones
        pairs = pd.read_parquet(root / "validation" / "macro_test_sample.parquet")
    jobs = [(int(a), int(b), order) for a, b in zip(pairs.a, pairs.b) for order in ("ab", "ba")]
    ask, rows, start = Asker(), [], time.time()
    size = 2 if thinking or few_shot else batch  # long answers, or six images per conversation
    for s in range(0, len(jobs), size):
        conversations = []
        for a, b, order in jobs[s:s + size]:
            first, second = (a, b) if order == "ab" else (b, a)
            texts = {"a": topic_text("A", cards, names, first, k), "b": topic_text("B", cards, names, second, k)}
            if few_shot:
                prompt = GUIDED_PAIR.format(images=PAIR_IMAGES["multimodal"], **texts)
            elif variant == "guided":
                prompt = GUIDED_PROMPT.format(images=PAIR_IMAGES["multimodal"], **texts)
            elif variant == "macro":
                prompt = MACRO_PROMPT.format(images=PAIR_IMAGES["multimodal"], **texts)
            else:
                prompt = PAIR_PROMPT.format(definition=definition, images=PAIR_IMAGES["multimodal"], **texts)
            image = pair_collage(annotation / "collages", first, second)
            conversations.append([{"role": "user", "content": opening + [{"type": "image", "image": image},
                                                                         {"type": "text", "text": prompt}]}])
        answers = generate(ask.model, ask.processor, conversations, 2048 if thinking else 160, thinking=thinking)
        for (a, b, order), raw in zip(jobs[s:s + size], answers):
            answer = parse(raw)
            same = answer.get("same")
            if variant == "macro" or variant.startswith("guided"):
                label = str(answer.get("label", "")).strip().upper()
                same = {"SAME": True, "DIFFERENT": False}.get(label)
                answer["narrative"] = answer.get("macro_topic", "")
            rows.append({"a": a, "b": b, "order": order, "same": same if isinstance(same, bool) else None,
                         "narrative": str(answer.get("narrative", "")), "reason": str(answer.get("reason", "")),
                         "kind_a": str(answer.get("kind_a", "")), "kind_b": str(answer.get("kind_b", "")),
                         "raw": raw})
    result = pd.DataFrame(rows)
    if test:
        out = out / "test"
        out.mkdir(exist_ok=True)
    result.to_parquet(out / f"{variant}{suffix}.parquet", index=False)
    print(f"{variant}{suffix}: {len(result)} judgments, {result.same.notna().mean():.1%} parsed "
          f"({(time.time() - start) / 60:.1f} min)", flush=True)


def evaluate_variants(out_dir):
    """Each prompt variant against every annotator's pair labels: confusion, precision, recall, F1, kappa,
    overall and on the pairs where both topics have a generic name (``variants/evaluation.parquet``)."""
    from sklearn.metrics import cohen_kappa_score

    root, out = Path(out_dir), overlap_dir(out_dir) / "variants"
    sample = pd.read_parquet(root / "validation" / "pairs_sample.parquet")[["pair_id", "a", "b", "stratum"]]
    names = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    generic = names.fillna("").str.contains(GENERIC_NAME)
    rows = []
    for path in sorted(p for p in out.glob("*.parquet") if p.stem.split("_k")[0] in VARIANTS):
        j = path.stem
        wide = pd.read_parquet(path).pivot_table(index=["a", "b"], columns="order", values="same", aggfunc="first")
        wide["llm"] = wide.get("ab").eq(True) & wide.get("ba").eq(True)
        files = sorted((root / "validation").glob("macro_pairs_*.csv")) or sorted((root / "validation").glob("pairs_*.csv"))
        for f in files:
            h = pd.read_csv(f).merge(sample, on="pair_id").merge(wide.reset_index(), on=["a", "b"])
            h["human"] = h.same.eq("yes")
            h["generic"] = h.a.map(generic) & h.b.map(generic)
            for subset, g in [("all", h), ("both names generic", h[h.generic]), ("not both generic", h[~h.generic])]:
                tp, fp = int((g.human & g.llm).sum()), int((~g.human & g.llm).sum())
                fn, tn = int((g.human & ~g.llm).sum()), int((~g.human & ~g.llm).sum())
                p, r = tp / (tp + fp) if tp + fp else np.nan, tp / (tp + fn) if tp + fn else np.nan
                rows.append({"variant": j, "annotator": h.annotator.iloc[0], "subset": subset, "n": len(g),
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
                             "f1": 2 * p * r / (p + r) if p + r else np.nan,
                             "kappa": cohen_kappa_score(g.human, g.llm) if g.human.nunique() + g.llm.nunique() > 2 else np.nan})
    table = pd.DataFrame(rows)
    table.to_parquet(out / "evaluation.parquet", index=False)
    print(table.round(3).to_string(index=False), flush=True)
    return table


def evaluate_gold(out_dir):
    """Each prompt variant against the human reference on the development pairs, leaving the ``FEW_SHOT`` pairs
    out of every variant so that all are compared on the same pairs (``variants/evaluation_gold.parquet``)."""
    from crypto_analysis.validation import binary_agreement

    root, out = Path(out_dir), overlap_dir(out_dir) / "variants"
    gold = pd.read_parquet(root / "validation" / "pairs_gold.parquet")
    gold = gold[~gold.pair_id.isin([e[0] for e in FEW_SHOT])]
    rows = []
    for path in sorted(p for p in out.glob("*.parquet") if p.stem.split("_k")[0] in ("macro", "guided", "guided_few")):
        wide = pd.read_parquet(path).pivot_table(index=["a", "b"], columns="order", values="same", aggfunc="first")
        wide["llm"] = wide.ab.eq(True) & wide.ba.eq(True)
        wide["orders_agree"] = wide.ab.eq(wide.ba)
        g = gold.merge(wide.reset_index(), on=["a", "b"])
        for subset, part in [("all", g), *g.groupby("source"), *g.groupby("stratum")]:
            r = binary_agreement(part.same, part.llm)
            f1 = 2 * r["llm_precision"] * r["llm_recall"] / (r["llm_precision"] + r["llm_recall"])
            rows.append({"variant": path.stem, "subset": subset, **r, "f1": f1, "orders_agree": part.orders_agree.mean()})
    table = pd.DataFrame(rows)
    table.to_parquet(out / "evaluation_gold.parquet", index=False)
    print(table.round(3).to_string(index=False), flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["judge", "communities", "evaluate-communities", "name", "variant", "evaluate",
                                         "evaluate-gold"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only-validation", action="store_true", help="judge: only the human-labelled pairs")
    parser.add_argument("--variant", choices=list(VARIANTS), default="macro", help="variant: prompt to compare")
    parser.add_argument("--samples", type=int, default=12, help="variant: coverage samples per topic")
    parser.add_argument("--test", action="store_true", help="variant: the held-out pairs of the test deck")
    args = parser.parse_args()
    if args.step == "judge":
        judge(args.out_dir, only_validation=args.only_validation)
    elif args.step == "variant":
        judge_validation(args.out_dir, args.variant, args.samples, test=args.test)
    elif args.step == "evaluate":
        evaluate_variants(args.out_dir)
    elif args.step == "evaluate-gold":
        evaluate_gold(args.out_dir)
    elif args.step == "name":
        name_communities(args.out_dir)
    elif args.step == "evaluate-communities":
        table = evaluate_communities(args.out_dir)
        print(table.round(2).to_string(index=False), flush=True)
    else:
        communities(args.out_dir)


if __name__ == "__main__":
    main()
