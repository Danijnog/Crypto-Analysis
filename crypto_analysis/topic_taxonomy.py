"""Macro-topics read off the topic hierarchy by an LLM, without a preset number of groups.

The topics of one modality are arranged in a Ward tree of their centroid embeddings. An LLM
(Qwen3.5-9B, greedy, no thinking) then

1. describes every topic (leaf) from its card (``topic_annotation``: c-TF-IDF words and coverage
   sample, plus the collage for image and multimodal topics): its theme, "none" when the samples share
   none, and which samples are antisemitic, racist or sexual. Weighting those samples by their region
   shares estimates the share of the topic carrying each attribute; these attributes cut across themes,
   so they are kept apart from the macro-topics;
2. walks the tree bottom-up and, at every node whose two children are both coherent, decides whether
   they belong to the same macro-topic, a broad content area defined by examples (``MACRO_LEVEL``),
   not by a number. A node is coherent when both children are and they belong together;
3. takes the highest coherent nodes as candidate macro-topics (topics without a theme form the group
   "mixed"). The tree often splits a macro-topic, as sibling topics can be close in embedding space for
   their surface form (e.g. "ICE" and "snow"), so the candidates are then filed one at a time, largest
   first, into an existing macro-topic or a new one (as TopicGPT, Pham et al. 2024), and a last pass
   merges duplicate macro-topics. This is the only step allowed to cross the hierarchy.

The number of macro-topics follows from these decisions. Runs differ by prompt variant (``base``;
``swap``: the two children shown in reverse order; ``paraphrase``: reworded prompts), so ``stability``
can measure how much the partition depends on them (adjusted Rand index between runs).

Writes to ``<out>/taxonomy/<modality>/``: ``linkage.npy`` and ``sizes.npy``, and per run
``<run>/leaves.parquet``, ``<run>/nodes.parquet``, ``<run>/macros.json`` and ``<run>/topics.parquet``
(topic, candidate, macro). Usage::

    python -m crypto_analysis.topic_taxonomy tree      --out-dir data/topics --modality text
    python -m crypto_analysis.topic_taxonomy build     --out-dir data/topics --modality text --run base
    python -m crypto_analysis.topic_taxonomy stability --out-dir data/topics --modality text
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image

from crypto_analysis.topic_annotation import OPEN_INPUT, generate, load_llm
from crypto_analysis.topic_search import MODALITIES, load_modality

RUNS = ["base", "swap", "paraphrase"]
ATTRIBUTES = ["antisemitic", "racist", "sexual"]
MACRO_LEVEL = ("animals, politics, sports, crypto-native culture, internet memes and characters, celebrities, "
               "food, religion, space, sexual content, hate")

LEAF = {
    "base": (
        "You are helping researchers describe the topics of the memecoins launched on pump.fun, a Solana "
        "launchpad where anyone can create a token by giving it a name, an optional description and an "
        "image. BERTopic grouped the tokens into topics. You see one topic. Its {k} samples cover the whole "
        "topic: the topic was split into {k} regions and each sample is the member closest to the centre of "
        "one region, shown with the share of the topic in that region. The collage layout and its numbers "
        "were added by the researchers.\n\n{body}\n\n"
        "Answer with a JSON object only, with these keys:\n"
        '- "theme": the theme shared by most samples, in at most 6 words, or "none" if the samples share no '
        "theme. Being a memecoin, or a theme so broad that it fits almost any memecoin (memes, humor, pop "
        "culture, random names), is not a theme\n"
        '- "summary": one sentence on what the tokens of this topic are about\n'
        '- "antisemitic", "racist", "sexual": lists of the numbers of the samples that are antisemitic, '
        "racist (other than antisemitic), or sexual or pornographic; [] if none"
    ),
    "paraphrase": (
        "Context: pump.fun is a Solana launchpad on which anyone can launch a memecoin with a name, an "
        "optional description and a picture. A topic model clustered these tokens, and researchers want to "
        "know what one of the clusters is about. The {k} numbered samples below were picked to span the whole "
        "cluster, one per region, each with the fraction of the cluster its region holds. The researchers "
        "drew the collage grid and its numbers.\n\n{body}\n\n"
        "Reply with nothing but a JSON object containing:\n"
        '- "theme": in 6 words or fewer, the subject that most samples have in common; write "none" when '
        "there is no common subject. Generic subjects that would describe nearly every memecoin (being a "
        "coin, memes, jokes, pop culture, made-up names) do not count\n"
        '- "summary": a single sentence describing what these tokens are about\n'
        '- "antisemitic", "racist", "sexual": for each, the list of sample numbers with antisemitic, racist '
        "(non-antisemitic) or sexual/pornographic content; an empty list when there are none"
    ),
}
NODE = {
    "base": (
        "You are helping researchers build a taxonomy of the memecoins launched on pump.fun. Their topics "
        "are arranged in a hierarchy; groups A and B below are siblings in it, merged because their content "
        "is similar.\n\n{a}\n\n{b}\n\n"
        "Do A and B belong to the same macro-topic? A macro-topic is a broad content area that researchers "
        "would use to describe what the platform's tokens are about, at the level of: {level}.\n"
        "Answer with a JSON object only, with these keys:\n"
        '- "same": true if A and B belong to the same macro-topic, false otherwise\n'
        '- "theme": if same, the macro-topic of A and B in at most 5 words, otherwise ""\n'
        '- "summary": if same, one sentence describing A and B together, otherwise ""'
    ),
    "paraphrase": (
        "Researchers are organising the memecoins of pump.fun into a taxonomy. Their topics form a tree, "
        "and the two groups below, A and B, sit next to each other in it because their content is close.\n\n"
        "{a}\n\n{b}\n\n"
        "Question: would a researcher file A and B under one and the same macro-topic? Macro-topics are broad "
        "areas of content describing what the tokens are about, comparable to: {level}.\n"
        "Reply with nothing but a JSON object containing:\n"
        '- "same": true when A and B fall under one macro-topic, else false\n'
        '- "theme": when same, a name of 5 words or fewer for that macro-topic, else ""\n'
        '- "summary": when same, a single sentence about A and B together, else ""'
    ),
}
ASSIGN = {
    "base": (
        "You are helping researchers build a taxonomy of the memecoins launched on pump.fun. A macro-topic is "
        "a broad content area that researchers would use to describe what the platform's tokens are about, at "
        "the level of: {level}.\n\nMacro-topics so far:\n{macros}\n\nNew group of topics: {group}\n\n"
        "Does the new group belong to one of the macro-topics so far? If none fits, create a new macro-topic "
        "at that broad level, not a narrower one. Answer with a JSON object only, with these keys:\n"
        '- "macro": the id of the macro-topic the group belongs to (e.g. "m2"), or "new"\n'
        '- "name": if new, the name of the new macro-topic in at most 5 words, otherwise ""\n'
        '- "definition": if new, one sentence defining it, otherwise ""'
    ),
    "paraphrase": (
        "Researchers are organising the memecoins of pump.fun into broad macro-topics: areas of content that "
        "describe what the tokens are about, comparable to: {level}.\n\nExisting macro-topics:\n{macros}\n\n"
        "Group to file: {group}\n\n"
        "Should this group be filed under one of the existing macro-topics? Only when none of them fits, open "
        "a new macro-topic, as broad as the examples above. Reply with nothing but a JSON object containing:\n"
        '- "macro": the id of the macro-topic to file it under (e.g. "m2"), or "new"\n'
        '- "name": when new, a name of 5 words or fewer, else ""\n'
        '- "definition": when new, a one-sentence definition, else ""'
    ),
}
DEDUP = (
    "You are helping researchers build a taxonomy of the memecoins launched on pump.fun. These are its "
    "macro-topics, broad content areas at the level of: {level}.\n\n{macros}\n\n"
    "Some of them may be the same macro-topic under different names. List only the groups of ids that are "
    "truly the same macro-topic and should be merged. Answer with a JSON object only: "
    '{{"merge": [{{"ids": ["m1", "m4"], "name": "...", "definition": "one sentence"}}]}}, '
    'or {{"merge": []}} if no macro-topics should be merged.'
)


def topic_tree(out_dir, modality, clip_dir):
    """Ward linkage of the L2-normalised topic centroids (``linkage.npy``) and topic sizes (``sizes.npy``)."""
    from scipy.cluster.hierarchy import linkage
    import torch

    out = Path(out_dir) / "taxonomy" / modality
    out.mkdir(parents=True, exist_ok=True)
    keys, _, X = load_modality(out_dir, modality, clip_dir)
    labels = pd.read_parquet(Path(out_dir) / modality / "full_topics.parquet").set_index("key").topic
    labels = labels.reindex(keys).fillna(-1).astype(np.int64).to_numpy()
    rows = np.flatnonzero(labels != -1)
    C = torch.zeros(int(labels.max()) + 1, X.shape[1], device="cuda")
    for s in range(0, len(rows), 1_000_000):
        part = rows[s:s + 1_000_000]
        C.index_add_(0, torch.from_numpy(labels[part]).cuda(), torch.from_numpy(X[part]).cuda().float())
    C = torch.nn.functional.normalize(C, dim=1).cpu().numpy()
    np.save(out / "linkage.npy", linkage(C, "ward"))
    np.save(out / "sizes.npy", np.bincount(labels[rows], minlength=len(C)))
    print(f"{modality}: tree of {len(C)} topics written", flush=True)


def parse(text):
    """The first JSON object in ``text`` after any reasoning block ({} when there is none or it does not
    parse)."""
    text = text.split("</think>")[-1]
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start >= 0 < end else {}
    except json.JSONDecodeError:
        return {}


def _numbers(value, k):
    values = value if isinstance(value, list) else [value]
    out = set()
    for v in values:
        try:
            n = int(str(v).strip().lstrip("#"))
        except ValueError:
            continue
        if 1 <= n <= k:
            out.add(n)
    return sorted(out)


class Asker:
    def __init__(self, quantized=False):
        self.model, self.processor = load_llm(quantized=quantized)
        self.calls = 0

    def __call__(self, prompt, image=None, max_new_tokens=400):
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
        self.calls += 1
        raw = generate(self.model, self.processor, [[{"role": "user", "content": content}]], max_new_tokens)[0]
        return raw, parse(raw)


def describe_leaves(ask, root, modality, variant):
    """One row per topic: theme (None when there is none), summary and the attribute shares."""
    cards = pd.read_parquet(root / "annotation" / modality / "cards.parquet").sort_values(["topic", "rank"])
    rows, start = [], time.time()
    for topic, grp in cards.groupby("topic"):
        k, shares = len(grp), grp.share.to_numpy()
        docs = grp.doc.tolist() if modality != "image" else [""] * k
        samples = "\n".join(f"{i + 1}. [{s:.0%}] {d}" for i, (d, s) in enumerate(zip(docs, shares)))
        body = OPEN_INPUT[modality].format(k=k, words=grp.words.iloc[0], samples=samples,
                                           shares=", ".join(f"{i + 1}: {s:.0%}" for i, s in enumerate(shares)))
        image = None if modality == "text" else Image.open(root / "annotation" / modality / "collages" / f"{topic}.jpg")
        raw, answer = ask(LEAF[variant].format(k=k, body=body), image)
        theme = str(answer.get("theme", "")).strip()
        row = {"topic": int(topic), "theme": None if theme.lower() in ("", "none", "null") else theme,
               "summary": str(answer.get("summary", "")).strip(), "parsed": bool(answer), "raw": raw}
        for attr in ATTRIBUTES:
            samples_with = _numbers(answer.get(attr, []), k)
            row[f"{attr}_samples"] = samples_with
            row[f"{attr}_share"] = float(shares[[n - 1 for n in samples_with]].sum()) if samples_with else 0.0
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"  leaves: {len(rows)}/{cards.topic.nunique()} ({(time.time() - start) / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


def _group_text(label, g, total):
    largest = "; ".join(theme for _, theme in g["leaf_themes"][:6])
    return (f"Group {label}: {g['theme']}. {g['summary']} ({g['n_topics']} topic(s), "
            f"{g['docs'] / total:.1%} of the documents; largest topics: {largest})")


def walk_tree(ask, Z, sizes, leaves, variant, swap, known=None):
    """Bottom-up pass over the linkage: coherence, theme and summary of every node (``known``: the
    ``nodes.parquet`` of an earlier pass, whose decisions are reused instead of asked again)."""
    known = None if known is None else known.set_index("node")
    total, n = sizes.sum(), len(sizes)
    groups = {}
    for row in leaves.itertuples():
        groups[row.topic] = {"coherent": row.theme is not None, "theme": row.theme or "", "summary": row.summary,
                             "n_topics": 1, "docs": int(sizes[row.topic]),
                             "leaf_themes": [(int(sizes[row.topic]), row.theme or "no theme")],
                             "asked": False, "raw": ""}
    start = time.time()
    for i, (a, b, height, _) in enumerate(Z):
        a, b = int(a), int(b)
        ga, gb = groups[a], groups[b]
        leaf_themes = sorted(ga["leaf_themes"] + gb["leaf_themes"], key=lambda x: -x[0])  # (docs, theme)
        node = {"coherent": False, "theme": "", "summary": "", "n_topics": ga["n_topics"] + gb["n_topics"],
                "docs": ga["docs"] + gb["docs"], "leaf_themes": leaf_themes, "asked": False, "raw": "",
                "left": a, "right": b, "height": float(height)}
        if known is not None:
            stored = known.loc[n + i]
            node.update(asked=bool(stored.asked), raw=stored.raw, coherent=bool(stored.coherent), theme=stored.theme,
                        summary=stored.summary)
        elif ga["coherent"] and gb["coherent"]:
            first, second = (gb, ga) if swap else (ga, gb)
            raw, answer = ask(NODE[variant].format(a=_group_text("A", first, total), b=_group_text("B", second, total),
                                                   level=MACRO_LEVEL), max_new_tokens=200)
            same = answer.get("same")
            same = same if isinstance(same, bool) else str(same).strip().lower() == "true"
            node.update(asked=True, raw=raw, coherent=same, theme=str(answer.get("theme", "")).strip(),
                        summary=str(answer.get("summary", "")).strip())
        groups[n + i] = node
        if (i + 1) % 50 == 0:
            print(f"  nodes: {i + 1}/{len(Z)} ({(time.time() - start) / 60:.1f} min)", flush=True)
    return groups


def _leaves_under(node, groups, n):
    stack, out = [node], []
    while stack:
        g = stack.pop()
        if g < n:
            out.append(g)
        else:
            stack += [groups[g]["left"], groups[g]["right"]]
    return out


def candidates(groups, n):
    """Highest coherent nodes: ``{candidate id: node}`` and the topics under each."""
    parent = {}
    for node, g in groups.items():
        if node >= n:
            parent[g["left"]] = parent[g["right"]] = node
    tops = [node for node, g in groups.items() if g["coherent"] and not groups.get(parent.get(node), {}).get("coherent")]
    tops.sort(key=lambda node: -groups[node]["docs"])
    return {f"c{i}": node for i, node in enumerate(tops)}


def _macro_list(macros):
    return "\n".join(f"m{i}: {m['name']}. {m['definition']} (e.g. {'; '.join(m['examples'][:5])})"
                     for i, m in enumerate(macros)) or "(none yet)"


def assign_candidates(ask, cands, groups, total, variant):
    """Files the candidates, largest first, one LLM decision each (as TopicGPT): into an existing
    macro-topic or a new one. Returns the macro-topics and the raw answers."""
    macros, raws = [], []
    for cid, node in cands.items():
        g = groups[node]
        group = _group_text("", g, total).replace("Group : ", "", 1)
        raw, answer = ask(ASSIGN[variant].format(level=MACRO_LEVEL, macros=_macro_list(macros), group=group),
                          max_new_tokens=150)
        raws.append({"candidate": cid, "raw": raw})
        choice = str(answer.get("macro", "")).strip().lower()
        index = int(choice[1:]) if choice[1:].isdigit() and choice.startswith("m") else -1
        if 0 <= index < len(macros):
            macros[index]["members"].append(cid)
            macros[index]["examples"].append(g["theme"])
        else:
            macros.append({"name": str(answer.get("name", "")).strip() or g["theme"],
                           "definition": str(answer.get("definition", "")).strip() or g["summary"],
                           "members": [cid], "examples": [g["theme"]]})
    return macros, raws


def dedup_macros(ask, macros):
    """One pass over the (short) list of macro-topics merging those that are the same."""
    raw, answer = ask(DEDUP.format(level=MACRO_LEVEL, macros=_macro_list(macros)), max_new_tokens=1500)
    merged_into = {}
    for merge in answer.get("merge", []) if isinstance(answer.get("merge"), list) else []:
        ids = [int(s[1:]) for s in map(str, merge.get("ids", [])) if s[1:].isdigit() and s.startswith("m")]
        ids = [i for i in dict.fromkeys(ids) if i < len(macros) and i not in merged_into]
        if len(ids) < 2:
            continue
        keep = macros[ids[0]]
        keep["name"] = str(merge.get("name", "")).strip() or keep["name"]
        keep["definition"] = str(merge.get("definition", "")).strip() or keep["definition"]
        keep["merged_from"] = [macros[i]["name"] for i in ids]
        for i in ids[1:]:
            keep["members"] += macros[i]["members"]
            keep["examples"] += macros[i]["examples"]
            merged_into[i] = ids[0]
    return [m for i, m in enumerate(macros) if i not in merged_into], raw


def leaves_only(out_dir, modality):
    """Only the leaf descriptions (theme, summary, attribute shares) of ``modality``'s topics, with the
    base prompt: ``taxonomy/<modality>/base/leaves.parquet``. They name the narratives built on them."""
    out = Path(out_dir) / "taxonomy" / modality / "base"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    leaves = describe_leaves(Asker(), Path(out_dir), modality, "base")
    leaves.to_parquet(out / "leaves.parquet", index=False)
    print(f"{modality}: {leaves.theme.notna().sum()}/{len(leaves)} topics with a theme, "
          f"{(~leaves.parsed).sum()} unparsed ({(time.time() - start) / 60:.1f} min)", flush=True)


def build(out_dir, modality, run):
    root = Path(out_dir)
    tree_dir = root / "taxonomy" / modality
    out = tree_dir / run
    out.mkdir(parents=True, exist_ok=True)
    Z, sizes = np.load(tree_dir / "linkage.npy"), np.load(tree_dir / "sizes.npy")
    n, total = len(sizes), sizes.sum()
    variant = "paraphrase" if run == "paraphrase" else "base"
    ask = Asker()
    start = time.time()
    base_leaves = tree_dir / "base" / "leaves.parquet"
    if run == "swap" and base_leaves.exists():  # same leaf prompt as base: only the node order changes
        shutil.copy(base_leaves, out / "leaves.parquet")
    if (out / "leaves.parquet").exists():
        leaves = pd.read_parquet(out / "leaves.parquet")
    else:
        leaves = describe_leaves(ask, root, modality, variant)
        leaves.to_parquet(out / "leaves.parquet", index=False)
    print(f"{modality}/{run}: {leaves.theme.notna().sum()}/{n} topics with a theme, "
          f"{(~leaves.parsed).sum()} unparsed ({(time.time() - start) / 60:.1f} min)", flush=True)

    known = pd.read_parquet(out / "nodes.parquet") if (out / "nodes.parquet").exists() else None
    groups = walk_tree(ask, Z, sizes, leaves, variant, swap=run == "swap", known=known)
    nodes = pd.DataFrame([{"node": k, **{c: v for c, v in g.items() if c != "leaf_themes"}}
                          for k, g in groups.items() if k >= n])
    nodes.to_parquet(out / "nodes.parquet", index=False)
    cands = candidates(groups, n)
    macros, raws = assign_candidates(ask, cands, groups, total, variant)
    assigned = len(macros)
    macros, dedup_raw = dedup_macros(ask, macros)
    print(f"{modality}/{run}: {len(cands)} candidates filed into {assigned} macro-topics, {len(macros)} after "
          f"merging duplicates", flush=True)

    topic_rows = {t: {"topic": t, "candidate": "mixed", "macro": "mixed"} for t in range(n)}
    for macro in macros:
        topics = [t for cid in macro["members"] for t in _leaves_under(cands[cid], groups, n)]
        macro.update(topics=sorted(topics), docs=int(sizes[topics].sum()), share=float(sizes[topics].sum() / total))
        for cid in macro["members"]:
            for t in _leaves_under(cands[cid], groups, n):
                topic_rows[t].update(candidate=cid, macro=macro["name"])
    macros.sort(key=lambda m: -m["docs"])
    mixed = [t for t, r in topic_rows.items() if r["macro"] == "mixed"]
    summary = {"run": run, "modality": modality, "candidates": len(cands), "macros": len(macros),
               "mixed_topics": len(mixed), "mixed_share": float(sizes[mixed].sum() / total), "llm_calls": ask.calls,
               "minutes": (time.time() - start) / 60,
               "candidate_nodes": {cid: {"node": int(node), "theme": groups[node]["theme"],
                                         "n_topics": groups[node]["n_topics"]} for cid, node in cands.items()},
               "assign_raw": raws, "dedup_raw": dedup_raw, "macro_topics": macros}
    (out / "macros.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    pd.DataFrame(topic_rows.values()).to_parquet(out / "topics.parquet", index=False)
    print(f"{modality}/{run}: {len(cands)} candidates -> {len(macros)} macro-topics; mixed: {len(mixed)} topics "
          f"({summary['mixed_share']:.1%} of the documents); {ask.calls} LLM calls, {summary['minutes']:.1f} min",
          flush=True)
    for m in macros:
        print(f"  {m['share']:6.1%}  {len(m['topics']):3d} topics  {m['name']}", flush=True)


def stability(out_dir, modality):
    """Adjusted Rand index between the runs' topic partitions, at candidate and macro level."""
    from sklearn.metrics import adjusted_rand_score

    tree_dir = Path(out_dir) / "taxonomy" / modality
    runs = {run: pd.read_parquet(tree_dir / run / "topics.parquet").set_index("topic").sort_index()
            for run in RUNS if (tree_dir / run / "topics.parquet").exists()}
    rows = [{"a": a, "b": b, "level": level, "ari": adjusted_rand_score(runs[a][level], runs[b][level])}
            for a, b in combinations(runs, 2) for level in ["candidate", "macro"]]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False), flush=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["tree", "leaves", "build", "stability"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=Path("data/clip"))
    parser.add_argument("--modality", choices=MODALITIES, default="text")
    parser.add_argument("--run", choices=RUNS, default="base")
    args = parser.parse_args()
    if args.step == "tree":
        topic_tree(args.out_dir, args.modality, args.clip_dir)
    elif args.step == "leaves":
        leaves_only(args.out_dir, args.modality)
    elif args.step == "build":
        build(args.out_dir, args.modality, args.run)
    else:
        stability(args.out_dir, args.modality)


if __name__ == "__main__":
    main()
