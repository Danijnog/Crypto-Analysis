"""Content attributes of every multimodal topic, measured as the share of the topic that shows each one.

The LLM (Qwen3.5-9B, greedy) sees a topic's ``CARD_SAMPLES`` coverage samples (numbered collage of their
images and numbered texts) with the definitions of the four attributes the paper keeps (hate, with a racist,
antisemitic or other target; sexual; violence; controlled substances; quoted in ``coin_labels.ATTRIBUTES``)
and lists, per attribute, the samples that show it; a sample showing harmful or sensitive content outside
them is "other", named. A topic's share of an attribute is the region-weighted share of its samples that
show it (each sample stands for its k-means region of the topic, ``topic_annotation.coverage_sample``); coins
inherit their topic's shares. The samples it marks are the topic's highlights for that attribute.

Three conditions measure what each modality contributes: ``both`` (images and texts), ``image`` (images
only) and ``text`` (texts only). Outputs in ``<out>/topic_attributes/``: ``samples_<condition>.parquet``
(topic, rank, share and one flag per attribute) and ``topics_<condition>.parquet`` (topic shares). Usage::

    python -m crypto_analysis.topic_attributes label   --out-dir data/topics
    python -m crypto_analysis.topic_attributes summary --out-dir data/topics
"""

import argparse
import json
from pathlib import Path
import time

import pandas as pd
from PIL import Image

from crypto_analysis.coin_labels import ANTISEMITISM, ATTRIBUTES
from crypto_analysis.topic_annotation import CARD_SAMPLES

KEPT = ["hate", "sexual", "violence", "controlled_substances"]
TARGETS = ["racist", "antisemitic", "other_target"]
CONDITIONS = ["both", "image", "text"]

PROMPT = """You label the content of memecoins launched on pump.fun, a Solana launchpad where anyone can create a \
token with a name, a description and an image. Below is one topic of tokens, shown by {k} numbered samples that cover \
the whole topic{shown}. Label every sample from what it shows{from_what}.

ATTRIBUTES. Mark an attribute only when its definition is met. Tone, intent and whether it is a joke do not matter. \
An object associated with an attribute (a weapon, a pill, a flag) is not enough by itself: what the sample shows or says \
must meet the definition. Do not guess what an unfamiliar symbol means; read a hidden meaning only when it is a known \
code (e.g. 1488). Being a memecoin is never an attribute by itself.
- hate: {hate}
  Targets: racist = race, ethnicity, nationality or immigration status; antisemitic = Jews ({antisemitism}); \
other_target = any other protected characteristic.
- sexual: {sexual}
- violence: {violence}
- controlled_substances: {controlled_substances}
- other: harmful or sensitive content that none of the attributes above covers, named by the kind of harm in at \
most 3 words. Never use it to describe what a sample shows: a sample that is not harmful or sensitive gets no label.

TOPIC
Name: {name}
Keywords: {words}
{samples}

Answer with a JSON object only, listing sample numbers (1 to {k}); [] when none:
{{"hate": [...], "racist": [...], "antisemitic": [...], "other_target": [...], "sexual": [...], "violence": [...], \
"controlled_substances": [...], "other": [{{"sample": n, "name": "..."}}]}}"""

SHOWN = {"both": " (the image shows their images, numbered like the texts)",
         "image": " (the image shows their images, numbered; their texts are not given)",
         "text": " (their texts only; their images are not given)"}
FROM_WHAT = {"both": ", in the image and in the text", "image": " in the image", "text": " in the text"}


def attributes_dir(out_dir):
    return Path(out_dir) / "topic_attributes"


def topic_prompt(condition, grp, name, k):
    if condition == "image":
        samples = f"Samples: images only, numbered 1 to {k} in the attached image."
    else:
        samples = "Samples:\n" + "\n".join(f"  {i + 1}. {d[:160]}" for i, d in enumerate(grp.doc))
    return PROMPT.format(k=k, shown=SHOWN[condition], from_what=FROM_WHAT[condition], name=name or "(none)",
                         words=grp.words.iloc[0], samples=samples, antisemitism="; ".join(ANTISEMITISM),
                         **{a: ATTRIBUTES[a] for a in KEPT})


def numbers(value, k):
    out = set()
    for v in value if isinstance(value, list) else []:
        v = v.get("sample") if isinstance(v, dict) else v
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= k:
            out.add(n)
    return out


def label(out_dir, conditions=CONDITIONS, only_validation=False, samples=CARD_SAMPLES):
    """Per-sample attribute flags of every topic in each condition, resumable per condition and topic."""
    from crypto_analysis.topic_taxonomy import Asker

    root, out = Path(out_dir), attributes_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if samples == CARD_SAMPLES else f"_k{samples}"  # sensitivity runs: other coverage sample counts
    annotation = root / "annotation" / f"multimodal{suffix}"
    cards = pd.read_parquet(annotation / "cards.parquet").sort_values(["topic", "rank"])
    names = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    ask = Asker()
    for condition in conditions:
        path = out / f"samples_{condition}{suffix}.parquet"
        done = pd.read_parquet(path) if path.exists() else None
        finished = set(done.topic) if done is not None else set()
        rows, start = [], time.time()
        todo = [t for t in cards.topic.unique() if t not in finished]
        if only_validation:  # the human-labelled topics first, so their agreement can be measured early
            human = set(pd.read_parquet(root / "validation" / "attributes_sample.parquet").topic)
            todo = [t for t in todo if t in human]
        for i, topic in enumerate(todo, 1):
            grp = cards[cards.topic == topic]
            k = len(grp)
            image = None if condition == "text" else Image.open(annotation / "collages" / f"{topic}.jpg")
            raw, answer = ask(topic_prompt(condition, grp, names.get(topic), k), image, max_new_tokens=700)
            flags = {a: numbers(answer.get(a), k) for a in KEPT + TARGETS}
            other = {int(o["sample"]): str(o.get("name", "")) for o in answer.get("other") or []
                     if isinstance(o, dict) and str(o.get("sample", "")).isdigit() and 1 <= int(o["sample"]) <= k}
            for n, r in enumerate(grp.itertuples(), 1):
                row = {"topic": int(topic), "rank": int(r.rank), "share": float(r.share), "parsed": bool(answer),
                       **{a: n in flags[a] for a in KEPT + TARGETS}, "other": n in other, "other_name": other.get(n)}
                row["none"] = not any(row[a] for a in KEPT) and not row["other"]
                row["raw"] = raw if n == 1 else None  # the answer, once per topic
                rows.append(row)
            if i % 25 == 0 or i == len(todo):
                frame = pd.DataFrame(rows)
                done = pd.concat([done, frame], ignore_index=True) if done is not None else frame
                done.to_parquet(path, index=False)
                rows = []
                print(f"  {condition}: {i}/{len(todo)} topics ({(time.time() - start) / 60:.1f} min)", flush=True)
        summary(out_dir, f"{condition}{suffix}")


def summary(out_dir, condition):
    """Region-weighted share of every topic showing each attribute (``topics_<condition>.parquet``)."""
    out = attributes_dir(out_dir)
    samples = pd.read_parquet(out / f"samples_{condition}.parquet")
    cols = KEPT + TARGETS + ["other", "none"]
    weighted = samples[cols].mul(samples.share, axis=0).groupby(samples.topic).sum()
    topics = weighted.div(samples.groupby("topic").share.sum(), axis=0)
    topics["other_names"] = samples.dropna(subset=["other_name"]).groupby("topic").other_name.agg(
        lambda x: json.dumps(sorted(set(x))))
    topics["parsed"] = samples.groupby("topic").parsed.all()
    topics.reset_index().to_parquet(out / f"topics_{condition}.parquet", index=False)
    print(f"{condition}: {len(topics)} topics, {topics.parsed.mean():.1%} parsed; topics with any share > 0: "
          + ", ".join(f"{a} {int((topics[a] > 0).sum())}" for a in KEPT + ["other"]), flush=True)
    return topics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["label", "summary"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument("--only-validation", action="store_true", help="label: only the human-labelled topics")
    parser.add_argument("--samples", type=int, default=CARD_SAMPLES, help="label: coverage samples per topic")
    args = parser.parse_args()
    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    if args.step == "label":
        label(args.out_dir, conditions, args.only_validation, args.samples)
    else:
        for c in conditions:
            summary(args.out_dir, c)


if __name__ == "__main__":
    main()
