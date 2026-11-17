"""Coin-level content attributes on a stratified sample of coins, with every coin's inclusion probability.

Topic-level shares (``topic_attributes``) give every coin of a topic the same attribute share, although harmful
and harmless coins share topics. Here each sampled coin gets its own flags.

Sample (``draw``). The frame is every coin with a multimodal topic (``coin_groups/coins.parquet``). Strata are
topic x launch month x graduated:

- every graduated coin is taken (inclusion probability 1);
- non-graduated coins are drawn with the same fraction in every topic x month stratum (at least one per
  stratum), so the sample keeps each topic's and macro-topic's size and the monthly launch volume.

Graduates are the scarce outcome, so taking all of them gives the attribute-graduation odds ratios the most
precision the platform allows. Because the non-graduate fraction is the same in every stratum, an unweighted
logistic regression gives the right odds ratios; only its intercept moves (Prentice and Pyke, 1979). Rates
and prevalences need the weights (``weight`` = 1 / inclusion probability).

Labels (``label``). The prompt and the input format are exactly those validated by the annotators for the
attributes (``topic_attributes.PROMPT``, image + text condition): up to 12 numbered coins of one topic in a
collage with their texts, the topic name and keywords. Here the 12 coins are random sampled coins of the topic,
shuffled so graduates and non-graduates mix, instead of the topic's coverage samples. Coins with the same image
and text get the same answer (greedy decoding), so each distinct (image, text) pair is labelled once.

Outputs in ``<out>/attribute_sample/``: ``sample.parquet`` (one row per sampled coin), ``contents.parquet``
(one row per distinct image + text, with its group and position), ``labels.parquet`` (flags per content).
Usage::

    python -m crypto_analysis.attribute_sample draw  --out-dir data/topics
    python -m crypto_analysis.attribute_sample label --out-dir data/topics --limit-groups 84   # timing run
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

SIZE = 500_000
GROUP = 12  # coins per prompt, as in the validated attribute prompt (topic_annotation.CARD_SAMPLES)
SEED = 20261112


def sample_dir(out_dir):
    return Path(out_dir) / "attribute_sample"


def frame(out_dir):
    """Every coin with a multimodal topic: mint, topic, image and text keys, document, launch day and month,
    graduation."""
    from crypto_analysis.presentation_figures import outcomes

    root = Path(out_dir)
    coins = pd.read_parquet(root / "coin_groups" / "coins.parquet", columns=["mint", "topic"])
    docs = pd.read_parquet(root / "docs_token.parquet", columns=["mint", "sha256", "text_key"])
    out = outcomes(root)[["mint", "created_at", "day", "month", "graduated"]]
    df = coins.merge(docs, on="mint", how="left", validate="one_to_one").merge(out, on="mint", how="left",
                                                                               validate="one_to_one")
    missing = df[["sha256", "text_key", "month"]].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing:,} coins with a topic have no image, text or launch time")
    return df


def allocate(sizes, total):
    """Per-stratum sample sizes proportional to ``sizes``, at least 1 each, summing to ``total`` (largest
    remainder)."""
    sizes = np.asarray(sizes, dtype=float)
    base = np.ones(len(sizes))
    quota = (total - base.sum()) * (sizes - 1) / (sizes - 1).sum()
    n = base + np.floor(quota)
    left = int(total - n.sum())
    n[np.argsort(-(quota - np.floor(quota)), kind="stable")[:left]] += 1
    return np.minimum(n, sizes).astype(int)


def draw(out_dir, size=SIZE, seed=SEED):
    df = frame(out_dir)
    rng = np.random.default_rng(seed)
    grads = df[df.graduated].assign(pi=1.0)
    rest = df[~df.graduated]
    strata = rest.groupby(["topic", "month"]).size().rename("n").reset_index()
    strata["quota"] = allocate(strata.n, size - len(grads))
    strata["pi"] = strata.quota / strata.n
    rest = rest.merge(strata, on=["topic", "month"])
    rest["u"] = rng.random(len(rest))
    rest = rest[rest.groupby(["topic", "month"]).u.rank(method="first") <= rest.quota]
    sample = pd.concat([grads, rest.drop(columns=["n", "quota", "u"])], ignore_index=True)
    sample["weight"] = 1 / sample.pi
    sample["content"] = sample.sha256 + ":" + sample.text_key

    # distinct contents, shuffled within topic and cut into groups of at most GROUP (sizes differ by at most one)
    contents = sample.drop_duplicates("content")[["content", "sha256", "text_key", "topic"]].copy()
    contents["u"] = rng.random(len(contents))
    contents = contents.sort_values(["topic", "u"]).drop(columns="u").reset_index(drop=True)
    parts = []
    for topic, grp in contents.groupby("topic", sort=True):
        for j, chunk in enumerate(np.array_split(grp.index.to_numpy(), -(-len(grp) // GROUP))):
            parts.append(pd.DataFrame({"row": chunk, "group": f"{topic}-{j}", "position": np.arange(1, len(chunk) + 1)}))
    order = pd.concat(parts).set_index("row")
    contents = contents.join(order)
    # interleave topics so a partial run covers the whole frame
    rank = contents.groupby("topic").group.transform(lambda g: pd.factorize(g)[0])
    contents["order"] = rank
    contents = contents.sort_values(["order", "topic", "position"]).drop(columns="order").reset_index(drop=True)

    out = sample_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["mint", "topic", "month", "day", "created_at", "graduated", "sha256", "text_key", "content", "pi", "weight"]
    sample[cols].to_parquet(out / "sample.parquet", index=False)
    contents.to_parquet(out / "contents.parquet", index=False)
    report = check(df, sample, contents)
    (out / "sample_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return sample, contents


def check(df, sample, contents):
    """The weighted sample must reproduce the frame's topic, month and graduation totals exactly."""
    w = sample.weight
    by = lambda col: (sample.groupby(col).weight.sum() - df.groupby(col).size()).abs().max()
    return {
        "frame_coins": len(df), "sample_coins": len(sample), "graduated": int(sample.graduated.sum()),
        "graduated_share_pct": round(100 * sample.graduated.mean(), 2),
        "non_graduate_fraction": round(float(sample.loc[~sample.graduated, "pi"].median()), 5),
        "non_graduate_weight": round(float(sample.loc[~sample.graduated, "weight"].median()), 3),
        "strata_non_graduate": int(sample.loc[~sample.graduated].groupby(["topic", "month"]).ngroups),
        "weighted_total": round(float(w.sum())), "max_topic_error": float(by("topic")),
        "max_month_error": float(by("month")), "weighted_graduation_pct": round(100 * float((w * sample.graduated).sum() / w.sum()), 4),
        "frame_graduation_pct": round(100 * df.graduated.mean(), 4),
        "distinct_contents": len(contents), "groups": int(contents.group.nunique()),
    }


def label(out_dir, limit_groups=None, flush_every=50):
    """Flags of every distinct content, one prompt per group, resumable per group."""
    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import collage
    from crypto_analysis.topic_attributes import KEPT, TARGETS, numbers, topic_prompt
    from crypto_analysis.topic_taxonomy import Asker

    root, out = Path(out_dir), sample_dir(out_dir)
    contents = pd.read_parquet(out / "contents.parquet")
    docs = pd.read_parquet(root / "docs_token.parquet", columns=["sha256", "text_key", "doc"]).drop_duplicates(
        ["sha256", "text_key"])
    contents = contents.merge(docs, on=["sha256", "text_key"], how="left", validate="many_to_one")
    contents["doc"] = contents.doc.fillna("").str[:300]  # as in topic_annotation.build_cards
    words = pd.read_parquet(root / "multimodal" / "full_info.parquet").set_index("topic").words
    names = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    path = out / "labels.parquet"
    done = pd.read_parquet(path) if path.exists() else None
    finished = set(done.group) if done is not None else set()
    groups = [g for g in pd.unique(contents.group) if g not in finished]
    if limit_groups:
        groups = groups[:limit_groups]
    con = connect(DATA_DIR, threads=8)
    ask = Asker()
    rows, start, coins = [], time.time(), 0
    for start_batch in range(0, len(groups), flush_every):
        batch = groups[start_batch:start_batch + flush_every]
        part = contents[contents.group.isin(batch)]
        images = load_images(con, part.sha256, DATA_DIR / "images")
        for group, grp in part.groupby("group", sort=False):
            grp = grp.sort_values("position").assign(words=words.get(int(grp.topic.iloc[0]), ""))
            k, topic = len(grp), int(grp.topic.iloc[0])
            image = collage([images.get(s) for s in grp.sha256])
            raw, answer = ask(topic_prompt("both", grp, names.get(topic), k), image, max_new_tokens=700)
            flags = {a: numbers(answer.get(a), k) for a in KEPT + TARGETS}
            other = {int(o["sample"]): str(o.get("name", "")) for o in answer.get("other") or []
                     if isinstance(o, dict) and str(o.get("sample", "")).isdigit() and 1 <= int(o["sample"]) <= k}
            for n, r in enumerate(grp.itertuples(), 1):
                row = {"content": r.content, "group": group, "topic": topic, "position": n, "parsed": bool(answer),
                       "image_missing": r.sha256 not in images,
                       **{a: n in flags[a] for a in KEPT + TARGETS}, "other": n in other, "other_name": other.get(n)}
                row["none"] = not any(row[a] for a in KEPT)  # none of the four attributes (LLM "other" aside)
                row["raw"] = raw if n == 1 else None
                rows.append(row)
            coins += k
        frame_ = pd.DataFrame(rows)
        done = pd.concat([done, frame_], ignore_index=True) if done is not None else frame_
        done.to_parquet(path, index=False)
        rows = []
        hours = (time.time() - start) / 3600
        print(f"  {start_batch + len(batch)}/{len(groups)} groups, {coins:,} contents, {hours * 60:.1f} min, "
              f"{coins / hours:,.0f} contents/h", flush=True)
    total, hours = len(contents), (time.time() - start) / 3600
    if coins:
        print(f"rate {coins / hours:,.0f} contents/h; all {total:,} contents would take {total / (coins / hours):.1f} h "
              f"(image loading included, model loading excluded)", flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["draw", "label"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-groups", type=int, default=None, help="label: stop after this many groups")
    args = parser.parse_args()
    if args.step == "draw":
        draw(args.out_dir, args.size, args.seed)
    else:
        label(args.out_dir, args.limit_groups)


if __name__ == "__main__":
    main()
