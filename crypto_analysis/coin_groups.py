"""Coin-level output of the topic pipeline: every coin's macro-topics and content attributes, inherited from its topic.

A coin belongs to one multimodal topic (after ``reduce_outliers``, ``token_topics.parquet``). Its macro-topics are
the communities that hold that topic (``narrative_overlap.MACRO_TOPICS``; a topic can be in more than one), and its
attributes are the topic's shares (``topic_attributes``: the region-weighted share of the topic's coverage samples
that show each attribute). Nothing is labelled per coin.

Outputs in ``<out>/coin_groups/``:

* ``coins.parquet``: mint, topic, macro_topics (list), and the attribute shares of the ``both`` condition.
* ``macro_topics.parquet``: per macro-topic, its topics, coins, share of all coins and, per condition (``both``,
  ``image``, ``text``), the coin-weighted prevalence of every attribute (expected share of its coins showing it).
* ``overall.parquet``: the same prevalence over all coins with a topic, per condition.

"other" is kept for the report; it has no published definition and the human annotator marked none. Usage::

    python -m crypto_analysis.coin_groups --out-dir data/topics
"""

import argparse
from pathlib import Path

import pandas as pd

from crypto_analysis.narrative_overlap import MACRO_TOPICS, overlap_dir
from crypto_analysis.topic_attributes import CONDITIONS, KEPT, TARGETS, attributes_dir

ATTRIBUTES = KEPT + TARGETS + ["other", "none"]


def build(out_dir):
    root = Path(out_dir)
    out = root / "coin_groups"
    out.mkdir(exist_ok=True)
    coins = pd.read_parquet(root / "token_topics.parquet", columns=["mint", "multimodal_topic_reduced"])
    coins = coins.rename(columns={"multimodal_topic_reduced": "topic"})
    no_topic = int(coins.topic.isna().sum())
    coins = coins.dropna(subset=["topic"]).astype({"topic": int})
    size = coins.topic.value_counts()

    macro = pd.read_parquet(overlap_dir(root) / MACRO_TOPICS)
    members = macro[["macro_topic", "topics"]].explode("topics").rename(columns={"topics": "topic"})
    members["topic"] = members.topic.astype(int)
    of_topic = members.groupby("topic").macro_topic.agg(lambda m: sorted(int(x) for x in m))

    shares = {c: pd.read_parquet(attributes_dir(root) / f"topics_{c}.parquet").set_index("topic")[ATTRIBUTES]
              for c in CONDITIONS}
    coins["macro_topics"] = coins.topic.map(of_topic)
    coins = coins.join(shares["both"], on="topic")
    coins.to_parquet(out / "coins.parquet", index=False)

    rows = []
    for m in macro.itertuples():
        topics = [int(t) for t in m.topics]
        weight = size.reindex(topics, fill_value=0)
        row = {"macro_topic": int(m.macro_topic), "name": m.name, "named_by": m.named_by, "topics": len(topics),
               "coins": int(weight.sum()), "coin_share": weight.sum() / len(coins)}
        for c in CONDITIONS:
            prevalence = shares[c].reindex(topics).mul(weight, axis=0).sum() / weight.sum()
            row.update({f"{c}: {a}": prevalence[a] for a in ATTRIBUTES})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("coins", ascending=False)
    table.to_parquet(out / "macro_topics.parquet", index=False)

    overall = pd.DataFrame({c: shares[c].reindex(size.index).mul(size, axis=0).sum() / size.sum() for c in CONDITIONS})
    overall.rename_axis("attribute").reset_index().to_parquet(out / "overall.parquet", index=False)

    multi = coins.macro_topics.map(len)
    print(f"{len(coins):,} coins with a topic ({no_topic:,} without); {(multi > 1).mean():.2%} in more than one "
          f"macro-topic; {len(table)} macro-topics, {int((table.topics > 1).sum())} with more than one topic", flush=True)
    return coins, table, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    build(parser.parse_args().out_dir)


if __name__ == "__main__":
    main()
