"""Graduation against coin-level content attributes and macro-topic hate, on the 500k coin sample.

Data: one row per sampled coin (``attribute_sample``): its topic, macro-topics, launch month, graduation, the
attribute flags of its content (image + text, Qwen3.5-9B, ``attribute_sample label``), its creator and how many
coins that creator launched among the 5.1M analysed coins (``creator_coins``, binned as in the internal review).

Sampling: every graduate is in the sample and non-graduates are drawn at the same fraction in every topic x
month stratum, so the selection depends on the outcome only and an unweighted logistic regression estimates the
odds ratios consistently (Prentice and Pyke, 1979); only the intercept moves. Rates, prevalences and conditional
probabilities use the design weights (``weight`` = 1 / inclusion probability). The weighted logistic regression
is reported as a robustness check, as point estimates only (statsmodels does not fully support cluster-robust
errors with frequency weights).

Models (logit of graduation; odds ratios with 95% intervals; standard errors clustered by topic and by creator,
since coins of one topic or one creator are not independent):
- ``attributes``: hate, sexual, violence, controlled substances, with launch month and creator size;
- ``attributes_within_topic``: the same with topic fixed effects, so each attribute is compared with coins of
  the same topic;
- ``macro_hate``: the hate prevalence of the coin's macro-topic (weighted share of its coins flagged hate, per 10
  percentage points), with launch month and creator size, clustered by macro-topic: do coins of more hateful
  macro-topics graduate more? A coin whose topic is in two macro-topics takes the mean of their prevalences.

Conditional probabilities (weighted): P(graduated | attribute) and P(graduated | no attribute), with 95% cluster
bootstrap intervals over topics; and per macro-topic, its hate prevalence and graduation rate.

Outputs in ``<out>/graduation_models/``: ``coins.parquet``, ``odds_ratios.parquet``,
``conditional.parquet``, ``macro_topics.parquet``. Usage::

    python -m crypto_analysis.graduation_models --out-dir data/topics            # needs every content labelled
    python -m crypto_analysis.graduation_models --out-dir data/topics --partial  # preview on the labelled part
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HARMS = ["hate", "sexual", "violence", "controlled_substances"]
SIZE_BINS = ([0, 1, 9, 99, 999, 1e9], ["1", "2-9", "10-99", "100-999", "1000+"])  # as reports/review/review_checks.py
DRAWS = 1000
SEED = 20261117


def models_dir(out_dir):
    return Path(out_dir) / "graduation_models"


def coin_frame(out_dir, partial=False):
    """One row per sampled coin with its flags, creator size and macro-topics."""
    from crypto_analysis.narrative_overlap import MACRO_TOPICS, overlap_dir

    root = Path(out_dir)
    sample = pd.read_parquet(root / "attribute_sample" / "sample.parquet")
    labels = pd.read_parquet(root / "attribute_sample" / "labels.parquet", columns=["content", "parsed"] + HARMS)
    labels = labels.drop_duplicates("content")
    missing = (~sample.content.isin(labels.content)).sum()
    if missing and not partial:
        raise ValueError(f"{missing:,} of {len(sample):,} sampled coins have no labels yet; use --partial for a preview")
    df = sample.merge(labels, on="content", how="inner", validate="many_to_one")
    df = df[df.parsed].drop(columns="parsed")
    creators = pd.read_parquet(root / "copycats" / "coins.parquet", columns=["mint", "creator"])
    creators["creator_coins"] = creators.creator.map(creators.creator.value_counts())
    df = df.merge(creators, on="mint", how="left", validate="one_to_one")
    df["creator_size"] = pd.cut(df.creator_coins, SIZE_BINS[0], labels=SIZE_BINS[1]).astype(str)
    macros = pd.read_parquet(overlap_dir(root) / MACRO_TOPICS)[["macro_topic", "name", "topics"]]
    membership = macros.explode("topics").rename(columns={"topics": "topic"}).astype({"topic": int})
    df["macro_topics"] = df.topic.map(membership.groupby("topic").macro_topic.agg(list))
    df[HARMS] = df[HARMS].astype(int)
    df["graduated"] = df.graduated.astype(int)
    print(f"{len(df):,} coins with labels ({len(df) / len(sample):.1%} of the sample), {df.graduated.sum():,} "
          f"graduated, {df.creator.nunique():,} creators, {df.topic.nunique()} topics", flush=True)
    return df, macros, membership


def macro_prevalence(df, membership):
    """Per macro-topic: coins (weighted), hate prevalence and graduation rate (weighted). A coin counts in each
    macro-topic of its topic."""
    rows = df.explode("macro_topics").rename(columns={"macro_topics": "macro_topic"})
    agg = rows.assign(w_hate=rows.weight * rows.hate, w_grad=rows.weight * rows.graduated).groupby("macro_topic").agg(
        sampled=("mint", "size"), coins=("weight", "sum"), hate_w=("w_hate", "sum"), grad_w=("w_grad", "sum"),
        hate_coins=("hate", "sum"), graduates=("graduated", "sum"))
    agg["hate_prevalence"] = agg.hate_w / agg.coins
    agg["graduation_rate"] = agg.grad_w / agg.coins
    agg["topics"] = membership.groupby("macro_topic").size()
    return agg.drop(columns=["hate_w", "grad_w"]).reset_index()


def logit(df, formula, clusters, weights=None):
    """Odds ratios with 95% intervals from a logit with cluster-robust errors (one or two cluster columns)."""
    import statsmodels.formula.api as smf

    groups = df[clusters].apply(lambda c: pd.factorize(c)[0]).to_numpy()
    groups = groups[:, 0] if groups.shape[1] == 1 else groups
    if weights is None:
        fit = smf.logit(formula, data=df).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    else:
        import statsmodels.api as sm
        fit = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=df[weights]).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int()
    table = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "odds_ratio": np.exp(fit.params.values),
                          "or_low": np.exp(ci[0].values), "or_high": np.exp(ci[1].values), "p_value": fit.pvalues.values})
    return table, fit


def odds_ratios(df, macro):
    hate_share = macro.set_index("macro_topic").hate_prevalence
    df = df.assign(macro_hate_10pp=df.macro_topics.map(lambda ms: 10 * np.mean([hate_share[m] for m in ms])),
                   macro_cluster=df.macro_topics.map(lambda ms: ms[0]))
    controls = "C(month) + C(creator_size, Treatment('1'))"
    attrs = " + ".join(HARMS)
    specs = {
        "attributes": (f"graduated ~ {attrs} + {controls}", ["topic", "creator"]),
        "attributes_within_topic": (f"graduated ~ {attrs} + {controls} + C(topic)", ["topic", "creator"]),
        "macro_hate": (f"graduated ~ macro_hate_10pp + {controls}", ["macro_cluster"]),
    }
    tables = []
    for name, (formula, clusters) in specs.items():
        for weighting, w in (("unweighted", None), ("weighted", "weight")):
            if name == "attributes_within_topic" and w:
                continue  # the weighted fit with 290 topic dummies adds nothing the unweighted one does not show
            table, fit = logit(df, formula, clusters, w)
            if w:  # statsmodels does not fully support cluster errors with frequency weights: point estimates only
                table[["or_low", "or_high", "p_value"]] = np.nan
            keep = table.term.isin(HARMS + ["macro_hate_10pp"]) | table.term.str.startswith("C(creator_size")
            tables.append(table[keep].assign(model=name, weighting=weighting, n=int(fit.nobs)))
            print(f"  {name} ({weighting}): {int(fit.nobs):,} coins", flush=True)
    return pd.concat(tables, ignore_index=True)


def conditional(df, draws=DRAWS, seed=SEED):
    """Weighted P(graduated | attribute) and P(graduated | no attribute), with 95% cluster bootstrap intervals
    over topics."""
    rng = np.random.default_rng(seed)
    topics = df.topic.unique()
    rows = []
    for attr in HARMS:
        sums = df.assign(w=df.weight, wg=df.weight * df.graduated).groupby(["topic", attr])[["w", "wg"]].sum()
        sums = sums.unstack(attr, fill_value=0).reindex(topics, fill_value=0)
        w, wg = sums["w"].to_numpy(), sums["wg"].to_numpy()  # topics x (0, 1)
        idx = rng.integers(0, len(topics), (draws, len(topics)))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = wg[idx].sum(1) / w[idx].sum(1)  # draws x (0, 1)
            point = wg.sum(0) / w.sum(0)
        for value in (1, 0):
            low, high = np.nanpercentile(boot[:, value], [2.5, 97.5])
            rows.append({"attribute": attr, "given": "present" if value else "absent", "coins_weighted": w[:, value].sum(),
                         "sampled": int((df[attr] == value).sum()), "p_graduated": point[value],
                         "low": low, "high": high})
        ratio = boot[:, 1] / boot[:, 0]
        rows[-2]["ratio_present_absent"] = point[1] / point[0]
        rows[-2]["ratio_low"], rows[-2]["ratio_high"] = np.nanpercentile(ratio, [2.5, 97.5])
    return pd.DataFrame(rows)


def run(out_dir, partial=False):
    out = models_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if partial else ""
    df, macros, membership = coin_frame(out_dir, partial)
    macro = macro_prevalence(df, membership).merge(macros[["macro_topic", "name"]], on="macro_topic")
    ors = odds_ratios(df, macro)
    cond = conditional(df)
    df.drop(columns="macro_topics").to_parquet(out / f"coins{suffix}.parquet", index=False)
    ors.to_parquet(out / f"odds_ratios{suffix}.parquet", index=False)
    cond.to_parquet(out / f"conditional{suffix}.parquet", index=False)
    macro.to_parquet(out / f"macro_topics{suffix}.parquet", index=False)

    pd.set_option("display.width", 250)
    print("\nodds ratios (95% CI)")
    print(ors[["model", "weighting", "term", "odds_ratio", "or_low", "or_high", "p_value", "n"]].round(4)
          .to_string(index=False))
    print("\nP(graduated | attribute), weighted, 95% cluster bootstrap over topics")
    print(cond.round(5).to_string(index=False))
    big = macro[macro.sampled >= 1000].sort_values("hate_prevalence", ascending=False)
    print("\nmacro-topics with at least 1,000 sampled coins, by hate prevalence")
    print(big[["name", "topics", "sampled", "coins", "hate_prevalence", "graduation_rate"]].round(4)
          .to_string(index=False))
    return df, ors, cond, macro


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="run on the contents labelled so far (preview)")
    args = parser.parse_args()
    run(args.out_dir, args.partial)


if __name__ == "__main__":
    main()
