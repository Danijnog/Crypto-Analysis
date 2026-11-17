"""Figures and tables for the presentation: the macro-topics (the paper's narratives), how the earlier narratives map
onto them, how much of the base each level covers, an image gallery of every macro-topic, and exploratory graduation
analyses by macro-topic, launch time and content attribute.

A coin's macro-topics and attribute shares are inherited from its multimodal topic after outlier reduction
(``coin_groups``); a coin whose topic sits in two macro-topics counts in both unless stated. A coin graduated when its
bonding curve completed (a row in the datalake's ``graduations``) at any time before the end of the data; 90% of
graduations happen within 24 hours of launch (median about 8 minutes).

Every formula used here is written out in ``reports/presentation/formulas.md``.

Outputs: figures (PDF + PNG) in ``reports/figures/presentation/``, tables (CSV) in ``reports/presentation/tables/``;
the coin-level outcomes are cached in ``<out>/presentation/outcomes.parquet``. Usage::

    export PUMPFUN_DATA_DIR=/path/to/pumpfun
    python -m crypto_analysis.presentation_figures figures --out-dir data/topics
    python -m crypto_analysis.presentation_figures gallery --out-dir data/topics --clip-dir data/clip
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIGURES = Path("reports/figures/presentation")
TABLES = Path("reports/presentation/tables")
SEED = 42
BOOTSTRAP = 2000
TOP_MACROS = 30  # macro-topics shown in the graduation forest
HEATMAP_MACROS = 25
MONTH_MACROS = 6
GRID = 16  # macro-topics per gallery image
# earlier narratives (coin-label ranks, ``coin_labels.NARRATIVES``) about people who are not public figures; images
# whose topic belongs to one of them have their faces blurred in the ``_blurred`` gallery
FACE_NARRATIVE_RANKS = (7, 11, 43)
HARMS = ["hate", "racist", "antisemitic", "sexual", "violence", "controlled_substances"]
HARM_LABELS = {"hate": "Hate", "racist": "Hate: racist", "antisemitic": "Hate: antisemitic", "sexual": "Sexual",
               "violence": "Violence", "controlled_substances": "Controlled substances", "none": "None of the four",
               "other": "Other (LLM only)"}
CONDITION_LABELS = {"both": "image + text", "image": "image only", "text": "text only"}


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

def outcomes(out_dir):
    """Coin-level launch time and graduation, cached: mint, created_at (unix s), day, month, graduated, grad_24h
    (graduated within 24 h of launch; missing for coins launched in the last 24 h of the data)."""
    path = Path(out_dir) / "presentation" / "outcomes.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from crypto_analysis.datalake import DATA_DIR, connect

    con = connect(DATA_DIR, threads=8)
    df = con.execute("""
        SELECT t.mint, t.created_at, g.block_time AS grad_time
        FROM tokens t LEFT JOIN (SELECT mint, min(block_time) AS block_time FROM graduations GROUP BY 1) g USING (mint)
    """).df()
    end = df.created_at.max()
    launch = pd.to_datetime(df.created_at, unit="s")
    df["day"] = launch.dt.floor("D")
    df["month"] = launch.dt.strftime("%Y-%m")
    df["graduated"] = df.grad_time.notna()
    df["grad_24h"] = ((df.grad_time - df.created_at) <= 86_400).fillna(False).astype(float)
    df.loc[df.created_at > end - 86_400, "grad_24h"] = np.nan
    df = df.drop(columns="grad_time")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_coins(out_dir):
    """Coins with a topic: mint, topic, macro_topics, attribute shares (image + text) and outcomes."""
    coins = pd.read_parquet(Path(out_dir) / "coin_groups" / "coins.parquet")
    merged = coins.merge(outcomes(out_dir), on="mint", how="inner", validate="one_to_one")
    if len(merged) != len(coins):
        raise ValueError(f"{len(coins) - len(merged):,} coins have no row in tokens")
    return merged


def topic_names(out_dir):
    leaves = pd.read_parquet(Path(out_dir) / "taxonomy" / "multimodal" / "base" / "leaves.parquet")
    return leaves.set_index("topic").theme


def _mask(text):
    from crypto_analysis.narrative_figures import _mask as mask

    return mask(str(text))


def poisson_ratio_ci(observed, expected, level=0.95):
    """Exact interval of a standardized ratio O/E, with O Poisson and E fixed (Ulm, 1990)."""
    from scipy.stats import chi2

    o, e = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
    a = (1 - level) / 2
    lo = np.where(o > 0, chi2.ppf(a, 2 * o) / 2, 0.0) / e
    hi = chi2.ppf(1 - a, 2 * (o + 1)) / 2 / e
    return lo, hi


def macro_table(out_dir, coins):
    """Macro-topics ranked by coins with size, union coverage, graduation (rate, Wilson interval, day-standardized
    ratio) and attribute prevalence (image + text)."""
    from crypto_analysis.plotting import wilson_interval

    root = Path(out_dir)
    from crypto_analysis.narrative_overlap import MACRO_TOPICS, overlap_dir

    macro = pd.read_parquet(root / "coin_groups" / "macro_topics.parquet").drop(columns="topics")
    members = pd.read_parquet(overlap_dir(root) / MACRO_TOPICS).set_index("macro_topic").topics
    macro["topics"] = macro.macro_topic.map(lambda m: sorted(int(t) for t in members[m]))
    macro = macro.sort_values(["coins", "macro_topic"], ascending=[False, True]).reset_index(drop=True)
    macro.insert(0, "rank", np.arange(1, len(macro) + 1))
    rank_of = dict(zip(macro.macro_topic, macro["rank"]))

    pairs = coins[["macro_topics", "graduated", "day"]].explode("macro_topics").rename(columns={"macro_topics": "macro"})
    pairs["macro"] = pairs.macro.astype(int)
    p_day = coins.groupby("day").graduated.mean()
    pairs["expected"] = pairs.day.map(p_day)
    grad = pairs.groupby("macro").agg(n=("graduated", "size"), graduated=("graduated", "sum"),
                                      expected=("expected", "sum"))
    macro = macro.join(grad, on="macro_topic")
    assert (macro.n == macro.coins).all(), "coin counts differ from coin_groups"

    # union coverage: share of coins with at least one macro-topic among the first k
    first = pairs.assign(rank=pairs.macro.map(rank_of)).groupby(level=0)["rank"].min()
    macro["cumulative_pct"] = 100 * first.value_counts().reindex(macro["rank"], fill_value=0).cumsum().to_numpy() / len(coins)
    macro["share_pct"] = 100 * macro.coin_share
    macro["grad_rate_pct"] = 100 * macro.graduated / macro.n
    lo, hi = wilson_interval(macro.graduated, macro.n)
    macro["grad_lo_pct"], macro["grad_hi_pct"] = 100 * lo, 100 * hi
    macro["oe_ratio"] = macro.graduated / macro.expected
    lo, hi = poisson_ratio_ci(macro.graduated, macro.expected)
    macro["oe_lo"], macro["oe_hi"] = lo, hi

    names = topic_names(root)
    size = coins.topic.value_counts()
    macro["topic_names"] = [
        "; ".join(_mask(names.get(t, f"topic {t}")) for t in sorted(ts, key=lambda t: -size.get(t, 0)))
        for ts in macro.topics]
    macro["name"] = macro.name.map(_mask)
    macro["n_topics"] = macro.topics.map(len)
    # expected coins showing at least one of the four attributes (image + text)
    any_harm = 1 - topic_shares(root, "both")["none"]
    macro["harm_coins"] = [sum(any_harm.get(t, 0) * size.get(t, 0) for t in ts) for ts in macro.topics]
    macro["harm_pct"] = 100 * macro.harm_coins / macro.n
    return macro, pairs


def coverage_curves(out_dir, coins, macro):
    """Cumulative share of coins against the number of groups, largest first, for topics, earlier narratives and
    macro-topics (union coverage, since a coin can be in two macro-topics)."""
    from crypto_analysis.narratives import ranked_narratives

    topics = coins.topic.value_counts().to_numpy()
    narratives = ranked_narratives(out_dir)
    rows = [pd.DataFrame({"level": "topics", "k": np.arange(1, len(topics) + 1),
                          "cumulative_pct": 100 * np.cumsum(topics) / len(coins)}),
            pd.DataFrame({"level": "earlier narratives", "k": narratives["rank"],
                          "cumulative_pct": 100 * narratives.cumulative}),
            pd.DataFrame({"level": "macro-topics", "k": macro["rank"], "cumulative_pct": macro.cumulative_pct})]
    return pd.concat(rows, ignore_index=True)


def transition_flows(out_dir, coins, macro):
    """Coins flowing from each earlier narrative (Hanley et al. definition) to each macro-topic, through the topics
    they share; a topic in two macro-topics sends half its coins to each (``split`` true)."""
    from crypto_analysis.narratives import narrative_dir

    root = Path(out_dir)
    narrative_of = pd.read_parquet(narrative_dir(root, "multimodal", "published") / "topics.parquet").set_index("topic").narrative
    size = coins.topic.value_counts()
    rows = []
    for m in macro.itertuples():
        for t in m.topics:
            t = int(t)
            k = sum(t in ts for ts in macro.topics)
            rows.append({"topic": t, "narrative": int(narrative_of[t]), "macro_topic": m.macro_topic,
                         "coins": size.get(t, 0) / k, "split": k > 1})
    return pd.DataFrame(rows)


def topic_shares(out_dir, condition="both"):
    """Attribute shares per topic, with ``none`` recomputed as the share showing none of the four attributes (so that
    "other" changes no reported figure, as in ``validation.without_other``)."""
    from crypto_analysis.topic_attributes import KEPT, attributes_dir

    root = Path(out_dir)
    shares = pd.read_parquet(attributes_dir(root) / f"topics_{condition}.parquet").set_index("topic")
    samples = pd.read_parquet(attributes_dir(root) / f"samples_{condition}.parquet")
    none4 = ~samples[KEPT].fillna(False).astype(bool).any(axis=1)
    shares["none"] = (none4 * samples.share).groupby(samples.topic).sum() / samples.groupby("topic").share.sum()
    return shares


def attribute_graduation(out_dir, coins, conditions=("both", "image", "text"), boot=BOOTSTRAP, seed=SEED):
    """Graduation rate among coins showing each attribute, r_a = sum_t s_ta g_t / sum_t s_ta n_t over topics t, with
    a 95% percentile interval from resampling topics (attributes are measured per topic); also the rate among coins
    not showing it and the ratio of the two. A resample with no topic showing the attribute gives no rate and is
    skipped (only possible for attributes found in a handful of topics)."""
    per_topic = coins.groupby("topic").agg(n=("graduated", "size"), g=("graduated", "sum"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_topic), size=(boot, len(per_topic)))
    rows = []
    for c in conditions:
        s = topic_shares(out_dir, c).reindex(per_topic.index).fillna(0)
        n, g = per_topic.n.to_numpy(float), per_topic.g.to_numpy(float)
        for a in HARMS + ["none", "other"]:
            w = s[a].to_numpy(float)
            rate = (w * g).sum() / (w * n).sum()
            rest = ((1 - w) * g).sum() / ((1 - w) * n).sum()
            with np.errstate(invalid="ignore"):
                boot_rate = (w[draws] * g[draws]).sum(1) / (w[draws] * n[draws]).sum(1)
                boot_rest = ((1 - w[draws]) * g[draws]).sum(1) / ((1 - w[draws]) * n[draws]).sum(1)
            ratio = boot_rate / boot_rest
            rows.append({"condition": c, "attribute": a, "coins": (w * n).sum(), "graduated": (w * g).sum(),
                         "topics_with_attribute": int((w > 0).sum()),
                         "grad_rate_pct": 100 * rate, "lo_pct": 100 * np.nanpercentile(boot_rate, 2.5),
                         "hi_pct": 100 * np.nanpercentile(boot_rate, 97.5), "rate_without_pct": 100 * rest,
                         "ratio": rate / rest, "ratio_lo": np.nanpercentile(ratio, 2.5),
                         "ratio_hi": np.nanpercentile(ratio, 97.5),
                         "resamples_without_attribute": int(np.isnan(boot_rate).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def _label_rows(macro, width=38):
    out = []
    for name, k in zip(macro.name, macro.n_topics):
        name = name if len(name) <= width else name[:width - 1] + "..."
        out.append(f"{name} ({k})" if k > 1 else name)
    return out


def coverage_figure(curves, path):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import (
        COLORS,
        FIG_6x4,
        pct_label,
        percent_axis,
        save_figure,
        tex_escape,
    )

    fig, ax = plt.subplots(figsize=(FIG_6x4[0], 3.4))
    styles = {"macro-topics": {"color": COLORS["blue"], "lw": 1.6},
              "earlier narratives": {"color": COLORS["ink_secondary"], "lw": 1.0},
              "topics": {"color": COLORS["muted"], "lw": 1.0, "ls": (0, (3, 2))}}
    for level, style in styles.items():
        c = curves[curves.level == level]
        ax.step(c.k, c.cumulative_pct / 100, where="post", **style)
    # direct labels: macro-topics above their curve, the two finer levels below theirs
    for level, k, dy in (("macro-topics", 6, 0.06), ("earlier narratives", 12, 0.07), ("topics", 30, -0.08)):
        c = curves[(curves.level == level) & (curves.k == k)].iloc[0]
        n = int(curves[curves.level == level].k.max())
        ax.text(k, c.cumulative_pct / 100 + dy, tex_escape(f"{n} {level}"), fontsize=7, ha="center",
                color=COLORS["blue"] if level == "macro-topics" else COLORS["ink_secondary"])
    m = curves[curves.level == "macro-topics"].reset_index(drop=True)
    for share in (50, 80):
        hit = m[m.cumulative_pct >= share].iloc[0]
        ax.plot(hit.k, hit.cumulative_pct / 100, "o", color=COLORS["blue"], ms=4, zorder=3)
        ax.annotate(tex_escape(f"{int(hit.k)} macro-topics: {hit.cumulative_pct:.1f}%"), (hit.k, hit.cumulative_pct / 100),
                    xytext=(6, -11), textcoords="offset points", fontsize=7, color=COLORS["ink"])
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100, 290]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0.9, 600)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(tex_escape("# of groups, largest first (log scale)"))
    ax.set_ylabel(pct_label("Coins covered"))
    percent_axis(ax)
    save_figure(fig, path)
    plt.close(fig)


def sizes_figure(macro, path, top=TOP_MACROS):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, pct_label, percent_axis, save_figure, tex_escape

    rows = macro.head(top)
    fig, ax = plt.subplots(figsize=(6, 0.16 * top + 0.7))
    colors = [COLORS["blue"] if k > 1 else COLORS["muted"] for k in rows.n_topics]
    y = np.arange(len(rows))
    ax.barh(y, rows.share_pct / 100, color=colors, height=0.72)
    for i, r in enumerate(rows.itertuples()):
        note = f"  {r.share_pct:.1f}%" + (f", {r.n_topics} topics" if r.n_topics > 1 else "")
        ax.text(r.share_pct / 100, i, tex_escape(note), va="center", fontsize=6.5, color=COLORS["ink_secondary"])
    ax.set_yticks(y, [tex_escape(n) for n in _label_rows(rows.assign(n_topics=1), 44)], fontsize=6.8)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, rows.share_pct.max() / 100 * 1.3)
    ax.grid(axis="x", color=COLORS["grid"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(pct_label("Coins"))
    percent_axis(ax, "x", decimals=0)
    ax.text(0.99, 0.02, tex_escape("blue: several topics; grey: one topic"), transform=ax.transAxes, ha="right",
            fontsize=6.5, color=COLORS["ink_secondary"])
    save_figure(fig, path)
    plt.close(fig)


def _spread(ys, min_gap):
    """Positions close to ``ys`` (descending order kept) at least ``min_gap`` apart."""
    order = np.argsort(-np.asarray(ys))
    out = np.asarray(ys, dtype=float).copy()
    for prev, cur in itertools.pairwise(order):
        out[cur] = min(out[cur], out[prev] - min_gap)
    return out


def transition_figure(flows, macro, narratives, path):
    """Alluvial from the earlier narratives (left) to the macro-topics (right): the multi-topic macro-topics are one
    node each, the single-topic ones one node together. Ribbons of topics in two macro-topics are red."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, save_figure, tex_escape

    multi = macro[macro.n_topics > 1]
    single = set(macro.macro_topic[macro.n_topics == 1])
    flows = flows.assign(node=flows.macro_topic.where(~flows.macro_topic.isin(single), -1))
    right_order = list(multi.macro_topic) + [-1]
    palette = {multi.macro_topic.iloc[0]: COLORS["aqua"], multi.macro_topic.iloc[1]: COLORS["blue"],
               multi.macro_topic.iloc[2]: COLORS["gold"]}
    color_of = {m: palette.get(m, COLORS["orange"]) for m in multi.macro_topic} | {-1: COLORS["muted"]}

    total = flows.coins.sum()
    gap = 0.012
    avail = 1 - gap * len(right_order)
    agg = flows.groupby(["narrative", "node"], as_index=False).agg(coins=("coins", "sum"), split=("split", "any"))
    # each narrative sits in the block of the node that takes most of its coins
    home = agg.sort_values("coins", ascending=False).drop_duplicates("narrative").set_index("narrative").node
    left_size = agg.groupby("narrative").coins.sum()
    pos = {n: i for i, n in enumerate(right_order)}
    left = pd.DataFrame({"coins": left_size, "home": home.reindex(left_size.index)})
    left["block"] = left.home.map(pos)
    left = left.sort_values(["block", "coins"], ascending=[True, False])

    ltop, y, prev_block = {}, 1.0, None
    for n, r in left.iterrows():
        if prev_block is not None and r.block != prev_block:
            y -= gap
        ltop[n] = y
        y -= r.coins / total * avail
        prev_block = r.block
    right_size = agg.groupby("node").coins.sum().reindex(right_order)
    rtop, y = {}, 1.0
    for node, v in right_size.items():
        rtop[node] = y
        y -= v / total * avail + gap

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    x0, x1, w = 0.0, 1.0, 0.03
    lcur, rcur = dict(ltop), dict(rtop)
    agg["li"] = agg.narrative.map({n: i for i, n in enumerate(left.index)})
    agg["ri"] = agg.node.map(pos)
    for r in agg.sort_values(["li", "ri"]).itertuples():
        h = r.coins / total * avail
        yl, yr = lcur[r.narrative], rcur[r.node]
        lcur[r.narrative] -= h
        rcur[r.node] -= h
        xa, xb = x0 + w, x1 - w
        xm = (xa + xb) / 2
        verts = [(xa, yl), (xm, yl), (xm, yr), (xb, yr), (xb, yr - h), (xm, yr - h), (xm, yl - h), (xa, yl - h), (xa, yl)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        c, alpha, z = (COLORS["red"], 0.85, 3) if r.split else (color_of[r.node], 0.4, 2)
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=c, edgecolor="none", alpha=alpha, zorder=z))
    for n, r in left.iterrows():
        h = r.coins / total * avail
        ax.add_patch(Rectangle((x0, ltop[n] - h), w, h, facecolor=COLORS["ink"], edgecolor="white", lw=0.15, zorder=4))
    for node, v in right_size.items():
        h = v / total * avail
        ax.add_patch(Rectangle((x1 - w, rtop[node] - h), w, h, color=COLORS["ink"], lw=0, zorder=4))

    # right labels: every node, spread apart
    info = macro.set_index("macro_topic")
    labels, ys = [], []
    for node, v in right_size.items():
        # shares as in the macro-topic table: a coin in two macro-topics counts in both
        if node == -1:
            share = info.share_pct[info.n_topics == 1].sum()
            labels.append(f"{len(single)} single-topic macro-topics, {share:.1f}%")
        else:
            labels.append(f"{info.name[node]}, {info.n_topics[node]} topics, {info.share_pct[node]:.1f}%")
        ys.append(rtop[node] - v / total * avail / 2)
    for text, y_text, y_node in zip(labels, _spread(ys, 0.028), ys):
        ax.plot([x1 + 0.005, x1 + 0.04], [y_node, y_text], color=COLORS["muted"], lw=0.4)
        ax.text(x1 + 0.045, y_text, tex_escape(text), va="center", ha="left", fontsize=6.3, color=COLORS["ink"])

    # left labels: the earlier narratives that hold a topic now in two macro-topics, and the largest ones
    split_narr = set(agg.narrative[agg.split])
    big = set(left.coins.nlargest(6).index)
    shown = [n for n in left.index if n in split_narr or n in big]
    ys = [ltop[n] - left.coins[n] / total * avail / 2 for n in shown]
    for n, y_text, y_node in zip(shown, _spread(ys, 0.026), ys):
        color = COLORS["red"] if n in split_narr else COLORS["ink"]
        ax.plot([x0 - 0.005, x0 - 0.04], [y_node, y_text], color=COLORS["muted"], lw=0.4)
        ax.text(x0 - 0.045, y_text, tex_escape(narratives.get(n, f"narrative {n}")), va="center", ha="right",
                fontsize=6.3, color=color)
    ax.text(x0 + w / 2, 1.035, tex_escape(f"{len(left)} earlier narratives"), ha="center", fontsize=7.5,
            color=COLORS["ink"])
    ax.text(x1 - w / 2, 1.035, tex_escape(f"{len(macro)} macro-topics"), ha="center", fontsize=7.5, color=COLORS["ink"])
    ax.text(0.5, -0.035, tex_escape("Red: topics in two macro-topics (half of their coins drawn to each). "
                                    "Labels: the largest earlier narratives and those holding a red topic."),
            ha="center", fontsize=6.3, color=COLORS["ink_secondary"])
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(-0.05, 1.06)
    ax.axis("off")
    save_figure(fig, path)
    plt.close(fig)


def graduation_forest(macro, base, path, top=TOP_MACROS):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, forest, pct_label, save_figure

    rows = macro.head(top).sort_values("grad_rate_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 0.17 * len(rows) + 0.8))
    series = [{"name": "graduated", "color": COLORS["gold"], "rate": rows.grad_rate_pct.to_numpy() / 100,
               "lo": rows.grad_lo_pct.to_numpy() / 100, "hi": rows.grad_hi_pct.to_numpy() / 100}]
    forest(ax, _label_rows(rows), series, ref=base, ref_label=f"all coins {base:.2%}")
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_xlabel(pct_label("Coins that graduated"))
    save_figure(fig, path)
    plt.close(fig)


def lift_figure(macro, n_coins, n_grad, path, label_min=20_000, n_labels=5):
    """Share of all coins against share of graduated coins, one point per macro-topic (log-log), with the diagonal
    (graduates as often as the base) and the 2x and 0.5x lines."""
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, FIG_6x4, pct_label, save_figure, tex_escape

    x = macro.n / n_coins
    y = macro.graduated.clip(lower=0.5) / n_grad
    fig, ax = plt.subplots(figsize=FIG_6x4)
    lim = np.array([x.min() * 0.6, x.max() * 1.6])
    for f, ls in ((1, "-"), (2, (0, (3, 2))), (0.5, (0, (3, 2)))):
        ax.plot(lim, lim * f, color=COLORS["ink_secondary"] if f == 1 else COLORS["muted"], lw=0.6, ls=ls, zorder=1)
    ax.text(lim[1] * 0.75, lim[1] * 0.75 * 2 * 1.1, "2x", fontsize=6.5, color=COLORS["ink_secondary"], ha="right")
    ax.text(lim[1] * 0.75, lim[1] * 0.75 * 0.5 * 0.82, "0.5x", fontsize=6.5, color=COLORS["ink_secondary"], ha="left")
    multi = macro.n_topics > 1
    ax.scatter(x[~multi], y[~multi], s=10, color=COLORS["muted"], edgecolor="white", lw=0.3, zorder=2,
               label="one topic")
    ax.scatter(x[multi], y[multi], s=18, color=COLORS["blue"], edgecolor="white", lw=0.3, zorder=3,
               label="several topics")
    big = macro[macro.n >= label_min]
    picks = pd.concat([big.nlargest(n_labels, "oe_ratio"), big.nsmallest(n_labels, "oe_ratio"), macro.head(3)])
    picks = picks.drop_duplicates("macro_topic")
    px, py = picks.n / n_coins, picks.graduated.clip(lower=0.5) / n_grad
    # labels to the right of their points, moved apart vertically (in log10 units) where they would overlap
    ly = 10 ** _spread(np.log10(py.to_numpy()), 0.07)
    for r, x_pt, y_pt, y_lab in zip(picks.itertuples(), px, py, ly):
        name = r.name if len(r.name) <= 28 else r.name[:27] + "..."
        ax.annotate(tex_escape(name), (x_pt, y_pt), xytext=(x_pt * 1.15, y_lab), fontsize=6, color=COLORS["ink"],
                    va="center", arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 0.4})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(min(y.min(), lim[0]) * 0.8, lim[1] * 2.5)
    ax.set_xlabel(pct_label("Share of all coins"))
    ax.set_ylabel(pct_label("Share of graduated coins"))
    fmt = plt.FuncFormatter(lambda v, _: tex_escape(f"{100 * v:g}"))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(loc="upper left")
    ax.grid(axis="x", color=COLORS["grid"])
    save_figure(fig, path)
    plt.close(fig)


def month_figure(coins, path):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import (
        COLORS,
        pct_label,
        percent_axis,
        save_figure,
        tex_escape,
        wilson_interval,
    )

    by = coins.groupby("month").agg(n=("graduated", "size"), g=("graduated", "sum")).reset_index()
    lo, hi = wilson_interval(by.g, by.n)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True, height_ratios=[2, 1])
    x = np.arange(len(by))
    ax.fill_between(x, lo, hi, color=COLORS["gold"], alpha=0.25, lw=0)
    ax.plot(x, by.g / by.n, "o-", color=COLORS["gold"], ms=3.5)
    for i, r in by.iterrows():
        ax.text(i, r.g / r.n, tex_escape(f"{r.g / r.n:.2%}"), fontsize=6.3, ha="center", va="bottom",
                color=COLORS["ink_secondary"], transform=ax.transData + plt.matplotlib.transforms.ScaledTranslation(
                    0, 4 / 72, fig.dpi_scale_trans))
    ax.set_ylabel(pct_label("Graduated"))
    ax.set_ylim(0, (hi.max()) * 1.25)
    percent_axis(ax, decimals=1)
    bx.bar(x, by.n, color=COLORS["muted"], width=0.7)
    bx.set_ylabel(tex_escape("Launches (k)"))
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: tex_escape(f"{v / 1000:,.0f}")))
    bx.set_xticks(x, [tex_escape(m) for m in by.month], rotation=0, fontsize=6.5)
    save_figure(fig, path)
    plt.close(fig)
    return by.assign(rate_pct=100 * by.g / by.n, lo_pct=100 * lo, hi_pct=100 * hi)


def month_macro_figure(pairs, coins, macro, path, top=MONTH_MACROS):
    """Monthly share of launches and of graduated coins in each of the ``top`` largest macro-topics."""
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, pct_label, percent_axis, save_figure, tex_escape

    pairs = pairs.join(coins.month)
    months = sorted(coins.month.unique())
    all_n = coins.groupby("month").size()
    all_g = coins[coins.graduated].groupby("month").size()
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.8), sharex=True)
    for ax, m in zip(axes.flat, macro.head(top).itertuples()):
        sub = pairs[pairs.macro == m.macro_topic]
        launches = sub.groupby("month").size().reindex(months, fill_value=0) / all_n.reindex(months)
        grads = sub[sub.graduated].groupby("month").size().reindex(months, fill_value=0) / all_g.reindex(months)
        x = np.arange(len(months))
        ax.plot(x, launches, "-", color=COLORS["blue"], lw=1.2)
        ax.plot(x, grads, "-", color=COLORS["gold"], lw=1.2)
        name = m.name if len(m.name) <= 30 else m.name[:29] + "..."
        ax.set_title(tex_escape(name), fontsize=7.5)
        ax.set_ylim(0, max(launches.max(), grads.max()) * 1.2)
        percent_axis(ax, decimals=1)
        ax.tick_params(labelsize=6.3)
        rows += [{"macro_topic": m.macro_topic, "name": m.name, "month": mo, "launch_share_pct": 100 * a,
                  "graduated_share_pct": 100 * b} for mo, a, b in zip(months, launches, grads)]
    for ax in axes[-1]:
        ax.set_xticks(x[::2], [tex_escape(mo[2:]) for mo in months[::2]], fontsize=6)
    axes[0, 0].text(0.03, 0.08, tex_escape("all launches"), transform=axes[0, 0].transAxes, color=COLORS["blue"],
                    fontsize=6.5)
    axes[0, 0].text(0.03, 0.20, tex_escape("graduated coins"), transform=axes[0, 0].transAxes, color="#a86b00",
                    fontsize=6.5)
    fig.supylabel(pct_label("Share of the month"), fontsize=8)
    save_figure(fig, path)
    plt.close(fig)
    return pd.DataFrame(rows)


def attribute_figure(table, base, path):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, forest, pct_label, save_figure

    rows = table[table.condition == "both"].set_index("attribute").reindex(HARMS + ["none"])
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    series = [{"name": "graduated", "color": COLORS["gold"], "rate": rows.grad_rate_pct.to_numpy() / 100,
               "lo": rows.lo_pct.to_numpy() / 100, "hi": rows.hi_pct.to_numpy() / 100}]
    forest(ax, [HARM_LABELS[a] for a in rows.index], series, ref=base, ref_label=f"all coins {base:.2%}")
    ax.set_xlabel(pct_label("Coins showing the attribute that graduated"))
    save_figure(fig, path)
    plt.close(fig)


def other_names(out_dir, coins, macro, condition="both"):
    """The LLM's "other" (no published definition; not a reported attribute), by the name it gave: expected coins
    (sum over marked samples of region share x topic size), samples, topics, the graduation rate of those coins
    (sum share x G_t / sum share x n_t) and the macro-topics they fall in, largest first."""
    from crypto_analysis.topic_attributes import KEPT, attributes_dir

    samples = pd.read_parquet(attributes_dir(Path(out_dir)) / f"samples_{condition}.parquet")
    per_topic = coins.groupby("topic").agg(n=("graduated", "size"), g=("graduated", "sum"))
    marked = samples[samples.other.fillna(False).astype(bool)].copy()
    marked["name"] = marked.other_name.fillna("(no name)").str.strip().str.lower().map(_mask)
    marked["also_kept"] = marked[KEPT].fillna(False).astype(bool).any(axis=1)
    marked["coins"] = marked.share * marked.topic.map(per_topic.n)
    marked["grads"] = marked.share * marked.topic.map(per_topic.g)
    macro_of = macro[["name", "topics"]].explode("topics").groupby("topics").name.agg(" | ".join)
    rows = []
    for name, grp in marked.groupby("name"):
        weight = grp.groupby("topic").coins.sum().sort_values(ascending=False)
        rows.append({"name": name, "coins": grp.coins.sum(), "samples": len(grp), "topics": grp.topic.nunique(),
                     "also_one_of_the_four": int(grp.also_kept.sum()),
                     "grad_rate_pct": 100 * grp.grads.sum() / grp.coins.sum(),
                     "macro_topics": "; ".join(dict.fromkeys(macro_of.get(t, "") for t in weight.index))})
    table = pd.DataFrame(rows).sort_values("coins", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["share_of_other_pct"] = 100 * table.coins / table.coins.sum()
    return table


def other_figure(table, path, top=20):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import (
        COLORS,
        count_label,
        save_figure,
        tex_escape,
        thousands_axis,
    )

    rows = table.head(top)
    fig, ax = plt.subplots(figsize=(5.2, 0.17 * len(rows) + 0.8))
    y = np.arange(len(rows))
    ax.barh(y, rows.coins, color=COLORS["muted"], height=0.72)
    for i, r in enumerate(rows.itertuples()):
        note = f"  {r.share_of_other_pct:.1f}%, {r.topics} topic" + ("s" if r.topics > 1 else "")
        ax.text(r.coins, i, tex_escape(note), va="center", fontsize=6.3, color=COLORS["ink_secondary"])
    ax.set_yticks(y, [tex_escape(n) for n in rows.name], fontsize=6.8)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, rows.coins.max() * 1.3)
    ax.grid(axis="x", color=COLORS["grid"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(count_label("coins (expected)"))
    thousands_axis(ax, "x")
    save_figure(fig, path)
    plt.close(fig)


def heatmap_figure(macro, path, top=HEATMAP_MACROS):
    """Attribute prevalence and graduation rate of the ``top`` macro-topics by expected coins showing any of the four
    attributes."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, GOLD_CMAP, RED_RAMP, save_figure, tex_escape

    rows = macro.sort_values("harm_coins", ascending=False).head(top)
    rows = rows.assign(name=[f"{n} [{h / 1000:.1f}k]" for n, h in zip(rows.name, rows.harm_coins)])
    values = np.column_stack([100 * rows[f"both: {a}"] for a in HARMS])
    grad = rows.grad_rate_pct.to_numpy()[:, None]
    fig, (ax, gx) = plt.subplots(1, 2, figsize=(6.4, 0.2 * top + 1.0), width_ratios=[len(HARMS), 1.25])
    reds = LinearSegmentedColormap.from_list("reds", ["#ffffff"] + RED_RAMP)
    vmax = max(5.0, np.percentile(values, 98))
    ax.imshow(values, cmap=reds, vmin=0, vmax=vmax, aspect="auto")
    gx.imshow(grad, cmap=GOLD_CMAP, vmin=0.8 * grad.min(), vmax=grad.max(), aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        if v >= 0.05:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.8,
                    color="white" if v > 0.6 * vmax else COLORS["ink"])
    for i, v in enumerate(grad[:, 0]):
        gx.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                color="white" if v > 0.65 * grad.max() else COLORS["ink"])
    ax.set_yticks(range(len(rows)), [tex_escape(n) for n in _label_rows(rows, 44)], fontsize=6.5)
    ax.set_xticks(range(len(HARMS)), [tex_escape(HARM_LABELS[a].replace("Hate: ", "")) for a in HARMS], fontsize=6.3,
                  rotation=30, ha="right")
    gx.set_yticks([])
    for a in (ax, gx):
        a.grid(False)
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
    ax.set_title(tex_escape("Coins showing the attribute (%); [expected coins showing any]"), fontsize=7.5)
    gx.set_title(tex_escape("Graduated (%)"), fontsize=7.5)
    gx.set_xticks([])
    save_figure(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------

def figures(out_dir):
    from crypto_analysis.narrative_figures import narratives_table
    from crypto_analysis.plotting import set_latex_style

    set_latex_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    coins = load_coins(out_dir)
    base = coins.graduated.mean()
    macro, pairs = macro_table(out_dir, coins)
    cols = ["rank", "macro_topic", "name", "named_by", "n_topics", "coins", "share_pct", "cumulative_pct", "graduated",
            "grad_rate_pct", "grad_lo_pct", "grad_hi_pct", "expected", "oe_ratio", "oe_lo", "oe_hi"]
    cols += [f"both: {a}" for a in HARMS + ["other"]] + ["harm_coins", "harm_pct", "topic_names"]
    out = macro[cols].copy()
    out[[f"both: {a}" for a in HARMS + ["other"]]] *= 100
    out.round(4).to_csv(TABLES / "macro_topics.csv", index=False)

    curves = coverage_curves(out_dir, coins, macro)
    curves.round(4).to_csv(TABLES / "coverage.csv", index=False)
    coverage_figure(curves, FIGURES / "coverage")
    sizes_figure(macro, FIGURES / "macro_sizes")

    earlier = narratives_table(out_dir)
    earlier_names = dict(zip(earlier.narrative, earlier.name.map(_mask)))
    flows = transition_flows(out_dir, coins, macro)
    flows.assign(narrative_name=flows.narrative.map(earlier_names),
                 macro_name=flows.macro_topic.map(macro.set_index("macro_topic").name)).to_csv(
        TABLES / "transition.csv", index=False)
    transition_figure(flows, macro, earlier_names, FIGURES / "transition")
    names = topic_names(out_dir)
    split = flows[flows.split].groupby("topic").agg(macro_topics=("macro_topic", list), coins=("coins", "sum"))
    split["topic_name"] = split.index.map(lambda t: _mask(names.get(t, "")))
    split["macro_names"] = split.macro_topics.map(lambda ms: " | ".join(macro.set_index("macro_topic").name[m] for m in ms))
    split.reset_index().to_csv(TABLES / "topics_in_two_macro_topics.csv", index=False)

    graduation_forest(macro, base, FIGURES / "graduation_by_macro")
    lift_figure(macro, len(coins), int(coins.graduated.sum()), FIGURES / "graduation_lift")
    month_figure(coins, FIGURES / "graduation_by_month").to_csv(TABLES / "graduation_by_month.csv", index=False)
    month_macro_figure(pairs, coins, macro, FIGURES / "macro_share_by_month").round(4).to_csv(
        TABLES / "macro_share_by_month.csv", index=False)
    attrs = attribute_graduation(out_dir, coins)
    attrs.round(5).to_csv(TABLES / "graduation_by_attribute.csv", index=False)
    attribute_figure(attrs, base, FIGURES / "graduation_by_attribute")
    heatmap_figure(macro, FIGURES / "macro_attributes")
    other = other_names(out_dir, coins, macro)
    other.round(4).to_csv(TABLES / "other_names.csv", index=False)
    other_figure(other, FIGURES / "other_names")

    m = curves[curves.level == "macro-topics"]
    summary = {
        "coins": len(coins), "graduated": int(coins.graduated.sum()), "base_rate_pct": 100 * base,
        "grad_24h_rate_pct": 100 * coins.grad_24h.mean(),
        "macro_topics": len(macro), "multi_topic_macro_topics": int((macro.n_topics > 1).sum()),
        "coins_in_two_macro_topics_pct": 100 * (coins.macro_topics.map(len) > 1).mean(),
        "macro_topics_for_50pct": int(m[m.cumulative_pct >= 50].k.iloc[0]),
        "macro_topics_for_80pct": int(m[m.cumulative_pct >= 80].k.iloc[0]),
        "macro_topics_for_90pct": int(m[m.cumulative_pct >= 90].k.iloc[0]),
        "topics_in_two_macro_topics": len(split),
        "first_month": coins.month.min(), "last_month": coins.month.max(),
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures in {FIGURES}, tables in {TABLES}")


def gallery(out_dir, clip_dir, seed=SEED):
    """Image grids of every macro-topic (``GRID`` per image), largest first: six k-means medoids of its coins'
    images, each with the most frequent name of its coins; ``_blurred`` versions pixelate faces in images whose
    topic belongs to an earlier narrative about people who are not public figures."""
    from crypto_analysis.clip_embeddings import load_embeddings
    from crypto_analysis.narrative_figures import (
        image_names,
        narratives_table,
        panel_grid,
        region_medoids,
    )

    root = Path(out_dir)
    coins = load_coins(root)
    macro, _ = macro_table(root, coins)
    earlier = narratives_table(root).set_index("rank")
    face_topics = {t for r in FACE_NARRATIVE_RANKS for t in earlier.topic_ids[r]}
    sha = pd.read_parquet(root / "token_topics.parquet", columns=["mint", "sha256"]).set_index("mint").sha256
    coins["sha256"] = coins.mint.map(sha)
    pairs = coins[["sha256", "topic", "macro_topics"]].explode("macro_topics").dropna(subset=["sha256"])
    keys, emb = load_embeddings(Path(clip_dir) / "images", "sha256")
    index = pd.Index(keys)
    rng = np.random.default_rng(seed)
    picked, rows = {}, []
    for m in macro.itertuples():
        sub = pairs[pairs.macro_topics == m.macro_topic]
        medoids = region_medoids(sub.sha256.to_numpy(), emb, index, rng=rng, seed=seed, label=f"macro-topic {m.rank}")
        topic_of = sub[sub.sha256.isin([s for s, _ in medoids])].groupby("sha256").topic.agg(lambda t: t.mode().iloc[0])
        picked[m.rank] = [(s, share, int(topic_of[s])) for s, share in medoids]
    names = image_names(s for v in picked.values() for s, _, _ in v)
    for rank, v in picked.items():
        rows += [{"rank": rank, "order": i, "sha256": s, "region_share": round(share, 4), "topic": t,
                  "name": names.get(s), "blurred": t in face_topics} for i, (s, share, t) in enumerate(v)]
    pd.DataFrame(rows).to_csv(TABLES / "gallery.csv", index=False)
    table = macro.set_index("rank")
    for first in range(1, len(macro) + 1, GRID):
        last = min(first + GRID - 1, len(macro))
        for blur in (False, True):
            panels = [{"title": f"{r}. {table.name[r]}",
                       "subtitle": f"{table.share_pct[r]:.1f}% of coins · {table.n_topics[r]} topic(s) · "
                                   f"{table.grad_rate_pct[r]:.2f}% graduated",
                       "images": [(s, names.get(s, "(no name)"), blur and t in face_topics) for s, _, t in picked[r]]}
                      for r in range(first, last + 1)]
            suffix = "_blurred" if blur else ""
            panel_grid(panels, FIGURES / f"gallery_{first:03d}-{last:03d}{suffix}.png")


# --------------------------------------------------------------------------------------------
# Copycats (CLIP near-duplicates, ``crypto_analysis.copycats``)
# --------------------------------------------------------------------------------------------

ROLES = ["original", "self-relaunch", "copy (other creator)", "copy of a graduate"]
KINDS = ["original", "exact copy", "near copy"]


def load_copycats(out_dir, coins):
    """``load_coins`` with the copycat kind and role of every coin whose image has a CLIP embedding."""
    from crypto_analysis.copycats import copycat_dir

    cc = pd.read_parquet(copycat_dir(out_dir) / "coins.parquet",
                         columns=["mint", "kind", "role", "self_copy", "after_graduation"])
    return coins.merge(cc, on="mint", how="inner", validate="one_to_one")


def _rate_rows(frame, by, p_day):
    """Coins, share, graduated, rate with Wilson interval and day-standardized ratio per group of ``by``."""
    from crypto_analysis.plotting import wilson_interval

    g = frame.assign(expected=frame.day.map(p_day)).groupby(by, observed=True).agg(
        coins=("graduated", "size"), graduated=("graduated", "sum"), expected=("expected", "sum")).reset_index()
    g["share_pct"] = 100 * g.coins / len(frame)
    g["grad_rate_pct"] = 100 * g.graduated / g.coins
    lo, hi = wilson_interval(g.graduated, g.coins)
    g["grad_lo_pct"], g["grad_hi_pct"] = 100 * lo, 100 * hi
    g["oe_ratio"] = g.graduated / g.expected
    g["oe_lo"], g["oe_hi"] = poisson_ratio_ci(g.graduated, g.expected)
    return g


def _topic_weighted(weights, num, den, draws):
    """sum_t w_t num_t / sum_t w_t den_t, with a percentile interval from resampling topics (``draws``)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (weights[draws] * num[draws]).sum(1) / (weights[draws] * den[draws]).sum(1)
    return (weights * num).sum() / (weights * den).sum(), np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5)


def copycat_attribute_table(out_dir, cc, boot=BOOTSTRAP, seed=SEED):
    """Per attribute (image + text): share of the coins showing it that are copies, copies of a graduate and
    self-relaunches, each with a topic-bootstrap interval (formulas as in ``attribute_graduation``)."""
    per_topic = cc.groupby("topic").agg(n=("graduated", "size"),
                                        copies=("kind", lambda k: (k != "original").sum()),
                                        after_grad=("after_graduation", "sum"),
                                        self_copy=("self_copy", "sum"))
    s = topic_shares(out_dir, "both").reindex(per_topic.index).fillna(0)
    draws = np.random.default_rng(seed).integers(0, len(per_topic), size=(boot, len(per_topic)))
    n = per_topic.n.to_numpy(float)
    rows = []
    for a in ["all coins"] + HARMS + ["none", "other"]:
        w = np.ones(len(per_topic)) if a == "all coins" else s[a].to_numpy(float)
        row = {"attribute": a, "coins": (w * n).sum()}
        for col in ("copies", "after_grad", "self_copy"):
            r, lo, hi = _topic_weighted(w, per_topic[col].to_numpy(float), n, draws)
            row.update({f"{col}_pct": 100 * r, f"{col}_lo_pct": 100 * lo, f"{col}_hi_pct": 100 * hi})
        rows.append(row)
    return pd.DataFrame(rows)


def copycat_graduation_figure(by_role_kind, base, path):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, forest, pct_label, save_figure

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    series = []
    for kind, color in (("original", COLORS["ink_secondary"]), ("exact copy", COLORS["blue"]),
                        ("near copy", COLORS["aqua"])):
        t = by_role_kind[by_role_kind.kind == kind].set_index("role").reindex(ROLES)
        series.append({"name": kind, "color": color, "rate": t.grad_rate_pct.to_numpy() / 100,
                       "lo": t.grad_lo_pct.to_numpy() / 100, "hi": t.grad_hi_pct.to_numpy() / 100})
    forest(ax, ROLES, series, ref=base, ref_label=f"all coins {base:.2%}", spacing=0.25)
    ax.set_xlabel(pct_label("Coins that graduated"))
    save_figure(fig, path)
    plt.close(fig)


def copycats_by_macro_figure(cc, pairs, macro, path, top=TOP_MACROS):
    """Share of each role among the coins of the ``top`` largest macro-topics, most copied first, and all coins."""
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import (
        COLORS,
        ROLE_COLORS,
        pct_label,
        percent_axis,
        save_figure,
        tex_escape,
    )

    roles = pairs[["macro"]].join(cc.role, how="inner")
    table = pd.crosstab(roles.macro, roles.role, normalize="index").reindex(columns=ROLES, fill_value=0)
    rows = macro.head(top).set_index("macro_topic")
    table = table.reindex(rows.index)
    table.index = _label_rows(rows.reset_index())
    table = table.sort_values("original")
    overall = cc.role.value_counts(normalize=True).reindex(ROLES, fill_value=0).rename("All coins")
    table = pd.concat([overall.to_frame().T, table])
    fig, ax = plt.subplots(figsize=(6.2, 0.17 * len(table) + 0.9))
    left = np.zeros(len(table))
    y = np.arange(len(table))
    for role in ROLES:
        ax.barh(y, table[role], left=left, color=ROLE_COLORS[role], height=0.72, label=tex_escape(role), lw=0)
        left += table[role].to_numpy()
    for i, v in enumerate(1 - table["original"]):
        ax.text(1.01, i, tex_escape(f"{v:.0%} copies"), va="center", fontsize=6.2, color=COLORS["ink_secondary"])
    ax.set_yticks(y, [tex_escape(r) for r in table.index], fontsize=6.6)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(len(table) - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=COLORS["grid"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(pct_label("Coins"))
    percent_axis(ax, "x", decimals=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=6.5, handlelength=1, borderaxespad=0.2)
    save_figure(fig, path)
    plt.close(fig)
    return table


def copycat_macro_graduation_figure(by_macro, macro, path, top=TOP_MACROS):
    """Graduation of originals against copies in the ``top`` largest macro-topics."""
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, dumbbell, pct_label, save_figure

    rows = macro.head(top)
    orig = by_macro[by_macro.copy_kind == "original"].set_index("macro").grad_rate_pct.reindex(rows.macro_topic) / 100
    copy = by_macro[by_macro.copy_kind == "copy"].set_index("macro").grad_rate_pct.reindex(rows.macro_topic) / 100
    order = np.argsort((copy - orig).to_numpy())[::-1]
    labels = [_label_rows(rows)[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 0.17 * len(rows) + 0.9))
    dumbbell(ax, labels, orig.to_numpy()[order], copy.to_numpy()[order], left_label="originals",
             right_label="copies (exact or near)", left_color=COLORS["blue"], right_color=COLORS["aqua"])
    ax.tick_params(axis="y", labelsize=6.6)
    ax.set_xlabel(pct_label("Coins that graduated"))
    save_figure(fig, path)
    plt.close(fig)


def copycat_attribute_figure(table, path):
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import COLORS, forest, pct_label, save_figure

    rows = table.set_index("attribute").reindex(HARMS + ["none"])
    overall = table.set_index("attribute").loc["all coins"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, col, color, title in ((axes[0], "copies", COLORS["aqua"], "Copies (exact or near)"),
                                  (axes[1], "after_grad", COLORS["gold"], "Copies of a graduate")):
        series = [{"name": col, "color": color, "rate": rows[f"{col}_pct"].to_numpy() / 100,
                   "lo": rows[f"{col}_lo_pct"].to_numpy() / 100, "hi": rows[f"{col}_hi_pct"].to_numpy() / 100}]
        ref = overall[f"{col}_pct"] / 100
        forest(ax, [HARM_LABELS[a] for a in rows.index], series, ref=ref, ref_label=f"all coins {ref:.1%}",
               value_fmt="{:.1%}")
        ax.set_xlabel(pct_label(title + " among coins showing it"), fontsize=7.5)
    save_figure(fig, path)
    plt.close(fig)


def copycat_topics_figure(per_topic, path):
    """One point per topic: share of its coins that are copies against its graduation rate."""
    import matplotlib.pyplot as plt

    from crypto_analysis.plotting import (
        COLORS,
        FIG_6x4,
        pct_label,
        percent_axis,
        save_figure,
        tex_escape,
    )

    fig, ax = plt.subplots(figsize=FIG_6x4)
    size = 4 + 60 * np.sqrt(per_topic.coins / per_topic.coins.max())
    ax.scatter(per_topic.copy_share, per_topic.grad_rate, s=size, color=COLORS["blue"], alpha=0.55,
               edgecolor="white", lw=0.3)
    ax.axhline(per_topic.graduated.sum() / per_topic.coins.sum(), color=COLORS["ink_secondary"], lw=0.6,
               ls=(0, (3, 2)))
    from scipy.stats import spearmanr

    rho, p = spearmanr(per_topic.copy_share, per_topic.grad_rate)
    ax.text(0.02, 0.96, tex_escape(f"Spearman rho = {rho:.2f} (p = {p:.1g}), {len(per_topic)} topics"),
            transform=ax.transAxes, fontsize=7, va="top", color=COLORS["ink"])
    picks = pd.concat([per_topic.nlargest(4, "grad_rate"), per_topic.nlargest(3, "copy_share"),
                       per_topic.nlargest(3, "coins")]).drop_duplicates("topic")
    for r in picks.itertuples():
        name = r.name if len(r.name) <= 26 else r.name[:25] + "..."
        ax.annotate(tex_escape(name), (r.copy_share, r.grad_rate), xytext=(4, 2), textcoords="offset points",
                    fontsize=5.8, color=COLORS["ink"])
    ax.set_xlabel(pct_label("Coins of the topic that are copies (exact or near)"))
    ax.set_ylabel(pct_label("Coins of the topic that graduated"))
    percent_axis(ax, "x", decimals=0)
    percent_axis(ax, "y", decimals=1)
    ax.grid(axis="x", color=COLORS["grid"])
    save_figure(fig, path)
    plt.close(fig)
    return rho, p


def copycats(out_dir):
    from crypto_analysis.copycats import THRESHOLD
    from crypto_analysis.plotting import set_latex_style

    set_latex_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    coins = load_coins(out_dir)
    p_day = coins.groupby("day").graduated.mean()
    macro, pairs = macro_table(out_dir, coins)
    cc = load_copycats(out_dir, coins)
    base = cc.graduated.mean()

    by_kind = _rate_rows(cc, ["kind"], p_day)
    by_role_kind = _rate_rows(cc, ["role", "kind"], p_day)
    by_role = _rate_rows(cc, ["role"], p_day)
    pd.concat([by_kind.assign(group="kind"), by_role.assign(group="role"), by_role_kind.assign(group="role x kind")]
              ).round(4).to_csv(TABLES / "copycats_overview.csv", index=False)
    copycat_graduation_figure(by_role_kind, base, FIGURES / "copycat_graduation")

    shares = copycats_by_macro_figure(cc, pairs, macro, FIGURES / "copycats_by_macro")
    mp = pairs[["macro", "day"]].join(cc[["graduated", "kind", "role"]], how="inner")
    mp["copy_kind"] = np.where(mp.kind == "original", "original", "copy")
    by_macro = _rate_rows(mp, ["macro", "copy_kind"], p_day)
    by_macro_role = _rate_rows(mp, ["macro", "role"], p_day)
    names = macro.set_index("macro_topic").name
    out = pd.concat([by_macro.rename(columns={"copy_kind": "group"}),
                     by_macro_role[by_macro_role.role != "original"].rename(columns={"role": "group"})])
    out.insert(1, "name", out.macro.map(names))
    out["share_pct"] = np.nan  # shares within a macro-topic are in copycats_by_macro.csv
    out.round(4).to_csv(TABLES / "copycats_macro_graduation.csv", index=False)
    (100 * shares).round(3).rename_axis("macro_topic").to_csv(TABLES / "copycats_by_macro.csv")
    copycat_macro_graduation_figure(by_macro, macro, FIGURES / "copycat_graduation_by_macro")

    attrs = copycat_attribute_table(out_dir, cc)
    attrs.round(4).to_csv(TABLES / "copycats_by_attribute.csv", index=False)
    copycat_attribute_figure(attrs, FIGURES / "copycats_by_attribute")

    per_topic = cc.groupby("topic").agg(coins=("graduated", "size"), graduated=("graduated", "sum"),
                                        copy_share=("kind", lambda k: (k != "original").mean()),
                                        after_grad_share=("after_graduation", "mean")).reset_index()
    per_topic["grad_rate"] = per_topic.graduated / per_topic.coins
    per_topic["name"] = per_topic.topic.map(topic_names(out_dir)).fillna("").map(_mask)
    rho, p = copycat_topics_figure(per_topic, FIGURES / "copycat_topics")
    per_topic.round(5).to_csv(TABLES / "copycats_by_topic.csv", index=False)

    summary = {"threshold": THRESHOLD, "coins": len(cc), "coins_without_embedding": len(coins) - len(cc),
               **{f"{k}_pct": 100 * v for k, v in cc.kind.value_counts(normalize=True).items()},
               **{f"{k}_pct": 100 * v for k, v in cc.role.value_counts(normalize=True).items()},
               "topic_spearman_copy_share_vs_graduation": rho, "topic_spearman_p": p}
    (TABLES / "copycats_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(by_role_kind.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["figures", "gallery", "copycats"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path)
    args = parser.parse_args()
    if args.command == "figures":
        figures(args.out_dir)
    elif args.command == "copycats":
        copycats(args.out_dir)
    else:
        FIGURES.mkdir(parents=True, exist_ok=True)
        TABLES.mkdir(parents=True, exist_ok=True)
        gallery(args.out_dir, args.clip_dir)


if __name__ == "__main__":
    main()
