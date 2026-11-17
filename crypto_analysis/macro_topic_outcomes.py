"""Graduation by macro-topic and by toxicity label: rates, time to graduation, and weekly series with peaks.

Questions:
- by macro-topic: the graduation rate of each, and, among graduates, whether some graduate faster than the rest
  (one against all);
- by toxicity label: the same;
- two weekly series per macro-topic, creations and graduations, and their clear peaks, to look up what happened.

Populations. Macro-topics: every analysed coin (5.1M with a multimodal topic); a coin counts in each macro-topic
of its topic (8 of 290 topics are in two). Toxicity labels: the coin-level flags of the 500k sample
(``attribute_sample``); rates use the design weights, and since every graduate is in the sample, the
time-to-graduation comparison among graduates needs no weights.

Time to graduation: hours from launch to the first graduation transaction. Coins launched near the end of the data
can only show up as graduates if they graduated fast, so the comparison keeps coins launched at least
``HORIZON_DAYS`` before the last launch and counts as graduates those that graduated within ``HORIZON_DAYS``
(98% of all graduations). One against all: Mann-Whitney U of the group's times against everyone else's,
rank-biserial effect ``faster`` = P(group faster) - P(group slower), Benjamini-Hochberg q over the groups.
Graduation rates: Wilson 95% intervals; one against all by a two-proportion chi-square test, BH q.

Peaks: weekly creations of each macro-topic as a share of all creations that week, so platform-wide surges do not
count. A week is a peak when its share is at least ``PEAK_RATIO`` times the macro-topic's median weekly share and
it has at least ``PEAK_MIN`` creations. Each peak lists the most common coin names of that macro-topic that week
and its busiest day, as a lead for looking the event up.

Events (``EVENTS``): for the largest news-driven peaks, the event of the same days found by web search on
2026-11-17, with one source each; seasonal peaks (Halloween, Christmas, New Year, Ramadan, Valentine's Day) are
left out. Peak weeks are grouped into episodes (consecutive weeks of one macro-topic) in ``episodes.parquet``.

Outputs in ``<out>/macro_topic_outcomes/``: ``coins.parquet``, ``rates.parquet``, ``times.parquet``,
``attribute_rates.parquet``, ``attribute_times.parquet``, ``weekly.parquet``, ``peaks.parquet``,
``episodes.parquet``; figures in ``reports/figures/macro_topic_outcomes/``. Usage::

    PUMPFUN_DATA_DIR=... python -m crypto_analysis.macro_topic_outcomes --out-dir data/topics [--partial]
    python -m crypto_analysis.macro_topic_outcomes --out-dir data/topics --figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_DAYS = 30
MIN_GRADUATES = 30  # time comparisons only for groups with at least this many graduates
PEAK_RATIO = 3.0
PEAK_MIN = 200
TOP_NAMES = 8
FIGURES = Path("reports/figures/macro_topic_outcomes")
# Largest news-driven peak episodes and the event of the same days (web search, 2026-11-17): macro-topic name,
# event date, short label, source.
EVENTS = [
    ("FIFA World Cup and Football", "2026-06-11", "World Cup opens", "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026"),
    ("Black Table", "2026-06-17", "$ANSEM launched", "https://cryptoast.fr/the-black-bull-lancement-memecoin-ansem/"),
    ("Jeffrey Epstein", "2026-01-30", "DOJ releases files", "https://globalnews.ca/news/11645115/jeffrey-epstein-files-release-jan-30-2026"),
    ("Venezuelan political satire", "2026-01-03", "Maduro captured", "https://www.pbs.org/newshour/world/us-strikes-venezuela-and-says-its-leader-maduro-has-been-captured-and-flown-out-of-the-country"),
    ("Trump buying Greenland", "2026-01-21", "Davos framework", "https://www.hppr.org/npr-news/2026-01-21/trump-backs-off-tariff-threats-and-hails-framework-of-a-future-deal-on-greenland"),
    ("Ayatollah Khamenei", "2026-02-28", "Khamenei killed", "https://www.npr.org/2026/02/28/nx-s1-5730158/israel-iran-strikes-trump"),
    ("Oil and Gas", "2026-03-09", "Brent above $114", "https://www.adn.com/nation-world/2026/03/09/oil-prices-swing-wildly-as-iran-war-threatens-transport-routes-and-production-across-middle-east/"),
    ("Virus and Disease", "2026-05-06", "WHO: Andes hantavirus", "https://www.ecdc.europa.eu/en/node/41897"),
    ("SpaceX and Space Exploration", "2026-06-12", "planned Nasdaq debut", "https://www.govconwire.com/articles/spacex-ipo-ticker-spcx-nasdaq"),
    ("Precious metals and gems", "2026-01-26", "gold above $5,000", "https://www.herobullion.com/silver-continues-historic-rally-gold-all-time-high-news-1-26-2026/"),
    ("Airline and Aviation Memes", "2026-01-20", "Musk-Ryanair feud", "https://www.irishtimes.com/business/2026/01/20/elon-musk-asks-followers-if-he-should-buy-ryanair"),
    ("LeBron James and the NBA", "2026-06-30", "leaves the Lakers", "https://thegrio.com/2026/06/30/lebron-james-lakers-exit-warriors-free-agency-2026/"),
]


def out_dir_of(out_dir):
    return Path(out_dir) / "macro_topic_outcomes"


# --- data --------------------------------------------------------------------------------------------------

def graduation_times(out_dir):
    """First graduation time of every graduated coin (cached)."""
    path = out_dir_of(out_dir) / "graduation_times.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from crypto_analysis.datalake import DATA_DIR, connect

    con = connect(DATA_DIR, threads=8)
    times = con.execute("SELECT mint, min(block_time) AS grad_time FROM graduations GROUP BY 1").df()
    path.parent.mkdir(parents=True, exist_ok=True)
    times.to_parquet(path, index=False)
    return times


def coin_frame(out_dir):
    """Every analysed coin: topic, macro-topics, launch and graduation time, hours to graduation and whether it
    enters the time comparison (``in_horizon``: launched at least ``HORIZON_DAYS`` before the last launch)."""
    from crypto_analysis.presentation_figures import outcomes

    root = Path(out_dir)
    coins = pd.read_parquet(root / "coin_groups" / "coins.parquet", columns=["mint", "topic", "macro_topics"])
    df = coins.merge(outcomes(root)[["mint", "created_at", "graduated"]], on="mint", how="left", validate="one_to_one")
    df = df.merge(graduation_times(out_dir), on="mint", how="left", validate="one_to_one")
    df["hours"] = (df.grad_time - df.created_at) / 3600
    horizon = HORIZON_DAYS * 86_400
    df["in_horizon"] = df.created_at <= df.created_at.max() - horizon
    df["graduated_in_horizon"] = df.graduated & (df.hours <= HORIZON_DAYS * 24)
    df["macro_topics"] = df.macro_topics.map(list)
    return df


def macro_names(out_dir):
    from crypto_analysis.narrative_overlap import MACRO_TOPICS, overlap_dir

    m = pd.read_parquet(overlap_dir(out_dir) / MACRO_TOPICS)
    return m.set_index("macro_topic").name, m.set_index("macro_topic").topics.map(len)


# --- statistics --------------------------------------------------------------------------------------------

def wilson(k, n, z=1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def rate_row(k, n, k_rest, n_rest):
    from scipy.stats import chi2_contingency

    low, high = wilson(k, n)
    table = np.array([[k, n - k], [k_rest, n_rest - k_rest]])
    p = chi2_contingency(table)[1] if min(k + k_rest, n + n_rest - k - k_rest) > 0 else np.nan
    rate, rest = k / n, k_rest / n_rest
    return {"coins": n, "graduates": k, "rate": rate, "rate_low": low, "rate_high": high, "rate_rest": rest,
            "ratio_to_rest": rate / rest if rest else np.nan, "p_value": p}


def time_row(group, rest):
    """Median and quartiles of hours to graduation, share graduating within a minute, Mann-Whitney against the rest."""
    from scipy.stats import mannwhitneyu

    row = {"graduates": len(group), "median_hours": np.median(group), "q1_hours": np.quantile(group, 0.25),
           "q3_hours": np.quantile(group, 0.75), "within_1_min": float((group <= 1 / 60).mean()),
           "median_hours_rest": np.median(rest)}
    if len(group) >= MIN_GRADUATES and len(rest) >= MIN_GRADUATES:
        u, p = mannwhitneyu(group, rest, alternative="two-sided")
        row.update({"faster": 1 - 2 * u / (len(group) * len(rest)), "p_value": p})
    else:
        row.update({"faster": np.nan, "p_value": np.nan})
    return row


# --- by macro-topic ----------------------------------------------------------------------------------------

def macro_tables(df, names, sizes):
    rows = df[["mint", "macro_topics", "graduated", "hours", "in_horizon", "graduated_in_horizon"]].explode("macro_topics")
    rows = rows.rename(columns={"macro_topics": "macro_topic"})
    total_n, total_k = len(df), int(df.graduated.sum())
    counts = rows.groupby("macro_topic").agg(n=("mint", "size"), k=("graduated", "sum"))
    rates = []
    for m, (n, k) in counts.iterrows():
        rates.append({"macro_topic": m, **rate_row(int(k), int(n), total_k - int(k), total_n - int(n))})
    rates = pd.DataFrame(rates)
    rates["q_value"] = bh(rates.p_value)

    timed = df[df.in_horizon & df.graduated_in_horizon]
    hours_by_mint = timed.set_index("mint").hours
    member = rows[rows.mint.isin(hours_by_mint.index)]
    times = []
    for m, grp in member.groupby("macro_topic"):
        inside = hours_by_mint.index.isin(grp.mint)
        times.append({"macro_topic": m, **time_row(hours_by_mint.values[inside], hours_by_mint.values[~inside])})
    times = pd.DataFrame(times)
    times["q_value"] = bh(times.p_value)
    for t in (rates, times):
        t.insert(1, "name", t.macro_topic.map(names))
        t.insert(2, "topics", t.macro_topic.map(sizes))
    return rates, times


# --- by toxicity label -------------------------------------------------------------------------------------

def attribute_tables(out_dir, df, partial):
    """Weighted graduation rate given each label (``graduation_models.conditional``) and time to graduation of
    labelled against unlabelled graduates."""
    from crypto_analysis.graduation_models import HARMS, coin_frame as sample_frame, conditional

    sample, _, _ = sample_frame(out_dir, partial)
    rates = conditional(sample)
    timed = sample.merge(df[["mint", "hours", "in_horizon", "graduated_in_horizon"]], on="mint")
    timed = timed[timed.in_horizon & timed.graduated_in_horizon]
    times = []
    for attr in HARMS:
        times.append({"attribute": attr, **time_row(timed.hours[timed[attr] == 1].to_numpy(),
                                                     timed.hours[timed[attr] == 0].to_numpy())})
    times = pd.DataFrame(times)
    times["q_value"] = bh(times.p_value)
    return rates, times


# --- weekly series and peaks -------------------------------------------------------------------------------

def weekly(df):
    """Weekly creations (by launch week) and graduations (by graduation week) of every macro-topic, with the
    platform totals and each macro-topic's share of the week's creations."""
    rows = df[["macro_topics", "created_at", "grad_time"]].explode("macro_topics").rename(columns={"macro_topics": "macro_topic"})
    week = lambda s: pd.to_datetime(s, unit="s").dt.to_period("W-SUN").dt.start_time
    created = rows.assign(week=week(rows.created_at)).groupby(["macro_topic", "week"]).size().rename("creations")
    grads = rows.dropna(subset=["grad_time"])
    graduated = grads.assign(week=week(grads.grad_time)).groupby(["macro_topic", "week"]).size().rename("graduations")
    series = pd.concat([created, graduated], axis=1).fillna(0).astype(int).reset_index()
    weeks = pd.date_range(series.week.min(), series.week.max(), freq="7D")
    full = pd.MultiIndex.from_product([series.macro_topic.unique(), weeks], names=["macro_topic", "week"])
    series = series.set_index(["macro_topic", "week"]).reindex(full, fill_value=0).reset_index()
    totals = df.assign(week=week(df.created_at)).groupby("week").size().rename("all_creations")
    series = series.merge(totals, on="week", how="left").fillna({"all_creations": 0})
    series["share"] = series.creations / series.all_creations.replace(0, np.nan)
    return series


def peaks(series, df, names, out_dir):
    """Weeks whose creation share is at least ``PEAK_RATIO`` times the macro-topic's median weekly share, with at
    least ``PEAK_MIN`` creations; with the busiest day and the most common coin names of the macro-topic."""
    from crypto_analysis.datalake import DATA_DIR, connect

    median = series[series.creations > 0].groupby("macro_topic").share.median().rename("median_share")
    s = series.merge(median, on="macro_topic")
    s["ratio"] = s.share / s.median_share
    found = s[(s.ratio >= PEAK_RATIO) & (s.creations >= PEAK_MIN)].copy()
    if found.empty:
        return found
    con = connect(DATA_DIR, threads=8)
    rows = df[["mint", "macro_topics", "created_at"]].explode("macro_topics").rename(columns={"macro_topics": "macro_topic"})
    rows["week"] = pd.to_datetime(rows.created_at, unit="s").dt.to_period("W-SUN").dt.start_time
    hits = rows.merge(found[["macro_topic", "week"]], on=["macro_topic", "week"])
    con.register("hits", hits[["mint"]].drop_duplicates())
    meta = con.execute("SELECT m.mint, lower(trim(m.meta_name)) AS name FROM token_metadata m JOIN hits USING (mint)").df()
    hits = hits.merge(meta.drop_duplicates("mint"), on="mint", how="left")
    hits["day"] = pd.to_datetime(hits.created_at, unit="s").dt.date
    lead = hits.groupby(["macro_topic", "week"]).agg(
        busiest_day=("day", lambda d: d.value_counts().idxmax()),
        top_names=("name", lambda n: "; ".join(f"{k} ({v})" for k, v in n.value_counts().head(TOP_NAMES).items())))
    found = found.merge(lead.reset_index(), on=["macro_topic", "week"])
    found.insert(1, "name", found.macro_topic.map(names))
    return found.sort_values("ratio", ascending=False)


def episodes(found):
    """Consecutive peak weeks of one macro-topic as one episode."""
    p = found.sort_values(["macro_topic", "week"]).copy()
    p["episode"] = p.groupby("macro_topic").week.diff().dt.days.ne(7).cumsum()
    return p.groupby("episode").agg(
        macro_topic=("macro_topic", "first"), name=("name", "first"), start=("week", "min"), end=("week", "max"),
        weeks=("week", "size"), creations=("creations", "sum"), graduations=("graduations", "sum"),
        max_ratio=("ratio", "max"), top_names=("top_names", "first")).sort_values("creations", ascending=False)


# --- figures -----------------------------------------------------------------------------------------------

DISPLAY = {"Black Table": "Black Table (Ansem)"}


def display_name(name):
    """Macro-topic names for figures: slurs masked, opaque names explained."""
    return DISPLAY.get(name, name).replace("Nigga", "N-word")


def figures(out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from crypto_analysis.plotting import (COLORS, GOLD, count_label, forest, format_date_axis, pct_label,
                                          plain_text, save_figure, set_latex_style, thousands_axis)

    out = out_dir_of(out_dir)
    set_latex_style()
    rates = pd.read_parquet(out / "rates.parquet")
    series = pd.read_parquet(out / "weekly.parquet")
    coins = pd.read_parquet(out / "coins.parquet", columns=["graduated"])
    overall = coins.graduated.mean()

    # graduation rate: the 15 highest and 15 lowest macro-topics with at least 10,000 coins
    big = rates[rates.coins >= 10_000].sort_values("rate", ascending=False)
    shown = pd.concat([big.head(15), big.tail(15)])
    with plain_text():
        fig, ax = plt.subplots(figsize=(6, 6.4))
        forest(ax, [f"{display_name(n)} ({c / 1000:.0f}k)" for n, c in zip(shown.name, shown.coins)],
               [{"name": "graduation rate", "color": GOLD, "rate": shown.rate.to_numpy(),
                 "lo": shown.rate_low.to_numpy(), "hi": shown.rate_high.to_numpy()}],
               ref=overall, ref_label=f"all coins {overall:.2%}")
        ax.axhline(14.5, color=COLORS["muted"], lw=0.6)
        ax.set_xlabel(pct_label("Graduation rate"))
        ax.set_title("Highest and lowest graduation rates, macro-topics with 10,000+ coins", fontsize=8, loc="left",
                     pad=14)
        fig.tight_layout()
        save_figure(fig, FIGURES / "graduation_rate_by_macro_topic")
        plt.close(fig)

    # weekly creations and graduations of the news-driven macro-topics, one panel each, event marked
    names, _ = macro_names(out_dir)
    by_name = {v: k for k, v in names.items()}
    for column, color, label, stem in (("creations", COLORS["blue"], "coins launched", "weekly_creations"),
                                       ("graduations", GOLD, "graduations", "weekly_graduations")):
        with plain_text():
            fig, axes = plt.subplots(4, 3, figsize=(6.4, 6.6), sharex=True)
            for ax, (name, date, event, _) in zip(axes.flat, EVENTS):
                s = series[series.macro_topic == by_name[name]]
                ax.plot(s.week, s[column], color=color, lw=1.2, solid_capstyle="round")
                when = pd.Timestamp(date)
                ax.axvline(when, color=COLORS["ink"], lw=0.5, ls=(0, (1, 2)), zorder=0.5)
                ax.set_title(display_name(name), fontsize=7, loc="left", pad=10)
                ax.text(0, 1.02, f"{when:%b} {when.day}: {event} (dotted)", transform=ax.transAxes, fontsize=5.8,
                        va="bottom", color=COLORS["ink_secondary"])
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                thousands_axis(ax)
                ax.tick_params(labelsize=6)
                format_date_axis(ax)
            for ax in axes[:, 0]:
                ax.set_ylabel(count_label(label), fontsize=7)
            fig.suptitle(f"Weekly {label} of the macro-topics with the largest news-driven peaks", fontsize=8,
                         x=0.01, ha="left")
            fig.tight_layout()
            save_figure(fig, FIGURES / stem)
            plt.close(fig)
    print(f"figures in {FIGURES}/", flush=True)


# --- run ---------------------------------------------------------------------------------------------------

def run(out_dir, partial=False):
    out = out_dir_of(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = coin_frame(out_dir)
    names, sizes = macro_names(out_dir)
    print(f"{len(df):,} coins, {int(df.graduated.sum()):,} graduated; time comparison: "
          f"{int((df.in_horizon & df.graduated_in_horizon).sum()):,} graduates launched before the last "
          f"{HORIZON_DAYS} days and graduated within {HORIZON_DAYS} days", flush=True)
    rates, times = macro_tables(df, names, sizes)
    series = weekly(df)
    found = peaks(series, df, names, out_dir)
    attr_rates, attr_times = attribute_tables(out_dir, df, partial)
    suffix = "_partial" if partial else ""
    df.drop(columns="macro_topics").to_parquet(out / "coins.parquet", index=False)
    rates.to_parquet(out / "rates.parquet", index=False)
    times.to_parquet(out / "times.parquet", index=False)
    series.to_parquet(out / "weekly.parquet", index=False)
    found.to_parquet(out / "peaks.parquet", index=False)
    episodes(found).to_parquet(out / "episodes.parquet", index=False)
    attr_rates.to_parquet(out / f"attribute_rates{suffix}.parquet", index=False)
    attr_times.to_parquet(out / f"attribute_times{suffix}.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    big = rates[rates.coins >= 10_000]
    cols = ["name", "topics", "coins", "graduates", "rate", "rate_low", "rate_high", "ratio_to_rest", "q_value"]
    print(f"\ngraduation rate by macro-topic (overall {df.graduated.mean():.4f}); {len(big)} macro-topics with "
          f"10,000+ coins, highest and lowest")
    print(big.sort_values("rate", ascending=False)[cols].head(15).round(4).to_string(index=False))
    print(big.sort_values("rate")[cols].head(10).round(4).to_string(index=False))
    t = times[times.graduates >= MIN_GRADUATES]
    cols = ["name", "graduates", "median_hours", "q1_hours", "q3_hours", "within_1_min", "faster", "q_value"]
    print(f"\ntime to graduation, one against all (median overall "
          f"{df[df.in_horizon & df.graduated_in_horizon].hours.median():.3f} h); significant at q < 0.05:")
    print(t[t.q_value < 0.05].sort_values("faster", ascending=False)[cols].round(3).to_string(index=False))
    print(f"\ntoxicity labels{' (partial)' if partial else ''}: rates")
    print(attr_rates.round(5).to_string(index=False))
    print("toxicity labels: time to graduation")
    print(attr_times.round(4).to_string(index=False))
    print(f"\npeaks ({len(found)}):")
    print(found[["name", "week", "creations", "share", "median_share", "ratio", "graduations", "busiest_day",
                 "top_names"]].head(30).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="toxicity labels: use the contents labelled so far")
    parser.add_argument("--figures", action="store_true", help="draw the figures from the saved tables")
    args = parser.parse_args()
    if args.figures:
        figures(args.out_dir)
    else:
        run(args.out_dir, args.partial)


if __name__ == "__main__":
    main()
