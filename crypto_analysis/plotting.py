"""Shared figure design for the project's notebooks: LaTeX-like style, palette and chart helpers.

Usage in a notebook::

    from crypto_analysis.plotting import FIG_4x3, FIG_6x4, set_latex_style, forest, count_label

    set_latex_style()
    fig, ax = plt.subplots(figsize=FIG_6x4)

**Sizes.** ``FIG_4x3`` fits a single column or a subfigure, ``FIG_6x4`` the full text width; fonts
are in points, so text in the saved PDF matches the document at 100% scale.

**Fonts.** With a TeX installation (``latex``, ``dvipng``, ``kpsewhich``) text is typeset by LaTeX
(``text.usetex``); otherwise matplotlib's bundled Computer Modern fonts and mathtext are used.

**Conventions.** Light background only. Count axes read ``# of X`` (``count_label``), percentage
axes ``Y (%)`` with plain numbers on the ticks (``pct_label`` + ``percent_axis``).

**Colour by meaning.** Replication roles keep one colour everywhere (``ROLE_COLORS``): blue =
original, red = self-relaunch (farms), aqua = copy by another creator, gold = copy of a graduate.
Gold also marks "graduated" in every other figure; grey is context. The four hues pass the
dataviz palette validator for all pairs (CVD-safe); aqua and gold sit below 3:1 contrast on white,
so they are always direct-labelled.

**Chart helpers** (one per form): ``forest`` (rates with intervals; build series with
``rate_series``, or ``forest_by_row`` for one coloured estimate per row), ``dumbbell`` (before/after),
``raincloud`` (distributions), ``ridgeline`` (many distributions on one axis), ``density_area``,
``ccdf``, ``streamgraph``, ``waffle``, ``image_treemap`` and ``radial_clock``.
"""

from contextlib import contextmanager
import logging
from pathlib import Path
import re
import shutil

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.transforms as mtransforms
import numpy as np

FIG_4x3 = (4, 3)
FIG_6x4 = (6, 4)

COLORS = {
    "blue": "#2a78d6",
    "red": "#e34948",
    "aqua": "#1baf7a",
    "gold": "#eda100",
    "orange": "#eb6834",
    "ink": "#1d1d1b",
    "ink_secondary": "#6b6a65",
    "muted": "#b9b7ae",
    "grid": "#ebe9e2",
}
GOLD = COLORS["gold"]
ROLE_COLORS = {
    "original": COLORS["blue"],
    "self-relaunch": COLORS["red"],
    "copy (other creator)": COLORS["aqua"],
    "copy of a graduate": COLORS["gold"],
}
ROLE_ORDER = list(ROLE_COLORS)
# Fixed order for categorical series that are not replication roles; never cycle past it.
CATEGORICAL = [COLORS["blue"], COLORS["aqua"], COLORS["gold"], COLORS["red"]]
# Sequential ramps (one hue, light -> dark): gold for graduation-related magnitudes, red for farms.
GOLD_CMAP = LinearSegmentedColormap.from_list("gold", ["#fdf8ec", "#f5d27a", GOLD, "#a86b00", "#5c3b00"])
RED_RAMP = ["#f4b3b2", "#ea7c7b", COLORS["red"], "#9e2524"]
BLUE_RAMP = ["#c6dbf3", "#8cb6e8", "#5b97df", COLORS["blue"], "#1d5aa5"]

_BASE_RC = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 10,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10,
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["muted"],
    "axes.linewidth": 0.6,
    "axes.labelcolor": COLORS["ink"],
    "axes.titlecolor": COLORS["ink"],
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.5,
    "xtick.color": COLORS["ink_secondary"],
    "ytick.color": COLORS["ink_secondary"],
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.1,
    "lines.markersize": 3,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed TrueType so the PDF text stays editable/searchable
    "ps.fonttype": 42,
}

_TEX_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
}

_MATHTEXT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,  # cmr10 has no U+2212 glyph
}

_TEX_SPECIALS = re.compile(r"([&%$#_{}])")
# Darker text shades for series colours that are too light for text on white
_TEXT_SHADES = {COLORS["gold"]: "#a86b00", COLORS["muted"]: COLORS["ink_secondary"], COLORS["aqua"]: "#12805a",
                "#f4b3b2": "#c0504f", "#f5d27a": "#a86b00"}


def text_color(color):
    """Colour to write a label in for a series drawn in ``color`` (light hues get a darker shade)."""
    return _TEXT_SHADES.get(color, color)
_KDE_MAX_POINTS = 200_000


# --------------------------------------------------------------------------------------------
# Style, text and saving
# --------------------------------------------------------------------------------------------

def latex_available():
    """True when the external tools needed for ``text.usetex`` are installed."""
    return all(shutil.which(tool) for tool in ("latex", "dvipng", "kpsewhich"))


def set_latex_style(use_tex=None):
    """Apply the project's style globally. ``use_tex=None`` uses real LaTeX when installed and
    Computer Modern mathtext otherwise. Returns whether real LaTeX is in use."""
    if use_tex is None:
        use_tex = latex_available()
    mpl.rcParams.update(_BASE_RC)
    mpl.rcParams.update(_TEX_RC if use_tex else _MATHTEXT_RC)
    return use_tex


def tex_escape(text):
    """Make free text safe for the LaTeX style.

    Under real LaTeX, special characters (``& % $ # _ { }``) are escaped. In both modes ``<`` and
    ``>`` go to math mode, because Computer Modern text fonts map them to ``¡`` and ``¿``.
    """
    text = str(text)
    if mpl.rcParams["text.usetex"]:
        text = _TEX_SPECIALS.sub(r"\\\1", text)
    return text.replace("<", "$<$").replace(">", "$>$")


def count_label(what):
    """Axis label for a count: ``# of <what>``."""
    return tex_escape(f"# of {what}")


def pct_label(what):
    """Axis label for a percentage: ``<what> (%)``; pair it with ``percent_axis``."""
    return tex_escape(f"{what} (%)")


def percent_axis(ax, axis="y", decimals=None):
    """Show fractions (0-1) on ``axis`` as plain percentage numbers (the "%" is in the label)."""
    fmt = PercentFormatter(1, decimals=decimals, symbol="")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands_axis(ax, axis="y"):
    """Thousands separators on a count axis."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: tex_escape(f"{v:,.0f}"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def log_ticks(ax, axis, values, labels=None):
    """Explicit ticks on a log axis (labels default to compact numbers)."""
    labels = labels or [_compact(v) for v in values]
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    setter(values, [tex_escape(lbl) for lbl in labels])
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(mpl.ticker.NullLocator())


def _compact(v):
    for unit, div in (("M", 1e6), ("k", 1e3)):
        if abs(v) >= div:
            return f"{v / div:g}{unit}"
    return f"{v:g}"


@contextmanager
def plain_text():
    """Temporarily disable LaTeX for figures with arbitrary user text (token names, emoji, CJK).

    The font is named directly rather than via the generic "serif" family: text objects keep
    the family name they were created with, and a generic name would be resolved again at
    display time, after this context has exited. Escape ``$`` in captions with ``plain_caption``.
    """
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Serif"}):
        yield


def plain_caption(text):
    """Free text for use inside ``plain_text()``: keeps ``$`` literal."""
    return str(text).replace("$", r"\$")


def ascii_label(text, width=18):
    """Short ASCII, LaTeX-safe version of a token name (names hold emoji and other scripts)."""
    clean = re.sub(r"[^\x20-\x7e]", "", str(text)).strip() or "(no ASCII name)"
    return tex_escape(clean[:width])


def format_date_axis(ax):
    """Compact month labels for a daily time series on the x axis."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def mark_events(ax, dates, codes, min_gap_days=12, levels=3):
    """Mark checkpoints on a time-series axis: a thin dotted line per date and a short code above
    the plot area. Codes that would collide (closer than ``min_gap_days``) are stacked on up to
    ``levels`` rows. Explain the codes in a table or the figure caption, not on the plot.
    """
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    last_on_level = [-np.inf] * levels
    for date, code in sorted(zip(dates, codes), key=lambda e: mdates.date2num(e[0])):
        x = mdates.date2num(date)
        free = [lvl for lvl in range(levels) if x - last_on_level[lvl] >= min_gap_days]
        level = free[0] if free else int(np.argmin(last_on_level))
        last_on_level[level] = x
        ax.axvline(date, color=COLORS["ink"], lw=0.4, ls=(0, (1, 2)), zorder=0.5)
        if code:
            ax.text(date, 1.01 + 0.06 * level, tex_escape(code), transform=trans, ha="center",
                    va="bottom", fontsize=6.5, color=COLORS["ink_secondary"], clip_on=False)


def save_figure(fig, path, formats=("pdf", "png")):
    """Save ``fig`` as ``path`` with each extension in ``formats`` (PDF for LaTeX, PNG for preview)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fontTools warns about the bundled cmr10 font's zeroed timestamps when embedding it; harmless.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"))


# --------------------------------------------------------------------------------------------
# Chart helpers
# --------------------------------------------------------------------------------------------

def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion ``k / n`` (scalars or arrays)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def rate_series(name, color, k, n):
    """A ``forest`` series from success counts ``k`` out of ``n`` (aligned arrays, NaN allowed)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    lo, hi = wilson_interval(k, n)
    return {"name": name, "color": color, "rate": k / n, "lo": lo, "hi": hi}


def _kde(values, grid, bw=0.12):
    from scipy.stats import gaussian_kde  # scipy is only needed for density-based charts

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) > _KDE_MAX_POINTS:
        v = np.random.default_rng(0).choice(v, _KDE_MAX_POINTS, replace=False)
    if len(v) < 2 or np.ptp(v) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(v, bw_method=bw)(grid)


def forest(ax, rows, series, *, value_fmt="{:.2%}", annotate=None, ref=None, ref_label=None,
           log=False, spacing=0.2):
    """Dot-and-interval plot of rates (one row per category, one coloured series per group).

    ``rows``: row labels, top to bottom. ``series``: list of dicts with ``name``, ``color`` and
    arrays ``rate``, ``lo``, ``hi`` aligned with ``rows`` (NaN = no estimate). With one series the
    value is printed next to each interval (``annotate`` defaults to that); with several, each
    series is labelled once, in its own colour, on the first row where it appears.
    The caller sets the x label (use ``pct_label``).
    """
    n = len(series)
    annotate = (n == 1) if annotate is None else annotate
    for j, s in enumerate(series):
        offset = (j - (n - 1) / 2) * spacing
        labelled = False
        for i in range(len(rows)):
            r, lo, hi = s["rate"][i], s["lo"][i], s["hi"][i]
            if not np.isfinite(r):
                continue
            y = i + offset
            ax.plot([lo, hi], [y, y], color=s["color"], lw=1.6, solid_capstyle="round", alpha=0.9)
            ax.plot(r, y, "o", ms=4.8, color=s["color"], mec="white", mew=0.7, zorder=3)
            if annotate:
                ax.text(hi, y, "  " + tex_escape(value_fmt.format(r)), va="center", fontsize=6.5,
                        color=COLORS["ink_secondary"])
            elif n > 1 and not labelled:
                ax.text(hi, y, "  " + tex_escape(s["name"]), va="center", fontsize=6.5,
                        color=text_color(s["color"]))
                labelled = True
    if ref is not None:
        ax.axvline(ref, color=COLORS["ink_secondary"], lw=0.6, ls=(0, (3, 2)), zorder=0)
        if ref_label:
            ax.text(ref, -0.6, " " + tex_escape(ref_label), fontsize=6.5, color=COLORS["ink_secondary"],
                    va="bottom")
    ax.set_yticks(range(len(rows)), [tex_escape(r) for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.7)
    if log:
        ax.set_xscale("log")
    ax.margins(x=0.12)
    ax.grid(axis="x", color=COLORS["grid"])
    ax.grid(axis="y", visible=False)
    percent_axis(ax, "x")


def forest_by_row(ax, rows, k, n, colors, **kwargs):
    """``forest`` with one estimate per row, each in its own colour (e.g. one row per role).
    ``k``/``n`` are success counts and totals aligned with ``rows``."""
    series = []
    for i, color in enumerate(colors):
        ki, ni = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        ki[i], ni[i] = np.asarray(k, dtype=float)[i], np.asarray(n, dtype=float)[i]
        series.append(rate_series(str(rows[i]), color, ki, ni))
    kwargs.setdefault("annotate", True)
    forest(ax, rows, series, spacing=0, **kwargs)


def dumbbell(ax, rows, left, right, *, left_label, right_label, left_color=None, right_color=None,
             value_fmt="{:.2%}"):
    """Before/after (or A vs B) per row: two dots joined by a grey bar, labelled once at the top."""
    left_color = left_color or COLORS["muted"]
    right_color = right_color or GOLD
    ink2 = COLORS["ink_secondary"]
    for i, (a, b) in enumerate(zip(left, right)):
        ax.plot([a, b], [i, i], color=COLORS["grid"], lw=3.2, solid_capstyle="round", zorder=1)
        ax.plot(a, i, "o", ms=6, color=left_color, mec="white", mew=0.7, zorder=3)
        ax.plot(b, i, "o", ms=6, color=right_color, mec="white", mew=0.7, zorder=3)
        lo, hi = min(a, b), max(a, b)
        ax.text(hi, i, "  " + tex_escape(value_fmt.format(hi)), va="center", fontsize=6.5, color=ink2)
        ax.text(lo, i, tex_escape(value_fmt.format(lo)) + "  ", va="center", ha="right", fontsize=6.5,
                color=ink2)
    for label, color in ((left_label, left_color), (right_label, right_color)):
        ax.plot([], [], "o", ms=5.5, color=color, mec="white", mew=0.7, label=tex_escape(label))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.8, handletextpad=0.3,
              borderaxespad=0.2)
    ax.set_yticks(range(len(rows)), [tex_escape(r) for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.margins(x=0.18)
    ax.grid(axis="x", color=COLORS["grid"])
    ax.grid(axis="y", visible=False)
    percent_axis(ax, "x")


def raincloud(ax, groups, *, log10=False, bw=0.15, max_points=1500, seed=0, summary_fmt=None):
    """Horizontal raincloud per group: density (cloud) above, jittered points (rain) and an IQR bar
    with the median below. ``groups``: list of ``(label, values, color)``, drawn bottom to top.
    With ``log10=True`` values are log-transformed (the caller sets ticks, e.g. ``log_ticks``)."""
    rng = np.random.default_rng(seed)
    transformed = []
    for _, values, _ in groups:
        v = np.asarray(values, dtype=float)
        v = np.log10(v[v > 0]) if log10 else v[np.isfinite(v)]
        transformed.append(v)
    lo = min(np.percentile(v, 0.5) for v in transformed if len(v))
    hi = max(np.percentile(v, 99.5) for v in transformed if len(v))
    pad = (hi - lo) * 0.05
    grid = np.linspace(lo - pad, hi + pad, 400)
    for i, ((label, _, color), v) in enumerate(zip(groups, transformed)):
        if not len(v):
            continue
        dens = _kde(v, grid, bw)
        ax.fill_between(grid, i + 0.06, i + 0.06 + dens / dens.max() * 0.55, color=color, alpha=0.85, lw=0)
        sample = rng.choice(v, size=min(max_points, len(v)), replace=False)
        ax.scatter(sample, i - 0.13 + rng.uniform(-0.08, 0.08, len(sample)), s=1.1, color=color, alpha=0.35, lw=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i - 0.13, i - 0.13], color=COLORS["ink"], lw=2.2, solid_capstyle="round")
        ax.plot(med, i - 0.13, "o", ms=4.2, color="white", mec=COLORS["ink"], mew=0.9, zorder=5)
        if summary_fmt:
            shown = 10 ** med if log10 else med
            ax.text(med, i - 0.3, tex_escape(summary_fmt.format(shown)), fontsize=6.5, ha="center", va="top",
                    color=COLORS["ink"])
    ax.set_yticks(range(len(groups)), [tex_escape(g[0]) for g in groups])
    ax.set_ylim(-0.55, len(groups) - 0.25)
    ax.grid(axis="x", color=COLORS["grid"])
    ax.grid(axis="y", visible=False)


def ridgeline(ax, rows, grid, *, split=None, colors=None, overlap=1.7, bw=0.07, label_width=16):
    """Joy-plot of one density per row, top to bottom, lower ridges drawn in front.

    ``rows``: list of ``(label, values)`` already on the axis scale of ``grid``. With ``split``,
    the area left of it is drawn in ``colors[0]`` (default grey) and right of it in ``colors[1]``
    (default gold). Each ridge is scaled to its own maximum.
    """
    before, after = colors or (COLORS["muted"], GOLD)
    for i, (label, values) in enumerate(rows):
        v = np.asarray(values, dtype=float)
        v = v[(v >= grid[0]) & (v <= grid[-1])]
        dens = _kde(v, grid, bw)
        if dens.max() > 0:
            dens = dens / dens.max() * overlap
        base, z = -i * 0.9, 2 * i
        if split is None:
            ax.fill_between(grid, base, base + dens, color=after, lw=0, zorder=z)
        else:
            ax.fill_between(grid, base, base + dens, where=grid <= split, color=before, lw=0, zorder=z)
            ax.fill_between(grid, base, base + dens, where=grid >= split, color=after, lw=0, zorder=z)
        ax.plot(grid, base + dens, color="white", lw=1.4, zorder=z + 1)
        ax.plot(grid, base + dens, color=COLORS["ink"], lw=0.45, zorder=z + 1)
        ax.text(grid[0] - (grid[-1] - grid[0]) * 0.01, base + 0.05, ascii_label(label, label_width),
                fontsize=6, color=COLORS["ink_secondary"], ha="right", va="bottom")
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(grid[0], grid[-1])


def density_area(ax, values, *, color=None, log10=False, bw=0.08, label=None, alpha=0.85, median=True,
                 grid=None):
    """Filled density curve (lighter than a histogram) with an optional median tick."""
    color = color or COLORS["blue"]
    v = np.asarray(values, dtype=float)
    v = np.log10(v[v > 0]) if log10 else v[np.isfinite(v)]
    if grid is None:
        lo, hi = np.percentile(v, [0.1, 99.9])
        pad = (hi - lo) * 0.08
        grid = np.linspace(lo - pad, hi + pad, 500)
    dens = _kde(v, grid, bw)
    ax.fill_between(grid, 0, dens, color=color, alpha=alpha, lw=0, label=label)
    ax.plot(grid, dens, color="white", lw=0.8)
    if median:
        m = np.median(v)
        ax.plot([m, m], [0, np.interp(m, grid, dens)], color=COLORS["ink"], lw=0.7)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    return grid, dens


def ccdf(ax, values, *, color=None, label=None, lw=1.3):
    """Complementary CDF on log-log axes: share of items with value >= x (y in %, plain numbers)."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v > 0]
    uniq, first = np.unique(v, return_index=True)
    share = 1 - first / len(v)
    ax.step(uniq, share, where="post", color=color or COLORS["blue"], lw=lw, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: tex_escape(f"{y * 100:g}")))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: tex_escape(_compact(x))))
    ax.grid(axis="both", color=COLORS["grid"])


def streamgraph(ax, x, table, colors, *, label_at=0.62, dark_text=("copy of a graduate",)):
    """Symmetric stacked area of ``table`` (rows = x, columns = series, in stacking order), each
    stream labelled in place at ``label_at`` (fraction of the x range)."""
    vals = table.to_numpy(dtype=float)
    ax.stackplot(x, vals.T, colors=colors, baseline="sym", alpha=0.95, linewidth=0.4, edgecolor="white")
    bottom = -vals.sum(axis=1) / 2 + np.c_[np.zeros(len(x)), np.cumsum(vals, axis=1)[:, :-1]].T
    k = int(len(x) * label_at)
    total = vals[k].sum()
    for j, name in enumerate(table.columns):
        if vals[k, j] <= 0:
            continue
        if vals[k, j] < 0.07 * total:  # too thin to hold its label: put it just outside the stack
            outer = bottom[j, k] + vals[k, j] if j == len(table.columns) - 1 else bottom[j, k]
            ax.annotate(tex_escape(name), (x[k], outer), xytext=(0, 9), textcoords="offset points",
                        fontsize=6.8, ha="center", va="bottom", color=COLORS["ink"],
                        arrowprops={"arrowstyle": "-", "color": COLORS["ink"], "lw": 0.5})
            continue
        ax.text(x[k], bottom[j, k] + vals[k, j] / 2, tex_escape(name), fontsize=6.8, ha="center", va="center",
                color=COLORS["ink"] if name in dark_text else "white")
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    top = vals.sum(axis=1).max() / 2
    ax.set_ylim(-top * 1.05, top * 1.3)
    return top


def waffle(ax, shares, colors, *, n_cols=10, n_rows=10, value_fmt="{:.1%}", legend_title=None):
    """Unit chart: ``n_cols * n_rows`` squares split by ``shares`` (dict label -> share, in order),
    filled column by column, with a vertical legend (label and value) to the right."""
    total = n_cols * n_rows
    counts = np.round(np.array(list(shares.values())) / sum(shares.values()) * total).astype(int)
    counts[np.argmax(counts)] += total - counts.sum()
    cells = np.repeat(np.arange(len(counts)), counts)
    for idx, k in enumerate(cells):
        col, row = divmod(idx, n_rows)
        ax.add_patch(Rectangle((col, n_rows - 1 - row), 0.88, 0.88, color=colors[k], lw=0))
    x0 = n_cols + 0.8
    y = n_rows - 0.6
    if legend_title:
        ax.text(x0, y + 0.3, tex_escape(legend_title), fontsize=6.8, color=COLORS["ink_secondary"], va="bottom")
    for (name, share), color in zip(shares.items(), colors):
        y -= 1.25
        ax.add_patch(Rectangle((x0, y), 0.7, 0.7, color=color, lw=0))
        ax.text(x0 + 1.0, y + 0.35, tex_escape(f"{name}   {value_fmt.format(share)}"), fontsize=6.8, va="center",
                color=COLORS["ink"])
    ax.set_xlim(-0.2, n_cols + 9)
    ax.set_ylim(-0.2, n_rows + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def image_treemap(ax, images, shas, sizes, *, badges=None, gold=None, width=6.0, height=4.0,
                  min_badge_width=0.42):
    """Tile thumbnails with area proportional to ``sizes`` (squarified treemap).

    ``images``: {sha256: PIL image}; ``badges``: optional text per tile; ``gold``: optional bool
    per tile, drawn as a gold badge (e.g. the image graduated) instead of a dark one.
    """
    import squarify  # only needed for treemaps

    rects = squarify.squarify(squarify.normalize_sizes(np.asarray(sizes, dtype=float), width, height),
                              0, 0, width, height)
    for k, (r, sha) in enumerate(zip(rects, shas)):
        x, y, w, h = r["x"], r["y"], r["dx"], r["dy"]
        if sha in images:
            img = images[sha].convert("RGB")
            iw, ih = img.size
            target = w / h  # centre-crop to the tile's aspect ratio
            if iw / ih > target:
                cw = max(1, int(ih * target))
                img = img.crop(((iw - cw) // 2, 0, (iw - cw) // 2 + cw, ih))
            else:
                ch = max(1, int(iw / target))
                img = img.crop((0, (ih - ch) // 2, iw, (ih - ch) // 2 + ch))
            ax.imshow(img, extent=(x, x + w, y + h, y), interpolation="lanczos")
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.8, ec="white"))
        if badges is not None and w > min_badge_width:
            is_gold = bool(gold[k]) if gold is not None else False
            ax.text(x + 0.04, y + h - 0.04, tex_escape(badges[k]), fontsize=5.2, va="bottom", ha="left",
                    color=COLORS["ink"] if is_gold else "white",
                    bbox={"boxstyle": "round,pad=0.18", "ec": "none", "alpha": 0.9,
                          "fc": GOLD if is_gold else COLORS["ink"]})
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def alluvial(ax, flows, left_order, right_order, *, left_labels=None, right_labels=None, color=None,
             gap=0.012, min_label_share=0.02):
    """Two-column alluvial (Sankey-style) diagram: ribbons from left to right categories.

    ``flows``: DataFrame with columns ``left``, ``right``, ``value``. Categories are stacked top to
    bottom in ``left_order`` / ``right_order`` with heights proportional to their totals; ribbons are
    smooth bands, grey unless ``color`` maps a left category to a colour.
    """
    total = flows.value.sum()
    lt = flows.groupby("left").value.sum().reindex(left_order).fillna(0) / total
    rt = flows.groupby("right").value.sum().reindex(right_order).fillna(0) / total

    def stack(totals):
        avail = 1 - gap * (len(totals) - 1)
        tops, y = {}, 1.0
        for k, v in totals.items():
            tops[k] = y
            y -= v * avail + gap
        return tops, avail

    ltop, lavail = stack(lt)
    rtop, ravail = stack(rt)
    lcursor, rcursor = dict(ltop), dict(rtop)
    x0, x1, w = 0.0, 1.0, 0.035
    order = flows.assign(li=flows.left.map({k: i for i, k in enumerate(left_order)}),
                         ri=flows.right.map({k: i for i, k in enumerate(right_order)})).sort_values(["li", "ri"])
    for r in order.itertuples():
        h_l, h_r = r.value / total * lavail, r.value / total * ravail
        yl, yr = lcursor[r.left], rcursor[r.right]
        lcursor[r.left] -= h_l
        rcursor[r.right] -= h_r
        xa, xb = x0 + w, x1 - w
        xm = (xa + xb) / 2
        verts = [(xa, yl), (xm, yl), (xm, yr), (xb, yr), (xb, yr - h_r), (xm, yr - h_r), (xm, yl - h_l),
                 (xa, yl - h_l), (xa, yl)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        c = (color or {}).get(r.left, COLORS["muted"])
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=c, edgecolor="none", alpha=0.45))
    for tops, totals, avail, x, labels, ha in ((ltop, lt, lavail, x0, left_labels, "right"),
                                               (rtop, rt, ravail, x1 - w, right_labels, "left")):
        for k, v in totals.items():
            h = v * avail
            ax.add_patch(Rectangle((x, tops[k] - h), w, h, color=COLORS["ink"], lw=0))
            if v >= min_label_share:
                tx = x - 0.01 if ha == "right" else x + w + 0.01
                ax.text(tx, tops[k] - h / 2, tex_escape((labels or {}).get(k, k)), ha=ha, va="center", fontsize=6.3,
                        color=COLORS["ink"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")


def radial_clock(ax, shares, *, color=None, title=None):
    """24-hour rose on a polar axis: ``shares`` has one value per hour (0-23), midnight at the top."""
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ax.bar(theta, shares, width=2 * np.pi / 24 * 0.9, color=color or COLORS["blue"], alpha=0.88,
           edgecolor="white", lw=0.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta[::3], [f"{h:02d}h" for h in range(0, 24, 3)], fontsize=6.5)
    ax.set_yticks([])
    ax.spines["polar"].set_color(COLORS["grid"])
    ax.grid(color=COLORS["grid"], lw=0.4)
    if title:
        ax.set_title(tex_escape(title), fontsize=8.5, pad=12, loc="center")
