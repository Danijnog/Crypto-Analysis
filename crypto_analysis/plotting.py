"""Shared LaTeX-style matplotlib configuration for the project's notebooks.

Usage in a notebook::

    from crypto_analysis.plotting import FIG_4x3, FIG_6x4, COLORS, save_figure, set_latex_style

    set_latex_style()
    fig, ax = plt.subplots(figsize=FIG_6x4)

Figures are sized in inches for a paper: ``FIG_4x3`` fits a single column or a subfigure,
``FIG_6x4`` fits the full text width. Fonts are set in points so text in the saved PDF has the
same size as the document text when included at 100% scale.

When a TeX installation (``latex``, ``dvipng``, ``kpsewhich``) is on the PATH, text is typeset by
LaTeX itself (``text.usetex``). Otherwise matplotlib's bundled Computer Modern fonts and mathtext
are used, which look the same for ordinary labels and need no system packages.
"""

from contextlib import contextmanager
import logging
from pathlib import Path
import re
import shutil

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

FIG_4x3 = (4, 3)
FIG_6x4 = (6, 4)

COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "grid": "#e4e3df",
}
# Fixed order for categorical series: always assign in this order, never cycle past it.
CATEGORICAL = [COLORS["blue"], COLORS["orange"], COLORS["aqua"]]

_BASE_RC = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["ink_secondary"],
    "axes.linewidth": 0.6,
    "axes.labelcolor": COLORS["ink"],
    "axes.titlecolor": COLORS["ink"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.5,
    "xtick.color": COLORS["ink_secondary"],
    "ytick.color": COLORS["ink_secondary"],
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
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


def latex_available():
    """True when the external tools needed for ``text.usetex`` are installed."""
    return all(shutil.which(tool) for tool in ("latex", "dvipng", "kpsewhich"))


def set_latex_style(use_tex=None):
    """Apply the project's LaTeX-like style globally.

    ``use_tex=None`` uses real LaTeX when it is installed and falls back to Computer Modern
    mathtext otherwise. Returns whether real LaTeX is in use.
    """
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


@contextmanager
def plain_text():
    """Temporarily disable LaTeX for figures with arbitrary user text (token names, emoji, CJK).

    Real LaTeX would fail on most of those strings, and Computer Modern lacks their glyphs.
    The font is named directly rather than via the generic "serif" family: text objects keep
    the family name they were created with, and a generic name would be resolved again at
    display time, after this context has exited. Escape ``$`` in captions with
    ``plain_caption`` so a name like "$A $B" is not parsed as math.
    """
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Serif"}):
        yield


def plain_caption(text):
    """Free text for use inside ``plain_text()``: keeps ``$`` literal."""
    return str(text).replace("$", r"\$")


def format_date_axis(ax):
    """Compact month labels for a daily time series on the x axis."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def mark_events(ax, dates, codes, min_gap_days=12, levels=3):
    """Mark checkpoints on a time-series axis: a thin vertical line per date and a short code above
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
        ax.axvline(date, color=COLORS["ink_secondary"], lw=0.5, ls=(0, (2, 2)), zorder=0.5)
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
