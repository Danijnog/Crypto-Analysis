"""Show stored token images in matplotlib grids.

Blobs are read straight from the image shards: the wanted hashes are grouped by shard (via the
``token_images`` ledger) and only those rows are read, so a grid never scans the whole store.
All functions take a DuckDB connection from ``crypto_analysis.datalake.connect``.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from crypto_analysis.datalake import DATA_DIR
from crypto_analysis.plotting import COLORS, plain_caption, plain_text


def load_images(con, shas, images_dir=DATA_DIR / "images"):
    """Return {sha256: PIL.Image} for the given hashes, reading only the shards that hold them."""
    wanted = pd.DataFrame({"sha256": pd.unique(pd.Series(list(shas), dtype=object))})
    con.register("gallery_wanted", wanted)
    locs = con.execute("""
        SELECT DISTINCT w.sha256, i.shard
        FROM gallery_wanted w JOIN token_images i USING (sha256)
        WHERE i.status = 'ok'
    """).df()
    con.unregister("gallery_wanted")
    out = {}
    for shard, grp in locs.groupby("shard"):
        con.register("gallery_shard", grp[["sha256"]])
        rows = con.execute(f"""
            SELECT b.sha256, b.bytes
            FROM read_parquet('{images_dir / shard}') b
            JOIN gallery_shard USING (sha256)
        """).df()
        con.unregister("gallery_shard")
        for sha, raw in zip(rows.sha256, rows.bytes):
            out[sha] = Image.open(io.BytesIO(raw))
    return out


def names_for(con, shas):
    """{sha256: most common meta_name among the tokens using that image}."""
    wanted = pd.DataFrame({"sha256": pd.unique(pd.Series(list(shas), dtype=object))})
    con.register("names_wanted", wanted)
    names = con.execute("""
        SELECT i.sha256, mode(m.meta_name) AS name
        FROM token_images i
        JOIN names_wanted USING (sha256)
        LEFT JOIN token_metadata m USING (mint)
        WHERE i.status = 'ok'
        GROUP BY 1
    """).df()
    con.unregister("names_wanted")
    return dict(zip(names.sha256, names.name.fillna("")))


def show_grid(con, shas, titles=None, ncols=8, size=1.8, suptitle=None):
    """Plot images in a grid; ``titles`` is an optional list of captions aligned with ``shas``.

    A ``None`` sha leaves its tile blank (useful to keep one group per row). Grids are exempt from
    the 4x3 / 6x4 figure sizes, and render without LaTeX because captions hold arbitrary token names.
    """
    shas = list(shas)
    titles = None if titles is None else list(titles)
    imgs = load_images(con, [sha for sha in shas if sha is not None])
    with plain_text():
        nrows = int(np.ceil(len(shas) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.35)))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(shas):]:
            ax.axis("off")
        for k, sha in enumerate(shas):
            ax = axes[k]
            if sha is None:
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():  # thin frame so white/blank artwork stays visible
                spine.set_visible(True)
                spine.set_color(COLORS["grid"])
            if sha in imgs:
                ax.imshow(imgs[sha].convert("RGB"))
            if titles is not None:
                caption = "\n".join(line[:22] for line in str(titles[k]).split("\n"))
                ax.set_title(plain_caption(caption), fontsize=7.5, fontweight="normal",
                             loc="center", color=COLORS["ink_secondary"])
        if suptitle:
            fig.suptitle(suptitle, color=COLORS["ink"])
    return fig
