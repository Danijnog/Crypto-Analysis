"""Catalogue of every multimodal narrative, for manual inspection.

Writes ``<out>/narratives/catalog.html``, a self-contained page searchable by name, word or coin name, and
``<out>/narratives/catalog.csv`` (the same without images). Narratives are ranked by coins, each coin placed
by its multimodal topic after outlier reduction, as in ``narrative_figures``. For each one: its name (the
coin-label name for the head, else the provisional name), share of coins, coins, graduation rate, its topics
with their c-TF-IDF words and leaf theme (``topic_taxonomy leaves``), its distinctive words (weighted log-odds,
``narrative_figures.distinctive_words``), its most frequent coin names, and six images covering its region
of the CLIP image space (``narrative_figures.representative_images``: k-means medoids, screened for nudity,
gore and hate symbols) with the most frequent name of their coins. Slurs in words and names are masked.

The page holds coin images, so it stays in ``data/`` (not versioned). Usage::

    python -m crypto_analysis.narrative_catalog --out-dir data/topics --clip-dir data/clip
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

import pandas as pd

from crypto_analysis.narrative_figures import (
    _mask,
    distinctive_words,
    image_names,
    narratives_table,
    representative_images,
)

NAMES = 12  # most frequent coin names shown per narrative
THUMB = 160


def coin_table(out_dir, ranked):
    """One row per coin: mint, narrative rank (outliers reduced), graduated, name."""
    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.narratives import coin_narratives

    root = Path(out_dir)
    coins = coin_narratives(root, {"n": ("multimodal", "published")}, reduced=True)[["mint", "n"]]
    coins["rank"] = coins.n.map(ranked.set_index("narrative")["rank"])
    flags = pd.read_parquet(root / "sampling" / "coins.parquet", columns=["mint", "graduated"])
    coins = coins.dropna(subset=["rank"]).merge(flags, on="mint")
    con = connect(DATA_DIR, threads=16)
    con.register("wanted", coins[["mint"]])
    names = con.execute("""
        SELECT w.mint, any_value(trim(m.meta_name)) AS name
        FROM wanted w JOIN token_metadata m USING (mint) GROUP BY 1
    """).df()
    return coins.merge(names, on="mint", how="left")


def frequent_names(coins, k=NAMES):
    """{rank: [(name, coins), ...]}: the ``k`` most frequent coin names of each narrative."""
    named = coins.dropna(subset=["name"])
    named = named[named.name != ""]
    counts = named.groupby(["rank", "name"]).size().rename("n").reset_index()
    counts = counts.sort_values(["rank", "n", "name"], ascending=[True, False, True]).groupby("rank").head(k)
    return {int(r): [(_mask(n), int(c)) for n, c in zip(g.name, g.n)] for r, g in counts.groupby("rank")}


def thumbnails(shas, size=THUMB):
    """{sha256: base64 JPEG} of the given images."""
    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import _rgb

    out = {}
    for sha, img in load_images(connect(DATA_DIR, threads=8), shas, DATA_DIR / "images").items():
        img = _rgb(img)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        out[sha] = base64.b64encode(buf.getvalue()).decode()
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Narrative catalogue</title>
<style>
:root {{ --bg:#f6f4ef; --card:#fff; --ink:#1b2433; --muted:#5b6577; --line:#e2ded4; --accent:#a35a12; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }}
header {{ position:sticky; top:0; background:var(--bg); padding:16px 24px; border-bottom:1px solid var(--line); z-index:1; }}
h1 {{ margin:0 0 8px; font-size:22px; }}
header p {{ margin:0 0 10px; color:var(--muted); }}
input {{ width:100%; max-width:560px; padding:8px 12px; font-size:15px; border:1px solid var(--line); border-radius:8px; }}
main {{ padding:16px 24px; display:grid; gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
.card h2 {{ margin:0; font-size:18px; }}
.meta {{ color:var(--muted); margin:2px 0 10px; }}
.head {{ color:var(--accent); font-weight:600; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:10px; margin:10px 0; }}
.grid figure {{ margin:0; }}
.grid img {{ width:100%; aspect-ratio:1; object-fit:contain; background:#f1efe9; border-radius:6px; }}
.grid figcaption {{ font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.row {{ margin:4px 0; }}
.row b {{ display:inline-block; min-width:120px; color:var(--muted); font-weight:600; }}
.topics div {{ margin:2px 0 2px 124px; }}
.hidden {{ display:none; }}
</style></head><body>
<header><h1>Narrative catalogue</h1>
<p>{n} multimodal narratives (published definition), ranked by coins. Head narratives (the coin-label list) are
marked. Images cover each narrative's region of the CLIP space; images read as nudity, gore or hate symbols are
screened out. Slurs masked.</p>
<input id="q" placeholder="Search names, words, coin names…" autofocus></header>
<main id="list">{cards}</main>
<script>
const q = document.getElementById("q"), cards = [...document.querySelectorAll(".card")];
q.addEventListener("input", () => {{ const t = q.value.toLowerCase();
  cards.forEach(c => c.classList.toggle("hidden", t && !c.dataset.text.includes(t))); }});
</script></body></html>"""


def card(row, topics, names, picked, thumbs, image_label, head):
    e = html.escape
    images = "".join(
        f'<figure><img loading="lazy" alt="{e(image_label.get(s, ""))}" src="data:image/jpeg;base64,{thumbs[s]}">'
        f'<figcaption title="{e(image_label.get(s, ""))}">{e(image_label.get(s, "(no name)"))}</figcaption></figure>'
        for s, _ in picked if s in thumbs)
    topic_rows = "".join(f"<div>{t['topic']}: <i>{e(t['theme'] or 'no theme')}</i> · {e(t['words'])}</div>"
                         for t in topics)
    name_list = ", ".join(f"{e(n)} ({c})" for n, c in names)
    text = " ".join([row["name"], row["words"], *(t["words"] for t in topics), *(t["theme"] or "" for t in topics),
                     *(n for n, _ in names)]).lower()
    label = ' <span class="head">· coin-label list</span>' if head else ""
    return (f'<section class="card" data-text="{e(text)}"><h2>{row["rank"]}. {e(row["name"])}{label}</h2>'
            f'<p class="meta">{row["share_pct"]:.2f}% of coins · {row["coins"]:,} coins · '
            f'{row["graduated_pct"]:.2f}% graduated · {row["topics"]} topic(s)</p>'
            f'<div class="grid">{images}</div>'
            f'<div class="row"><b>Distinctive words</b>{e(row["words"])}</div>'
            f'<div class="row"><b>Coin names</b>{name_list}</div>'
            f'<div class="row topics"><b>Topics</b>{topic_rows}</div></section>')


def build(out_dir, clip_dir):
    from crypto_analysis.coin_labels import NARRATIVES

    root = Path(out_dir)
    ranked = narratives_table(root)
    n = len(ranked)
    coins = coin_table(root, ranked)
    grad = coins.groupby("rank").graduated.mean()
    ranked["graduated_pct"] = 100 * ranked["rank"].map(grad).fillna(0)
    ranked["words"] = ranked["rank"].map(distinctive_words(root, clip_dir, ranked, n)).fillna("")
    names = frequent_names(coins)
    info = pd.read_parquet(root / "multimodal" / "full_info.parquet").set_index("topic").words
    themes = pd.read_parquet(root / "taxonomy" / "multimodal" / "base" / "leaves.parquet").set_index("topic").theme
    topics = {r: [{"topic": int(t), "words": " ".join(str(info.get(t, "")).split()[:10]), "theme": themes.get(t)}
                  for t in ts] for r, ts in zip(ranked["rank"], ranked.topic_ids)}
    picked = representative_images(root, clip_dir, ranked, n)
    shas = [s for ss in picked.values() for s, _ in ss]
    labels, thumbs = image_names(shas), thumbnails(shas)

    out = root / "narratives"
    rows = ranked.to_dict("records")
    cards = "".join(card(r, topics[r["rank"]], names.get(r["rank"], []), picked[r["rank"]], thumbs, labels,
                         r["rank"] in NARRATIVES) for r in rows)
    (out / "catalog.html").write_text(PAGE.format(n=n, cards=cards))
    table = ranked[["rank", "narrative", "name", "topics", "coins", "share_pct", "graduated_pct", "words"]].copy()
    table["topic_ids"] = ranked.topic_ids.map(lambda ts: " ".join(map(str, ts)))
    table["coin_names"] = table["rank"].map(lambda r: json.dumps(names.get(r, []), ensure_ascii=False))
    table.round(4).to_csv(out / "catalog.csv", index=False)
    size = (out / "catalog.html").stat().st_size / 1e6
    print(f"{n} narratives, {len(thumbs)} images: {out / 'catalog.html'} ({size:.1f} MB), {out / 'catalog.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.out_dir, args.clip_dir)


if __name__ == "__main__":
    main()
