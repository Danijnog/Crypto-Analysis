"""Coin-level multi-label annotation: narratives and content attributes of each pump.fun coin.

``narratives`` gives each topic a single narrative, so each coin gets one too, but a coin can tell two
stories (Trump and a dog) or tell one story in a hateful way (geopolitics with antisemitism, an animal
used as a racist slur). Each coin is therefore labelled on two independent axes:

* narratives: any of the 29 named narratives among the 32 largest of the 2026-11-01 checkpoint
  (``reports/checkpoints/2026-11-01-narratives.md``; its three no-theme narratives are left out), main
  one first, plus a free-text narrative outside the list. A narrative is a collection of documents
  that address the same event or issue (Hanley et al., IEEE S&P 2024; Gerard et al., ICWSM 2026);
* attributes: ``ATTRIBUTES``, each present or absent by a definition quoted from a peer-reviewed paper
  (hate: Kiela et al., NeurIPS 2020, with hateful symbols from Qu et al., CCS 2025; sexual, violence,
  self-harm: Markov et al., AAAI 2023, with the image wording of Qu et al., CCS 2025; controlled
  substances and profanity: Ghosh et al., NAACL 2025). Hate also records its targets (Kiela et al.'s
  protected characteristics), so racist (race, ethnicity, nationality or immigration status) is read
  from it, and the type of antisemitism (Chandra et al., WebSci 2021). Where each definition comes from
  is in ``reports/references/definitions.md``.

The pilot asks Qwen3.5-9B (greedy, no thinking) about a sample of coins twice, from the text alone and
from the text with the image, so the image's contribution is measured rather than self-reported. The
sample has a uniform stratum, which estimates prevalences, and a stratum drawn from topics likely to
hold harmful content, which supplies positive cases.

Writes to ``<out>/coin_labels/pilot/``: ``sample.parquet``, ``images/<sha256>.jpg``,
``labels_text.parquet`` and ``labels_multimodal.parquet``. Usage::

    python -m crypto_analysis.coin_labels sample  --out-dir data/topics
    python -m crypto_analysis.coin_labels label   --out-dir data/topics
    python -m crypto_analysis.coin_labels summary --out-dir data/topics

Human validation: ``page`` writes ``annotation/coin_annotation.html``, a self-contained page that each
annotator fills independently and exports as CSV into ``annotation/``; ``agreement`` then reports
inter-annotator agreement (Cohen's or Fleiss' kappa, Krippendorff's alpha, MASI distance for the
narrative sets; Passonneau, 2006) and the LLM's agreement with the human majority::

    python -m crypto_analysis.coin_labels page      --out-dir data/topics
    python -m crypto_analysis.coin_labels agreement --out-dir data/topics
"""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

CHECKPOINT = "checkpoints/2026-11-01-narratives"
RANDOM_COINS = 300
SENSITIVE_COINS = 200
IMAGE_SIZE = 512
DESCRIPTION_CHARS = 600
SEED = 42

# Names written for the prompt from each narrative's leaf themes; ids are the checkpoint ranks.
NARRATIVES = {
    1: "Trump, US politics and geopolitics (presidents, wars, nations, empires)",
    2: "Dogs",
    3: "Space, aliens and Elon Musk",
    4: "Cats",
    5: "Pump.fun and Solana platform memes",
    6: "Buy calls and 'official coin' hype",
    7: "Food and cooking; bulls and cows",
    9: "Sexuality, LGBTQ and female characters",
    10: "Justice, crime, truth and conspiracies",
    11: "Money and getting rich",
    12: "Bitcoin and crypto",
    14: "Wild animals (penguins incl. the Nietzschean penguin meme, wolves, eagles)",
    15: "AI agents and robots",
    16: "Whales and the ocean",
    17: "Monkeys, apes and bananas",
    18: "Sea animals and fishing",
    19: "Christmas",
    20: "Family, children, parents and the elderly",
    21: "Winning, losing, fighting and last chances",
    22: "Religion and spiritual figures",
    23: "China, North Korea and East Asia (incl. pandas)",
    24: "Gold, silver and other metals",
    25: "Pepe the Frog",
    26: "Football (soccer) and the World Cup",
    27: "Airdrops and aviation",
    28: "Jews, Israel and Palestine",
    29: "Running, racing and speedruns",
    30: "Viruses and pandemics",
    31: "Nature, plants and green",
}

# Definitions quoted from their sources (reports/references/definitions.md); text and image wordings joined.
ATTRIBUTES = {
    "hate": "a direct or indirect attack on people based on characteristics, including ethnicity, race, "
            "nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and "
            "disability or disease. Attack means violent or dehumanizing (comparing people to non-human things, e.g. "
            "animals) speech, statements of inferiority, and calls for exclusion or segregation; mocking hate crime "
            "is also hate. In images: hateful symbols, negative stereotypes, comparing certain groups to "
            "animals/objects, e.g. swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, "
            "anti-Semitic, anti-Muslim, anti-Black or white power symbols. Attacking individuals or famous people is "
            "not hate unless the attack is based on one of these characteristics; attacking groups perpetrating "
            "hate (e.g. terrorist groups) is not hate.",
    "sexual": "content that depicts explicit or implicit sexual behavior. In images: nudity, sexual acts, sexual "
              "services, or content otherwise meant to arouse sexual excitement, e.g. erotic content, genital "
              "exposure, pornography, suggestive poses, sexual fetish, sex toys or adult products, cartoon or "
              "animated nudity.",
    "sexual_minors": "sexual content involving minors.",
    "violence": "content that depicts or shows support for physical violence. In images: violent acts and the "
                "suffering or humiliation of others, e.g. gun violence, knife attacks, assaults, sexual assault, "
                "torture, homicide, war and combat scenes, mass shootings, child abuse, animal cruelty.",
    "self_harm": "content that contains a graphic description or shows support for self-harm; suicide, cutting, "
                 "eating disorders, and other attempts at harming oneself.",
    "controlled_substances": "trafficking, consumption, or use of regulated or controlled substances, including but "
                             "not limited to illegal drugs, tobacco, alcohol, and cannabis.",
    "profanity": "swear words, curse words, or other obscene or profane language, including offensive words used "
                 "without any intention to act on them or without any explicit targeted recipient.",
}
HATE_TARGETS = ["ethnicity", "race", "nationality", "immigration_status", "religion", "caste", "sex",
                "gender_identity", "sexual_orientation", "disability_or_disease"]
RACE_TARGETS = ["ethnicity", "race", "nationality", "immigration_status"]
ANTISEMITISM = {
    "political": "hatred toward Jews based on the belief that Jews seek national and/or world power",
    "economic": "the belief that Jews perform and control economic activities harmful for others or the society",
    "religious": "bias and discrimination against Jews due to their religious belief in Judaism",
    "racial": "prejudice against Jews as a race/ethnic group, including Holocaust denial",
}
SUBSTANCES = ["illegal_drugs", "cannabis", "alcohol", "tobacco"]

# Topics whose leaf description mentions these, or whose samples carried an attribute, form the
# sensitive stratum. It only decides where positive cases are sought; prevalences use the uniform stratum.
SENSITIVE = re.compile(r"nazi|hitler|holocaust|jew|antisemit|racis|slur|hate|porn|sex|nud|breast|goon|drug|weed|"
                       r"cannabis|cocaine|meth|fentanyl|kill|murder|shoot|gun|weapon|war\b|terror|violen|suicid|"
                       r"death|dead|victim|blood", re.IGNORECASE)

INTRO = """You label coins launched on pump.fun, a Solana memecoin launchpad. Every coin is a memecoin, so being a \
memecoin, a token or "on Solana" is never a narrative or an attribute by itself. Label only what the coin's name, \
symbol, description{and_image} show; read a hidden meaning only when it is a known code (e.g. 1488)."""

COIN = """COIN
Name: {name}
Symbol: {symbol}
Description: {description}{image}"""

NARRATIVE = ("A narrative is a collection of coins that address the same event or issue (on pump.fun an issue can be "
             "a person, an animal, a meme or a cultural reference).")

QUESTIONS = """NARRATIVES. {narrative} Choose every narrative below that the coin clearly belongs to, the main one \
first; a coin can belong to several or to none. If it belongs to a narrative that is not listed, name it in \
"other_narrative".
{narratives}

ATTRIBUTES. Mark an attribute only when its definition is met. Tone, intent and whether it is a joke do not matter. \
An object associated with an attribute (a weapon, a pill, a flag) is not enough by itself: what the coin shows or says \
must meet the definition. Do not guess what an unfamiliar symbol means.
{attributes}

Answer with a JSON object only, with these keys, in this order:
- "description": one sentence on what the coin shows{in_image}
- "narratives": list of narrative ids (e.g. ["N3", "N25"]), main one first; [] if none
- "other_narrative": a narrative not listed, in at most 5 words, or null
- "attributes": object mapping each attribute that is present to its evidence in at most 10 words; {{}} if none
- "hate_targets": if hate is present, the characteristics attacked, among {targets}; otherwise []
- "antisemitism": if the hate targets Jews, its type: {antisemitism}; otherwise null
- "substances": if controlled_substances is present, which among {substances}; otherwise []
Format: {{"description": "...", "narratives": [...], "other_narrative": null, "attributes": {{...}}, \
"hate_targets": [...], "antisemitism": null, "substances": [...]}}"""


def _ids(values, known):
    out = []
    for v in values if isinstance(values, list) else [values]:
        m = re.search(r"\d+", str(v))
        if m and int(m.group()) in known and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def _known(values, known):
    values = {str(x).strip().lower().replace(" ", "_") for x in values} if isinstance(values, list) else set()
    return [k for k in known if k in values]


def coin_prompt(row, with_image):
    has_image = with_image and isinstance(row.sha256, str)
    description = (row.description or "")[:DESCRIPTION_CHARS] or "(none)"
    coin = COIN.format(name=row.name or "(none)", symbol=row.symbol or "(none)", description=description,
                       image="\nImage: attached" if has_image else ("\nImage: none" if with_image else ""))
    questions = QUESTIONS.format(
        narrative=NARRATIVE, narratives="\n".join(f"N{i}: {name}" for i, name in NARRATIVES.items()),
        attributes="\n".join(f"- {attr}: {d}" for attr, d in ATTRIBUTES.items()),
        in_image=", in the text and in the image" if has_image else "", targets=", ".join(HATE_TARGETS),
        antisemitism="; ".join(f"{k} ({d})" for k, d in ANTISEMITISM.items()), substances=", ".join(SUBSTANCES))
    return f"{INTRO.format(and_image=' and image' if with_image else '')}\n\n{coin}\n\n{questions}"


def read_answer(answer):
    """Flat label row from the parsed JSON answer."""
    attrs = answer.get("attributes") or {}
    attrs = {str(k).strip().lower(): str(v) for k, v in attrs.items()} if isinstance(attrs, dict) else {}
    other = answer.get("other_narrative")
    antisemitism = str(answer.get("antisemitism") or "").strip().lower()
    row = {"description": str(answer.get("description", "")),
           "narratives": _ids(answer.get("narratives") or [], NARRATIVES),
           "other_narrative": None if other in (None, "", "null", "None") else str(other),
           "hate_targets": _known(answer.get("hate_targets"), HATE_TARGETS),
           "antisemitism": antisemitism if antisemitism in ANTISEMITISM else None,
           "substances": _known(answer.get("substances"), SUBSTANCES),
           "evidence": json.dumps({k: v for k, v in attrs.items() if k in ATTRIBUTES})}
    row.update({attr: attr in attrs for attr in ATTRIBUTES})
    return row


def sample_coins(out_dir, n_random=RANDOM_COINS, n_sensitive=SENSITIVE_COINS, seed=SEED):
    """Pilot coins with their text, image hash, topic and checkpoint narrative (``sample.parquet``),
    and their images resized to ``IMAGE_SIZE`` (``images/``)."""
    from crypto_analysis.datalake import DATA_DIR, connect
    from crypto_analysis.gallery import load_images
    from crypto_analysis.topic_annotation import _rgb

    root, out = Path(out_dir), Path(out_dir) / "coin_labels" / "pilot"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coins = pd.read_parquet(root / "token_topics.parquet", columns=["mint", "sha256", "text_topic"])
    coins["text_topic"] = coins.text_topic.fillna(-1).astype(int)

    leaves = pd.read_parquet(root / CHECKPOINT / "leaves.parquet")
    flagged = leaves[[f"{a}_share" for a in ("antisemitic", "racist", "sexual")]].gt(0).any(axis=1)
    matched = (leaves.theme.fillna("") + " " + leaves.summary.fillna("")).str.contains(SENSITIVE)
    sensitive_topics = set(leaves.topic[flagged | matched])

    uniform = coins.iloc[rng.choice(len(coins), n_random, replace=False)].assign(stratum="uniform")
    pool = coins[coins.text_topic.isin(sensitive_topics) & ~coins.mint.isin(uniform.mint)]
    sensitive = pool.iloc[rng.choice(len(pool), n_sensitive, replace=False)].assign(stratum="sensitive")
    sample = pd.concat([uniform, sensitive], ignore_index=True)
    print(f"{len(coins):,} coins; {len(sensitive_topics)} sensitive topics holding {len(pool):,} coins", flush=True)

    ranked = pd.read_parquet(root / CHECKPOINT / "narratives_ranked.parquet")
    narrative = {t: r for r, ts in zip(ranked["rank"], ranked.topic_ids) for t in ts}
    sample["checkpoint_narrative"] = sample.text_topic.map(narrative).astype("Int64")

    con = connect(DATA_DIR, threads=8)
    con.register("pilot", sample[["mint"]])
    text = con.execute("""
        SELECT p.mint, coalesce(m.meta_name, t.name) AS name, coalesce(m.meta_symbol, t.symbol) AS symbol,
               m.description
        FROM pilot p LEFT JOIN tokens t USING (mint) LEFT JOIN token_metadata m USING (mint)
    """).df().drop_duplicates("mint")
    sample = sample.merge(text, on="mint", how="left")

    images = load_images(con, sample.sha256.dropna(), DATA_DIR / "images")
    for sha, img in images.items():
        img = _rgb(img)
        img.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        img.save(out / "images" / f"{sha}.jpg", quality=90)
    sample.loc[~sample.sha256.isin(images), "sha256"] = None
    print(f"{len(sample)} coins sampled, {sample.sha256.notna().sum()} with an image", flush=True)
    sample.to_parquet(out / "sample.parquet", index=False)
    return sample


def label_coins(out_dir, conditions=("text", "multimodal"), limit=None):
    """LLM labels of the pilot coins from the text alone and from the text with the image."""
    from PIL import Image

    from crypto_analysis.topic_taxonomy import Asker

    out = Path(out_dir) / "coin_labels" / "pilot"
    sample = pd.read_parquet(out / "sample.parquet").head(limit)
    ask = Asker()
    for condition in conditions:
        rows, start = [], time.time()
        for row in sample.itertuples():
            with_image = condition == "multimodal"
            image = (Image.open(out / "images" / f"{row.sha256}.jpg")
                     if with_image and isinstance(row.sha256, str) else None)
            raw, answer = ask(coin_prompt(row, with_image), image, max_new_tokens=400)
            rows.append({"mint": row.mint, "parsed": bool(answer), "raw": raw, **read_answer(answer)})
            if len(rows) % 50 == 0:
                print(f"  {condition}: {len(rows)}/{len(sample)} ({(time.time() - start) / 60:.1f} min)", flush=True)
        labels = pd.DataFrame(rows)
        labels.to_parquet(out / f"labels_{condition}.parquet", index=False)
        print(f"{condition}: {len(labels)} coins, {labels.parsed.mean():.1%} parsed "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarise(out_dir):
    """Multi-label prevalence, attribute prevalence (uniform stratum, Wilson 95% CI) and the image's
    contribution (attributes found with the image but not from the text alone)."""
    out = Path(out_dir) / "coin_labels" / "pilot"
    sample = pd.read_parquet(out / "sample.parquet")
    text = sample.merge(pd.read_parquet(out / "labels_text.parquet"), on="mint")
    mm = sample.merge(pd.read_parquet(out / "labels_multimodal.parquet"), on="mint")
    for frame in (text, mm):
        frame["antisemitic"] = frame.hate & frame.antisemitism.notna()
        frame["racist"] = frame.hate & frame.hate_targets.map(lambda t: any(x in t for x in RACE_TARGETS))
        frame["illegal_drugs"] = frame.substances.map(lambda s: "illegal_drugs" in s or "cannabis" in s)
        frame["harmful"] = frame[["hate", "sexual", "sexual_minors", "violence", "self_harm"]].any(axis=1) | \
            frame.illegal_drugs
        frame["n_narratives"] = frame.narratives.map(len) + frame.other_narrative.notna()
    attrs = list(ATTRIBUTES) + ["antisemitic", "racist", "illegal_drugs", "harmful"]

    uni = mm[mm.stratum == "uniform"]
    n = len(uni)
    print(f"parsed: text {text.parsed.mean():.1%}, multimodal {mm.parsed.mean():.1%}; uniform stratum n={n}")
    print("\nnarratives per coin (multimodal, uniform):")
    print(uni.n_narratives.clip(upper=3).value_counts(normalize=True).sort_index().rename({3: "3+"}).to_string())
    multi = uni.n_narratives >= 2
    print(f"2+ narratives: {multi.mean():.1%}; harmful attribute together with a narrative: "
          f"{(uni.harmful & (uni.n_narratives >= 1)).mean():.1%}")

    rows = []
    for a in attrs:
        k = int(uni[a].sum())
        lo, hi = _wilson(k, n)
        pos = mm[a] & mm.parsed
        image_only = int((pos & ~text.set_index("mint").loc[mm.mint, a].to_numpy()).sum())
        rows.append({"attribute": a, "uniform_%": 100 * k / n, "ci_low_%": 100 * lo, "ci_high_%": 100 * hi,
                     "positives_all": int(pos.sum()), "image_only": image_only,
                     "image_only_%": 100 * image_only / max(int(pos.sum()), 1)})
    print("\n" + pd.DataFrame(rows).round(1).to_string(index=False))

    inside = mm[mm.checkpoint_narrative.isin(list(NARRATIVES))]
    agree = [int(c) in ns for c, ns in zip(inside.checkpoint_narrative, inside.narratives)]
    print(f"\ncoins whose topic is in a listed narrative: {len(inside)}; LLM includes that narrative for "
          f"{np.mean(agree):.1%}")
    harmful = mm[mm.harmful]
    print(f"\nharmful coins ({len(harmful)}): their narratives")
    print(harmful.narratives.explode().dropna().map(lambda i: NARRATIVES[int(i)]).value_counts().head(15).to_string())


def _text(value):
    return value if isinstance(value, str) else ""


def build_page(out_dir, width=384, seed=SEED):
    """Self-contained human annotation page (``annotation/coin_annotation.html``): the pilot coins in a fixed
    shuffled order with the definitions given to the LLM, answers kept in the browser and exported as CSV.
    Images of coins that any saved LLM run marked as sexual content involving minors are withheld."""
    from crypto_analysis.topic_annotation import _jpeg_base64

    out = Path(out_dir) / "coin_labels" / "pilot"
    sample = pd.read_parquet(out / "sample.parquet").sample(frac=1, random_state=seed)
    withheld = set()
    for path in out.glob("labels_*.parquet"):
        labels = pd.read_parquet(path)
        withheld |= set(labels.mint[labels.sexual_minors])
    coins = [{"mint": row.mint, "name": _text(row.name), "symbol": _text(row.symbol),
              "description": _text(row.description)[:DESCRIPTION_CHARS], "withheld": row.mint in withheld,
              "image": _jpeg_base64(out / "images" / f"{row.sha256}.jpg", width)
                       if isinstance(row.sha256, str) and row.mint not in withheld else None}
             for row in sample.itertuples()]
    data = {"coins": coins, "narratives": {str(k): v for k, v in NARRATIVES.items()}, "attributes": ATTRIBUTES,
            "hate_targets": HATE_TARGETS, "antisemitism": ANTISEMITISM, "substances": SUBSTANCES,
            "narrative_definition": NARRATIVE}
    template = (Path(__file__).parent / "templates" / "coin_annotation.html").read_text()
    page = out / "annotation" / "coin_annotation.html"
    page.parent.mkdir(exist_ok=True)
    page.write_text(template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(f"{len(coins)} coins ({len(withheld)} images withheld), {page} ({page.stat().st_size / 1e6:.1f} MB)")


def _masi(a, b):
    """MASI distance between two label sets (Passonneau, 2006)."""
    a, b = set(a), set(b)
    if a == b:
        return 0.0
    monotonicity = 2 / 3 if a <= b or b <= a else 1 / 3 if a & b else 0.0
    return 1 - len(a & b) / len(a | b) * monotonicity


def krippendorff_alpha(units, distance=lambda a, b: float(a != b)):
    """Krippendorff's alpha; ``units`` holds each unit's values, one per coder (units with fewer than two
    values are not pairable and are skipped)."""
    units = [u for u in units if len(u) >= 2]
    values = [v for u in units for v in u]
    n = len(values)
    observed = sum(sum(distance(u[i], u[j]) for i in range(len(u)) for j in range(len(u)) if i != j) / (len(u) - 1)
                   for u in units) / n
    expected = sum(distance(values[i], values[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    return 1 - observed / expected if expected else np.nan


def fleiss_kappa(units):
    """Fleiss' kappa for units rated by the same number of coders (``units``: each unit's values)."""
    categories = sorted({v for u in units for v in u})
    counts = np.array([[u.count(c) for c in categories] for u in units], dtype=float)
    m = counts.sum(axis=1)[0]
    p = counts.sum(axis=0) / counts.sum()
    agreement = ((counts * (counts - 1)).sum(axis=1) / (m * (m - 1))).mean()
    chance = (p ** 2).sum()
    return (agreement - chance) / (1 - chance) if chance < 1 else np.nan


def _binary_labels(frame):
    """Binary label columns (attributes and the derived antisemitic, racist and illegal_drugs) of human or LLM rows."""
    out = pd.DataFrame({a: frame[a].astype(bool) for a in ATTRIBUTES}, index=frame.index)
    out["antisemitic"] = out.hate & frame.antisemitism.fillna("").astype(str).ne("")
    out["racist"] = out.hate & frame.hate_targets.map(lambda t: any(x in t for x in RACE_TARGETS))
    out["illegal_drugs"] = frame.substances.map(lambda s: "illegal_drugs" in s or "cannabis" in s)
    return out


def _read_annotations(out):
    rows = []
    for path in sorted((out / "annotation").glob("*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows.append(frame[frame.reviewed == "1"])
    if not rows:
        raise FileNotFoundError(f"no annotation CSVs in {out / 'annotation'}")
    human = pd.concat(rows, ignore_index=True)
    for col in ("narratives", "attributes", "hate_targets", "substances"):
        human[col] = human[col].map(lambda s: [x for x in s.split(";") if x])
    human["narratives"] = human.narratives.map(lambda ns: [int(n) for n in ns])
    for attr in ATTRIBUTES:
        human[attr] = human.attributes.map(lambda a, attr=attr: attr in a)
    return human


def agreement(out_dir):
    """Agreement among the human annotators (Cohen's kappa for two, Fleiss' kappa for three or more,
    Krippendorff's alpha always; MASI distance for the narrative sets) and of the LLM (text and image)
    with the human majority, on the coins every annotator reviewed (``annotation/agreement.parquet``)."""
    from sklearn.metrics import cohen_kappa_score

    out = Path(out_dir) / "coin_labels" / "pilot"
    human = _read_annotations(out)
    coders = sorted(human.annotator.unique())
    complete = human.groupby("mint").annotator.nunique().eq(len(coders))
    human = human[human.mint.isin(complete.index[complete])].sort_values(["mint", "annotator"])
    print(f"{len(coders)} annotators ({', '.join(coders)}); {human.mint.nunique()} coins reviewed by all")

    llm = pd.read_parquet(out / "labels_multimodal.parquet").set_index("mint")
    llm_bin = _binary_labels(llm)
    human_bin = _binary_labels(human).assign(mint=human.mint.to_numpy(), annotator=human.annotator.to_numpy())
    rows = []
    for label in [*ATTRIBUTES, "antisemitic", "racist", "illegal_drugs"]:
        wide = human_bin.pivot(index="mint", columns="annotator", values=label)
        units = wide.astype(int).to_numpy().tolist()
        majority = wide.mean(axis=1)
        decided = majority.ne(0.5)
        consensus, predicted = majority[decided].gt(0.5), llm_bin.loc[majority.index[decided], label]
        tp = int((consensus & predicted).sum())
        rows.append({"label": label, "human_positives": int(wide.any(axis=1).sum()),
                     "cohen_kappa": cohen_kappa_score(wide.iloc[:, 0], wide.iloc[:, 1]) if len(coders) == 2 else np.nan,
                     "fleiss_kappa": fleiss_kappa(units) if len(coders) >= 3 else np.nan,
                     "alpha": krippendorff_alpha(units),
                     "llm_kappa_mean": np.mean([cohen_kappa_score(wide[c], llm_bin.loc[wide.index, label]) for c in coders]),
                     "llm_precision": tp / max(int(predicted.sum()), 1), "llm_recall": tp / max(int(consensus.sum()), 1)})
    table = pd.DataFrame(rows)

    sets = human.assign(labels=[sorted(n) + (["other"] if o.strip() else [])
                                for n, o in zip(human.narratives, human.other_narrative)])
    units = sets.groupby("mint").labels.apply(list).tolist()
    main = human.pivot(index="mint", columns="annotator", values="main_narrative").replace("", "none")
    narrative = {"label": "narratives (set, MASI)", "human_positives": int(sum(any(u) for u in units)),
                 "alpha": krippendorff_alpha(units, _masi),
                 "cohen_kappa": cohen_kappa_score(main.iloc[:, 0], main.iloc[:, 1]) if len(coders) == 2 else np.nan}
    table = pd.concat([table, pd.DataFrame([narrative])], ignore_index=True)
    table.to_parquet(out / "annotation" / "agreement.parquet", index=False)
    print("\n" + table.round(3).to_string(index=False))
    print("\ncohen_kappa for narratives is on the main narrative; LLM columns compare the LLM (text and image) "
          "with the human majority, ties left out")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sample", "label", "summary", "page", "agreement"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="label only the first coins (smoke test)")
    args = parser.parse_args()
    if args.step == "sample":
        sample_coins(args.out_dir)
    elif args.step == "label":
        label_coins(args.out_dir, limit=args.limit)
    elif args.step == "summary":
        summarise(args.out_dir)
    elif args.step == "page":
        build_page(args.out_dir)
    else:
        agreement(args.out_dir)


if __name__ == "__main__":
    main()
