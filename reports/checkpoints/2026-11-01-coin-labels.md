# Checkpoint 2 (2026-11-01): coin-level multi-label annotation, pilot

Frozen state after moving from one narrative per coin to coin-level multi-label labels, before human annotation.
It builds on [checkpoint 1](2026-11-01-narratives.md) (284 narratives; the top 32 cover 50.1% of texts in topics).
Data snapshot (gitignored): `data/topics/checkpoints/2026-11-01-coin-labels/`.

## Why multi-label

In checkpoint 1 each text has one topic and so one narrative. A coin can tell two stories (Trump and a dog), or
tell one story in a hateful way (geopolitics with antisemitism, an animal used as a racist slur). The topic-level
attribute shares showed that hate content was fragmented: antisemitic memes, Nazi references and racist slurs
formed single-topic narratives outside the top 32, and also appeared inside other narratives (Europe and empire,
Pepe, bananas). With one label per coin, focusing on the top 32 would hide that content.

## Design

Each coin is labelled on two independent axes, with definitions quoted from peer-reviewed CS papers
([../references/definitions.md](../references/definitions.md), [../references/references.bib](../references/references.bib)):

- **Narratives** (any number, main one first): the 29 named narratives among the 32 largest of checkpoint 1 (the
  three no-theme ones left out), plus a free-text narrative outside the list. Definition: a collection of
  documents that address the same event or issue (Hanley et al., IEEE S&P 2024; Gerard et al., ICWSM 2026).
- **Attributes** (present or absent):
  - hate, with its targets (Kiela et al., NeurIPS 2020; hateful symbols from Qu et al., CCS 2025);
  - antisemitism type (Chandra et al., WebSci 2021);
  - racist = hate against race, ethnicity, nationality or immigration status;
  - sexual, sexual_minors, violence, self_harm (Markov et al., AAAI 2023; image wording from Qu et al., CCS 2025);
  - controlled_substances with type, and profanity (Ghosh et al., NAACL 2025).
- Dropped for lack of a published definition: tragedy, real_person. Extremism is folded into hate (hateful
  symbols).

Model: Qwen3.5-9B, local, greedy, no thinking. Each coin is asked twice, from the text alone and from the text
with the image, so the image's contribution is measured rather than self-reported. Annotation rule added in
run 2: "An object associated with an attribute (a weapon, a pill, a flag) is not enough by itself… Do not guess
what an unfamiliar symbol means."

Pilot sample: 500 coins, all with an image.

- 300 coins drawn uniformly from the 5,141,193 coins; prevalences come from these.
- 200 coins from 75 topics with signs of sensitive content; these supply positive cases.

Module: `crypto_analysis/coin_labels.py` (steps `sample`, `label`, `summary`, `page`, `agreement`).

## Results (run 2, text + image, uniform stratum n = 300)

| | Value |
|---|---|
| Coins with 0 / 1 / 2 / 3+ narratives | 8.0% / 78.7% / 10.7% / 2.7% |
| Coins with 2+ narratives | 13.3% |
| Coins with harmful content (hate, sexual, sexual_minors, violence, self_harm, illegal drugs) | 11.0% (95% CI 7.9–15.0) |
| Harmful content together with a narrative | 10.7% |
| LLM includes the coin's checkpoint-1 narrative (coins in a listed narrative) | 64.6% |

| Attribute | Uniform % (95% CI) | Positives, all 500 | Of these, only with the image |
|---|---|---|---|
| hate | 2.7 (1.4–5.2) | 16 | 38% |
| racist | 2.3 (1.1–4.7) | 14 | 36% |
| antisemitic | 1.0 (0.3–2.9) | 5 | 20% |
| sexual | 4.3 (2.5–7.3) | 28 | 50% |
| violence | 2.0 (0.9–4.3) | 15 | 93% |
| controlled_substances | 1.7 (0.7–3.8) | 13 | 54% |
| profanity | 1.7 (0.7–3.8) | 15 | 33% |
| self_harm | 0.3 | 3 | — |
| sexual_minors | 0.3 | 2 | — |
| any harmful | 11.0 (7.9–15.0) | 72 | 57% |

Narratives of the harmful coins, most frequent first: sexuality, Trump and geopolitics, winning and fighting, Jews
and Israel, justice and conspiracies, Pepe, dogs.

Run 1 to run 2 (evidence rule), positives among 500:

| Attribute | Run 1 → run 2 | Effect of the evidence rule |
|---|---|---|
| hate | 27 → 16 | The false positives went away: the Gadsden flag read as a white-supremacist symbol, "Pedo House", "Are you gay?". The 14 checked positives are all correct. One probable miss: "a jew and hitler". |
| controlled_substances | 16 → 13 | "Pill" alone no longer counts. |
| violence | 13 → 15 | Unchanged problem: a held weapon, a tank or a soldier is still marked. |

The prompt is now **frozen**: the same 500 coins are the human validation set, so tuning on them would inflate
LLM–human agreement. Violence is a known weakness, to be measured by its κ.

## Human validation (next)

- Page: `data/topics/coin_labels/pilot/annotation/coin_annotation.html`.
  - Self-contained and entirely in English, with the same definitions given to the LLM.
  - Coins in a fixed shuffled order; annotators do not see the LLM labels.
  - Images withheld for the 2 coins the LLM marked sexual_minors.
- Two or three annotators, working independently, each export a CSV to `annotation/`.
- `python -m crypto_analysis.coin_labels agreement --out-dir data/topics` reports:
  - per attribute: Cohen's κ (two annotators), Fleiss' κ (three or more) and Krippendorff's α;
  - narratives: α with MASI distance on the narrative sets, and κ on the main narrative;
  - the LLM's κ, precision and recall against the human majority.

## Open points

- Checkpoint 1's narratives were judged with our own definition. Re-judging the 1,631 topic pairs with the
  published one takes about 1 h of GPU.
- Violence: re-tune only on a fresh sample, not on the validation set.
- Scaling to 5.1 M coins: LLM labels on a larger stratified sample, then a classifier on the text and image
  embeddings.
