# Checkpoint 2026-11-01: coin narratives (text modality)

Frozen state of the topic → narrative pipeline before the multi-label extension.
Data snapshot (gitignored): `data/topics/checkpoints/2026-11-01-narratives/`.

## Scope

Topics are a tool for the downstream analyses (graduation, trades, coordination, replication), not a replication
of Steffen et al. (ICWSM 2025). A *narrative* is

> a recurring story, cultural reference or point of view that coins exploit to attract attention, for example
> "Trump and US politics", "dog coins", "AI agents", "sexual humor" or "antisemitism". Two topics belong to the same
> narrative when a pump.fun trader would recognise them as the same market "narrative".

## Pipeline as frozen

| Step | Module | Output |
|---|---|---|
| Text topics: BERTopic (MiniLM, cuML UMAP/HDBSCAN) on 2,340,289 deduplicated texts | `topic_search.py` | 398 topics, 1,484,318 texts in topics (63.4%) |
| Narrative embeddings: `paraphrase-multilingual-mpnet-base-v2`, topic centroids | `topic_search.embed_text_docs`, `narratives.topic_centroids` | `text_mpnet.npy`, `narratives/centroids.npy` |
| Coverage samples: 12 k-means medoids per topic, with region shares | `topic_annotation.build_cards` | `annotation/text/cards.parquet` |
| Leaf descriptions + attribute shares (antisemitic, racist, sexual) by Qwen3.5-9B | `topic_taxonomy.describe_leaves` | `leaves.parquet` |
| Candidate pairs: 5 nearest centroids per topic (1,631 pairs) | `narratives.neighbour_pairs` | `narratives/candidates.parquet` |
| Same-narrative judgments by Qwen3.5-9B, both orders, edge only if both agree (91.9% order-consistent, 138 edges) | `narratives.ask_pairs` | `narratives/candidates.parquet` |
| Narratives: Louvain communities on the agreement graph (seed 42) | `narratives.narrative_graph` | `narratives/topics.parquet` |

Reproduce: `python -m crypto_analysis.narratives graph --out-dir data/topics` (judgments are cached in
`candidates.parquet`).

Retired attempts kept for the record: per-topic open coding (`topic_annotation.py pilot/open`), the LLM tree walk
(`topic_taxonomy.py build`), and threshold calibration on HAC (`narratives.py pairs/calibrate`, best F1 0.15).

## Results

- 284 narratives: 44 with 2+ topics (40.2% of texts in topics) and 240 single-topic narratives, mostly legitimate
  single-subject narratives (dogs, cats); 18 are no-theme leaves.
- Long tail: **32 narratives cover 50.1%** of texts in topics and 127 cover 80%. The analyses focus on these 32
  first.

| # | Narrative (provisional name) | Topics | Share | Cumulative |
|---|---|---|---|---|
| 1 | Trump, US politics and geopolitics | 16 | 4.8% | 4.8% |
| 2 | Dogs | 1 | 3.9% | 8.7% |
| 3 | Space and Elon Musk | 12 | 3.2% | 11.9% |
| 4 | Cats | 1 | 3.1% | 15.0% |
| 5 | Pump.fun / Solana meta-memes | 5 | 3.1% | 18.1% |
| 6 | "Official coin" / buy-call names | 1 | 3.0% | 21.1% |
| 7 | Food (mixed with bull/cow) | 10 | 2.5% | 23.5% |
| 8 | No theme | 1 | 2.1% | 25.6% |
| 9 | Sexual and LGBTQ humor | 6 | 1.7% | 27.3% |
| 10 | Justice, crime and "truth" | 5 | 1.7% | 28.9% |
| 11 | Money and wealth | 7 | 1.6% | 30.6% |
| 12 | Bitcoin and crypto | 3 | 1.6% | 32.1% |
| 13 | No theme (names and slurs) | 1 | 1.5% | 33.7% |
| 14 | "Nietzschean" animals (penguin, wolf, eagle) | 3 | 1.4% | 35.1% |
| 15 | AI agents and robots | 7 | 1.3% | 36.4% |
| 16 | Whales | 1 | 1.2% | 37.6% |
| 17 | Monkeys and apes | 3 | 1.1% | 38.7% |
| 18 | Sea animals | 3 | 1.0% | 39.7% |
| 19 | Christmas | 1 | 1.0% | 40.7% |
| 20 | Family and generations | 5 | 0.9% | 41.6% |
| 21 | Winning, fighting, "last chance" | 5 | 0.9% | 42.5% |
| 22 | Religion | 2 | 0.8% | 43.3% |
| 23 | China and East Asia | 4 | 0.8% | 44.1% |
| 24 | Precious metals | 2 | 0.8% | 44.9% |
| 25 | Pepe | 2 | 0.7% | 45.6% |
| 26 | Football and World Cup | 4 | 0.7% | 46.3% |
| 27 | Airdrops and aviation | 1 | 0.7% | 46.9% |
| 28 | Antisemitism | 2 | 0.7% | 47.6% |
| 29 | Running and speedruns | 2 | 0.6% | 48.2% |
| 30 | Viruses (hantavirus) | 1 | 0.6% | 48.8% |
| 31 | Nature and "green" | 3 | 0.6% | 49.4% |
| 32 | No theme | 1 | 0.6% | 50.1% |

Names are provisional and written by hand; final English names come from the naming step. Full ranking with
topic ids and leaf themes: `narratives_ranked.parquet` in the snapshot.

## Known issues

- #7 mixes food with bull/cow coins; #21 is heterogeneous; frog and Pepe topics are not merged.
- Single-label by construction: each text has one topic and so one narrative. Hate content is fragmented into
  single-topic narratives outside the top 32 (antisemitic memes #87, Nazi references #203, racist slurs #75,
  #109, #119, #201), and it also appears inside other narratives (Europe and empire in #1, Pepe #25, bananas #17).
  Next step: coin-level multi-label narratives plus cross-cutting attributes.

## Not done yet

Final naming, image and multimodal linkage, human validation (κ/α), stability (ARI), downstream analyses.
