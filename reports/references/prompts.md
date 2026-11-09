# Prompts of the pipeline

Literal text of the prompts in use, taken from the code. Values in `{braces}` are filled per item. Every prompt
runs on Qwen3.5-9B (`/home/datalake/models/Qwen3.5-9B`, bf16) with greedy decoding (`do_sample=False`) and reasoning
off (`enable_thinking=False`). Definitions and their sources: [definitions.md](definitions.md).

| Step | Constant | Code | Unit | Image shown |
|---|---|---|---|---|
| Topic name and summary | `LEAF["base"]` | `topic_taxonomy.py` | topic | collage of the coverage samples |
| Macro-topic pair classification (few-shot) | `GUIDED_INTRO`, `FEW_SHOT`, `GUIDED_PAIR` | `narrative_overlap.py` | topic pair, order ab, then ba if SAME | collage of each example pair and of the pair, A above B |
| Macro-topic name | `NAME_PROMPT` | `narrative_overlap.py` | community with 2+ topics | collages of its 4 largest topics |
| Content attributes | `PROMPT` | `topic_attributes.py` | topic, per sample | collage (conditions `both`, `image`); none (`text`) |
| Image safety screen | `SCREEN_PROMPT` | `validation.py` | image | the image |

## 1. Topic name and summary

The answer's `theme` is the topic's name and `summary` its one-sentence description. The attribute lists of this
prompt are used only to find topics with a harm signal when sampling the attributes deck.

```text
You are helping researchers describe the topics of the memecoins launched on pump.fun, a Solana launchpad where anyone can create a token by giving it a name, an optional description and an image. BERTopic grouped the tokens into topics. You see one topic. Its {k} samples cover the whole topic: the topic was split into {k} regions and each sample is the member closest to the centre of one region, shown with the share of the topic in that region. The collage layout and its numbers were added by the researchers.

The topic was built from the token image and text together. The collage shows the {k} sample images, numbered in their top-left corner; the texts below have the same numbers. Topic keywords: {words}
Samples (share of the topic in brackets):
{samples}

Answer with a JSON object only, with these keys:
- "theme": the theme shared by most samples, in at most 6 words, or "none" if the samples share no theme. Being a memecoin, or a theme so broad that it fits almost any memecoin (memes, humor, pop culture, random names), is not a theme
- "summary": one sentence on what the tokens of this topic are about
- "antisemitic", "racist", "sexual": lists of the numbers of the samples that are antisemitic, racist (other than antisemitic), or sexual or pornographic; [] if none
```

## 2. Macro-topic pair classification

Few-shot with images and text, 18 coverage samples per topic, the configuration chosen on the 145
development pairs. One conversation holds the instructions, the five examples and the pair. Each pair is first asked
in the order ab; those answered SAME are asked in the order ba, and an edge of the graph needs SAME in both. A topic
is shown as:

```text
Topic A
Name: {{topic name}}
Keywords: {{c-TF-IDF terms}}
Samples:
  1. {{first 120 characters of the sample's text}}
  2. ...
```

The conversation, in order:

1. The instructions:

```text
You classify pairs of topics of memecoins launched on pump.fun, a Solana launchpad where anyone can create a token with a name, a description and an image. A topic is a group of similar tokens. Researchers group topics into macro-topics by their thematic similarity.

A macro-topic is a broader group of topics with a thematic similarity. Each topic is one of three kinds. A focused topic follows one well-defined content pattern: most of its tokens show the same person, organisation, country or conflict, character or franchise, kind of animal, product or activity. A broad topic has no such pattern: its tokens show many unrelated things, as in internet memes or pop culture in general. A name-only topic has no shared content: its tokens share only how they were created, such as names starting with one letter or generic coin names.

Classify the pair into exactly one class:
SAME: both topics are focused on the same subject.
SAME: both topics are focused, on different subjects that one natural broader theme covers, such as elk and raccoons (animal memecoins).
SAME: both topics are broad (internet culture).
SAME: both topics are name-only (token creation pattern).
DIFFERENT: one topic is broad and the other follows a well-defined pattern, even when that pattern could fit inside the broad topic, such as internet culture and SpongeBob.
DIFFERENT: both topics are focused, on different subjects with no natural broader theme.
DIFFERENT: one topic is centred on a specific event and the other on its subject in general or on another event, even when the same person or entity appears in both.

Decide the kind of each topic by what its samples and images show, not by its name alone. Topics in one macro-topic can differ in their samples and in their style. Harmful or offensive content does not decide the class.

Examples follow, each after its image. The topics were built from the token image and text together. The image shows topic A's sample images (top, labelled A) and topic B's (bottom, labelled B), each tile numbered like the samples; the labels and numbers were added by the researchers.
```

2. Each example: the image of the pair (A above B), then `Example n`, the two topics as above, and
   `Answer: {JSON}`. The examples, from the development pairs and left out of every evaluation:

| Pair | kind_a | kind_b | label | macro_topic | reason |
|---|---|---|---|---|---|
| P004 | focused | focused | SAME | animal memecoins | Two different animals, covered by the theme animals. |
| P043 | broad | focused | DIFFERENT | (empty) | A broad topic and a well-defined franchise pattern. |
| P024 | broad | broad | SAME | internet culture | Both show many unrelated memes with no pattern. |
| P064 | focused | focused | DIFFERENT | (empty) | Different subjects; only a loose region links them, no natural theme. |
| P066 | name-only | name-only | SAME | token creation pattern | Tokens share a name initial, not a content. |

3. The text `Now the pair to classify, after its image.`, the image of the pair, and:

```text
 The topics were built from the token image and text together. The image shows topic A's sample images (top, labelled A) and topic B's (bottom, labelled B), each tile numbered like the samples; the labels and numbers were added by the researchers.

{topic A}

{topic B}

Answer with a JSON object only, with these keys:
- "kind_a": "focused", "broad" or "name-only"
- "kind_b": "focused", "broad" or "name-only"
- "label": "SAME" or "DIFFERENT"
- "macro_topic": if SAME, the shared subject in at most 4 words, otherwise ""
- "reason": at most 15 words
```

The zero-shot comparison (`GUIDED_PROMPT`) sends the image of the pair, then the instructions followed by the text of
step 3, with no examples.

## 3. Macro-topic name

Run on the communities of the chosen configuration (clique percolation, k = 3, over the judged edges of every
candidate pair). Single-topic macro-topics keep their topic's name. The four largest topics are shown by keywords and
four samples each, with their collages stacked in the image; the names of up to 60 topics are listed.

```text
You name macro-topics of memecoins launched on pump.fun, a Solana launchpad where anyone can create a token with a name, a description and an image. A topic is a group of similar tokens. A macro-topic is a broader group of topics with a thematic similarity. Each topic is one of three kinds. A focused topic follows one well-defined content pattern: most of its tokens show the same person, organisation, country or conflict, character or franchise, kind of animal, product or activity. A broad topic has no such pattern: its tokens show many unrelated things, as in internet memes or pop culture in general. A name-only topic has no shared content: its tokens share only how they were created, such as names starting with one letter or generic coin names.

The {n} topics below were grouped into one macro-topic. Their names, largest topic first:
{names of up to 60 largest topics}

The {shown} largest are shown by their keywords and samples; the image shows each one's sample images, one block per topic in the same order, labelled by the researchers.

{topics}

Answer with a JSON object only, with these keys:
- "name": the subject the topics share, in at most 5 words
- "description": one sentence on what the tokens of this macro-topic are about
```

## 4. Content attributes

Shown here for the `both` condition. The other conditions change two passages:

| Condition | Text after "cover the whole topic" | Text after "from what it shows" | Samples |
|---|---|---|---|
| `both` | ` (the image shows their images, numbered like the texts)` | `, in the image and in the text` | numbered texts |
| `image` | ` (the image shows their images, numbered; their texts are not given)` | ` in the image` | "images only, numbered 1 to k in the attached image" |
| `text` | ` (their texts only; their images are not given)` | ` in the text` | numbered texts, no image |

```text
You label the content of memecoins launched on pump.fun, a Solana launchpad where anyone can create a token with a name, a description and an image. Below is one topic of tokens, shown by {k} numbered samples that cover the whole topic (the image shows their images, numbered like the texts). Label every sample from what it shows, in the image and in the text.

ATTRIBUTES. Mark an attribute only when its definition is met. Tone, intent and whether it is a joke do not matter. An object associated with an attribute (a weapon, a pill, a flag) is not enough by itself: what the sample shows or says must meet the definition. Do not guess what an unfamiliar symbol means; read a hidden meaning only when it is a known code (e.g. 1488). Being a memecoin is never an attribute by itself.
- hate: a direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. Attack means violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation; mocking hate crime is also hate. In images: hateful symbols, negative stereotypes, comparing certain groups to animals/objects, e.g. swastika, Confederate flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic, anti-Muslim, anti-Black or white power symbols. Attacking individuals or famous people is not hate unless the attack is based on one of these characteristics; attacking groups perpetrating hate (e.g. terrorist groups) is not hate.
  Targets: racist = race, ethnicity, nationality or immigration status; antisemitic = Jews (political; economic; religious; racial); other_target = any other protected characteristic.
- sexual: content that depicts explicit or implicit sexual behavior. In images: nudity, sexual acts, sexual services, or content otherwise meant to arouse sexual excitement, e.g. erotic content, genital exposure, pornography, suggestive poses, sexual fetish, sex toys or adult products, cartoon or animated nudity.
- violence: content that depicts or shows support for physical violence. In images: violent acts and the suffering or humiliation of others, e.g. gun violence, knife attacks, assaults, sexual assault, torture, homicide, war and combat scenes, mass shootings, child abuse, animal cruelty.
- controlled_substances: trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- other: harmful or sensitive content that none of the attributes above covers, named by the kind of harm in at most 3 words. Never use it to describe what a sample shows: a sample that is not harmful or sensitive gets no label.

TOPIC
Name: {topic name}
Keywords: {terms}
Samples:
  1. {text of sample 1}
  2. ...

Answer with a JSON object only, listing sample numbers (1 to {k}); [] when none:
{"hate": [...], "racist": [...], "antisemitic": [...], "other_target": [...], "sexual": [...], "violence": [...], "controlled_substances": [...], "other": [{"sample": n, "name": "..."}]}
```

## 5. Image safety screen

Run on every image before it goes into an annotation deck. Images flagged, or whose answer cannot be read, are
withheld.

```text
You check images before researchers show them to human annotators. Does this image show sexual content involving a minor (anyone who appears to be under 18), including sexualised drawings? Answer with a JSON object only: {"minor_sexual": true or false}
```

## Retired prompts

Kept in the code as a record, not used in the current pipeline: `narrative_overlap.MACRO_PROMPT` (first macro-topic
guideline, zero-shot), `narratives.PAIR_PROMPT` (same-narrative judge, with `DEFINITIONS`), `coin_labels.INTRO`, `COIN` and `QUESTIONS` (coin-level labels), and `topic_taxonomy.NODE` and
`ASSIGN` (first macro-topic attempt, by walking the topic tree).
