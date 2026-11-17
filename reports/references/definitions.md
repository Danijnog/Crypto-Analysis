# Definitions of the coin labels and where each comes from

Every label applied to a coin (narrative axis and content attributes) uses a definition written by the authors of a
peer-reviewed paper at a computer-science venue. Quotes are verbatim. Page numbers refer to the PDF that was read; when
that PDF is a preprint, the proceedings page is given only when the preprint is the camera-ready layout, and it is
marked *inferred*. Full references: [references.bib](references.bib). PDFs and text extractions (gitignored):
`data/references/pdf/`. Checked on 2026-11-01.

## Narrative axis

**Definition used.** A narrative is a collection of documents (here, coins) that address the same event or issue.

| Source | Venue | Location | Verbatim |
|---|---|---|---|
| Hanley, Kumar & Durumeric `hanley2024specious` | IEEE S&P 2024 | §3.1 "Narrative Definition", p. 2 of arXiv:2308.02068v3 | "we define a narrative/story using the same definition as the Event Registry [38], Hanley et al. [39], and Miranda et al. [40]: collections of documents that seek to address the same event or issue." |
| Gerard, Hanley, Luceri & Ferrara `gerard2026bridging` | ICWSM 2026 | "Cross-Platform Narrative Migration", p. 7 of arXiv:2505.21729v1 | "Following prior work on narrative tracking in multilingual information ecosystems (Hanley, Kumar, and Durumeric 2024), we define a narrative as a collection of posts that focus on the same issue or event." |
| Gerard et al. `gerard2025modeling` | ICWSM 2025 | method only | The story-cluster → narrative-cluster procedure we adapt. The paper gives no explicit definition of "narrative", so no definition is taken from it. |

The definition in the 2026-11-01 checkpoint ("a recurring story, cultural reference or point of view that coins
exploit to attract attention… two topics belong to the same narrative when a pump.fun trader would recognise them as
the same market narrative") is our own wording. It is not a published definition, so it must be replaced by the one
above. Pump.fun-specific examples ("issue" covers a person, an animal, a meme or a cultural reference) are our
operationalisation and should be reported as such.

## Content attributes

### hate

**Definition used** (`kiela2020hateful`): "A direct or indirect attack on people based on characteristics, including
ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and
disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g.
animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also
considered hate speech." Exceptions from the same paper: "attacking individuals/famous people is allowed if the attack
is not based on any of the protected characteristics listed in the definition. Attacking groups perpetrating hate
(e.g. terrorist groups) is also not considered hate."

For images, hateful symbols (`qu2025unsafebench`): "Hateful symbols, negative stereotypes, comparing certain groups
to animals/objects, or otherwise expressing or promoting hate based on identity. For example, swastika, Confederate
flag, SS-runes, KKK hood and cross, Celtic cross, anti-Semitic symbols, skull and crossbones, anti-Muslim symbols,
anti-Black symbols, white power symbols, etc."

**Targets** are recorded from Kiela et al.'s list of protected characteristics. *Racist* = hate whose target is race,
ethnicity, nationality or immigration status. Targets as a separate dimension of hate follow ElSherief et al.

| Source | Venue | Location | Verbatim |
|---|---|---|---|
| Kiela et al. `kiela2020hateful` | NeurIPS 2020 | §2.1 "Hatefulness Definition", p. 3 of the proceedings PDF (vol. 33 p. 2613, *inferred*) | definition and exceptions above |
| Qu et al. `qu2025unsafebench` | ACM CCS 2025 | Appendix, Table 9 "Unsafe image taxonomy", category 1 "Hate", p. 17 of arXiv:2405.03486v3 | symbol definition above |
| Davidson et al. `davidson2017automated` | ICWSM 2017 | §1, p. 1 of arXiv:1703.04009 (proc. p. 512, *inferred*) | "we define hate speech as language that is used to expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group." Also: "our definition does not include all instances of offensive language" (why profanity is a separate attribute) |
| ElSherief et al. `elsherief2018hate` | ICWSM 2018 | §1, p. 1 of arXiv:1804.04257 (camera-ready) | "hate speech – speech that denigrates a person because of their innate and protected characteristics"; Hatebase target categories (p. 3): archaic, class, disability, ethnicity, gender, nationality, religion, sexual orientation |

### antisemitic (hate with target = Jews)

**Definition used** (IHRA working definition, as quoted by `chandra2021subverting`): "Antisemitism is a certain
perception of Jews, which may be expressed as hatred toward Jews. Rhetorical and physical manifestations of
antisemitism are directed toward Jewish or non-Jewish individuals and/or their property, toward Jewish community
institutions and religious facilities." The type is recorded with the paper's four categories:

| Category | Verbatim (Chandra et al., WebSci 2021) | Location (arXiv:2104.05947v3) |
|---|---|---|
| political | "the hatred toward Jews based on the belief that Jews seek national and/or world power." | §3.1, p. 3 (proc. p. 150, *inferred*) |
| economic | "based on the implicit belief that Jews perform and control the economic activities which are harmful for others or the society." | §3.2, p. 3 (proc. p. 150, *inferred*) |
| religious | "deals with bias and discrimination against Jews due to their religious belief in Judaism." | §3.3, p. 4 (proc. p. 151, *inferred*) |
| racial | "based on the prejudice against Jews as a race/ethnic group." The category includes Holocaust denial. | §3.4, p. 4 (proc. p. 151, *inferred*) |

The IHRA quote is in §1, p. 1 (proc. p. 148, *inferred*). Chandra et al. annotate "after looking at the text as well
as the image" (§4.2, p. 4), as we do.

### sexual

**Definition used** (`markov2023holistic`, category S): "Content that depicts explicit or implicit sexual behavior."
For images (`qu2025unsafebench`, category 5 "Sexual"): "Nudity, sexual acts, sexual services, or content otherwise
meant to arouse sexual excitement. For example, erotic content, genital exposure, pornography, suggestive poses,
sexual fetish, sexual violence, nudity in public, sensual dancing, sex toys or adult products, cartoon or animated
nudity, etc."

| Source | Venue | Location |
|---|---|---|
| Markov et al. `markov2023holistic` | AAAI 2023 | §2 "Taxonomy", p. 2 of arXiv:2208.03274 (proc. p. 15010, *inferred*) |
| Qu et al. `qu2025unsafebench` | ACM CCS 2025 | Table 9, p. 17 of arXiv:2405.03486v3 |
| Qu et al. `qu2023unsafe` (supporting) | ACM CCS 2023 | Appendix Table 7, Theme 1, p. 19 of arXiv:2305.13873: "Images that depict nudity, sexual acts, or sexual behavior in an explicit manner, including genitalia, breasts, and sexual poses" |

### sexual_minors

**Definition used** (`markov2023holistic`, subcategory S3): "sexual content involving minors". §2, p. 2 of
arXiv:2208.03274 (proc. p. 15010, *inferred*). Coins with this label are never shown to human annotators.

### violence

**Definition used** (`markov2023holistic`, category V): "Content that depicts or shows support for physical
violence." For images (`qu2025unsafebench`, category 3 "Violence"): "Violent acts and the suffering or humiliation of
others. For example, gun violence, knife attacks, assaults, rape or sexual assault, torture, homicide, war and combat
scenes, mass shootings, child abuse, animal cruelty, etc."

Locations: Markov et al. §2, p. 3 of arXiv:2208.03274 (proc. p. 15011, *inferred*); Qu et al. 2025, Table 9, p. 17.
Supporting source: Qu et al. 2023, Table 7, Theme 3 "Violent", p. 19.

### self_harm

**Definition used** (`markov2023holistic`, category SH): "Content that contains a graphic description or shows
support for self-harm." For images (`qu2025unsafebench`, category 4 "Self-Harm"): "Suicide, cutting, eating
disorders, and other attempts at harming oneself."

Locations: Markov et al. §2, p. 3 (proc. p. 15011, *inferred*); Qu et al. 2025, Table 9, p. 17.

### controlled_substances

**Definition used** (`ghosh2025aegis`, S5 "Controlled/Regulated Substances"): "trafficking, consumption, or use of
regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis." The
type is recorded (illegal drugs, cannabis, alcohol, tobacco), so analyses can keep only illegal drugs. Image
support: `qu2025unsafebench`, category 7 "Illegal Activity": "Drug use, theft, vandalism, and other illegal
activities."

Locations: Aegis 2.0 (NAACL 2025), Appendix A.13.3 "catdesc System Prompt", proceedings p. 6023 (read from the ACL
Anthology PDF, not inferred); Qu et al. 2025, Table 9, p. 17. In Aegis this text is a policy ("Should not – Encourage
or assist people in the …"); we use its scope (which substances, which activities), not its "should not" framing.

### profanity

**Definition used** (`ghosh2025aegis`, S12 "Profanity"): "Contain swear words, curse words, or other obscene or
profane language – This includes offensive words used without any intention to act on them or without any explicit
targetted recipient". Location: Appendix A.13.3, proceedings p. 6026 (read from the ACL Anthology PDF). Davidson et
al. (above) give the reason to keep profanity apart from hate.

## Dropped: no definition by the authors of a CS-venue paper was found

| Former attribute | Decision |
|---|---|
| extremism (Nazi, KKK, terrorist symbols) | Merged into **hate**: UnsafeBench's hate definition lists the swastika, SS-runes and KKK hood and cross as hateful symbols. Kiela et al. exclude attacks *on* terrorist groups from hate. |
| tragedy (real deaths, victims) | Dropped. |
| real_person | Dropped. UnsafeBench's "Political" category ("Politicians, ballot-boxes, protests, or other content that may be used to influence the political process or to campaign") covers only politicians. |

Also defined in UnsafeBench Table 9 but not adopted, because they overlap narratives rather than content attributes:
Political, Deception (conspiracies), Shocking and Harassment.

## Related work read but not used for definitions

- Long, Wong & Cai `long2025bridging` (WWW Companion 2025) and Long, Li & Cai `long2025coinclip` (CIKM 2025):
  multimodal analyses of pump.fun memecoins. They speak of "cultural narratives" but give no definition.
- Piper, So & Bamman (EMNLP 2021, §1, p. 2) state that "there is no consensus when it comes to a minimal definition"
  of narrative. That supports using the operational, event/issue-based definition above.
- No explicit definition was found in Yang & Luo (ACM TIST 2017, illicit drugs on Instagram) or Balsamo et al.
  (WWW 2019, opioids on Reddit); both use keyword lists.
- Sood, Antin & Churchill (CHI 2012) and Wang et al. (CSCW 2014) on profanity were not accessible (ACM DL refused
  automated download), so they are not cited for definitions.

## PDF files

| File in `data/references/pdf/` | Reference |
|---|---|
| `arxiv_2308.02068.pdf` | hanley2024specious |
| `arxiv_2505.21729.pdf` | gerard2026bridging |
| `arxiv_2409.07684.pdf` | gerard2025modeling |
| `neurips2020_hateful_memes.pdf`, `arxiv_2005.04790.pdf` | kiela2020hateful |
| `arxiv_1703.04009.pdf` | davidson2017automated |
| `arxiv_1804.04257.pdf` | elsherief2018hate |
| `arxiv_2104.05947.pdf` | chandra2021subverting |
| `arxiv_2208.03274.pdf` | markov2023holistic |
| `arxiv_2405.03486.pdf` | qu2025unsafebench |
| `arxiv_2305.13873.pdf` | qu2023unsafe |
| `acl_2025.naacl-long.306.pdf`, `arxiv_2501.09004.pdf` | ghosh2025aegis |
| `arxiv_2412.04913.pdf` | long2025bridging |
| `acl_2021.emnlp-main.26.pdf` | Piper et al. 2021 |
