"""Macro-topics as an affiliation model (BigCLAM) fitted to the LLM's pair judgments, compared with the
overlapping community detection of ``narrative_overlap`` on the human-labelled pairs.

BigCLAM (Yang and Leskovec, WSDM 2013, Sec. 4-5): every topic u has a non-negative affiliation vector F_u, one
strength per community, and two topics are judged the same macro-topic with probability
p(u, v) = 1 - exp(-F_u . F_v) (Eq. 1). F maximises the log-likelihood (Eq. 2)

    l(F) = sum over edges log(1 - exp(-F_u . F_v)) - sum over non-edges F_u . F_v.

What follows the paper:
- the model, the likelihood and the projection F >= 0;
- the epsilon-community (Sec. 4): an edge between topics sharing no community has the background probability
  eps = 2|E| / (|V|(|V| - 1)) (about 10^-8 in the paper's networks, 0.04 here);
- membership: u belongs to c when F_uc >= delta = sqrt(-log(1 - eps)), with the same eps;
- initialisation from locally minimal neighbourhoods (lowest conductance among their neighbours);
- the number of communities K maximises the likelihood of a 20% hold-out of the node pairs.

What departs from it, and why:
- the likelihood runs over the *judged* pairs only. Only each topic's candidate neighbours and a random sample of
  far pairs were judged; an unjudged pair is missing, not a non-edge. A pair is an edge when both orders said
  yes, as in ``narrative_overlap.graph``; any other judged pair is a non-edge;
- F is updated for all topics at once by projected gradient ascent with a backtracking line search, instead of
  one topic at a time: the graph has 290 nodes, so the full gradient is cheap, and the objective is unchanged.
  One gradient step moves less than the paper's sweep over all nodes, so the stopping rule is a relative gain
  below 10^-7 per step instead of 0.001% per sweep;
- the hold-out is repeated over ``FOLDS`` random splits and averaged, so that K does not hang on one split;
- with ``far="weighted"`` (k = 20 only), each judged far pair is weighted by the inverse of its inclusion
  probability (pairs outside the candidates / far pairs judged, about 34), so the far sample stands for every
  unjudged pair and grouping topics that were never compared is no longer free; eps is then the estimated
  background probability of all pairs. With ``far=True`` the far pairs enter with weight 1, with ``False`` not at all;
- the paper does not say how to fill communities beyond the locally minimal neighbourhoods; we use the
  neighbourhoods of random topics.

As in ``narrative_overlap.detect``, topics in no community are macro-topics of their own.

Comparison (``compare``): every configuration of every algorithm on the 145 development pairs (the 150
human-labelled pairs without the few-shot ones), in two settings: *in graph*, as ``evaluate_communities`` does,
where the LLM's judgment of each human-labelled pair is part of the input; and *masked*, where the judgments of
the development and test pairs are removed from the input, so a pair is in one macro-topic only through the rest
of the graph. 95% bootstrap intervals over the pairs; the test pairs are scored only for the configuration each
algorithm chose on the development pairs.

Outputs in ``<out>/narratives/overlap/``: ``bigclam_k.parquet`` (hold-out likelihood per K),
``communities_bigclam.parquet`` (same columns as ``communities_guided.parquet``), ``affiliations_bigclam.parquet``
(F of each configuration), ``affiliation_comparison.parquet``. Usage::

    python -m crypto_analysis.macro_affiliation fit     --out-dir data/topics
    python -m crypto_analysis.macro_affiliation compare --out-dir data/topics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_analysis.topic_search import SEED

TOL = 1e-7  # relative likelihood gain per step at which to stop (the paper: 0.001% per sweep over all nodes)
MAX_ITER = 5000
HOLDOUT = 0.2
FOLDS = 5
K_GRID = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100)
FAR = (True, False, "weighted")  # far pairs: weight 1, left out, or weighted by inverse inclusion probability


# --- model -------------------------------------------------------------------------------------------------

def loglik(F, u, v, y, eps, w=None):
    """Eq. 2 over the observed pairs (u, v) with labels y (1 edge, 0 non-edge), with the epsilon-community;
    ``w`` weights each pair (1 by default)."""
    w = np.ones(len(y)) if w is None else w
    x = np.einsum("ij,ij->i", F[u], F[v])
    terms = np.where(y == 1, np.log1p(-(1 - eps) * np.exp(-x)), np.log1p(-eps) - x)
    return (w * terms).sum()


def gradient(F, u, v, y, eps, w=None):
    w = np.ones(len(y)) if w is None else w
    x = np.einsum("ij,ij->i", F[u], F[v])
    e = (1 - eps) * np.exp(-x)
    d = w * np.where(y == 1, e / (1 - e), -1.0)  # d l / d x for an edge and a non-edge
    G = np.zeros_like(F)
    np.add.at(G, u, d[:, None] * F[v])
    np.add.at(G, v, d[:, None] * F[u])
    return G


def background(n, n_edges):
    """eps = 2|E| / (|V|(|V| - 1)), the background edge probability (Sec. 4)."""
    return 2 * n_edges / (n * (n - 1))


def fit(F0, u, v, y, eps, w=None, tol=TOL, max_iter=MAX_ITER):
    """Projected gradient ascent with a backtracking (Armijo) line search; returns F and its log-likelihood."""
    F, ll, step = F0.copy(), loglik(F0, u, v, y, eps, w), 1.0
    for _ in range(max_iter):
        G = gradient(F, u, v, y, eps, w)
        step = min(step * 2, 1e3)
        while step > 1e-12:
            new = np.clip(F + step * G, 0, 1e3)
            new_ll = loglik(new, u, v, y, eps, w)
            if new_ll >= ll + 1e-4 * (G * (new - F)).sum():
                break
            step /= 2
        else:
            break
        gain = new_ll - ll
        F, ll = new, new_ll
        if gain < tol * abs(ll):
            break
    return F, ll


def conductance(adj, deg, total, S):
    vol = deg[S].sum()
    cut = vol - adj[np.ix_(S, S)].sum()
    den = min(vol, total - vol)
    return cut / den if den > 0 else np.inf


def initial(n, u, v, y, K, seed=SEED):
    """Locally minimal neighbourhoods (Yang and Leskovec, Sec. 5): topics by the conductance of {u} and its
    neighbours, lowest first; a topic's neighbourhood seeds a community unless a neighbour's already did. If
    there are fewer than K, the rest are seeded by random topics' neighbourhoods."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n))
    adj[u[y == 1], v[y == 1]] = adj[v[y == 1], u[y == 1]] = 1
    deg, total = adj.sum(1), adj.sum()
    hoods = [np.flatnonzero(adj[i]).tolist() + [i] for i in range(n)]
    phi = np.array([conductance(adj, deg, total, hoods[i]) if deg[i] else np.inf for i in range(n)])
    seeds, blocked = [], np.zeros(n, bool)
    for i in np.argsort(phi, kind="stable"):
        if np.isfinite(phi[i]) and not blocked[i]:
            seeds.append(hoods[i])
            blocked[hoods[i]] = True
    while len(seeds) < K:
        seeds.append(hoods[rng.integers(n)])
    F = np.zeros((n, K))
    for c, members in enumerate(seeds[:K]):
        F[members, c] = 1.0
    return F


def delta(eps):
    """Membership threshold (Sec. 5): sqrt(-log(1 - eps)), eps the background edge probability."""
    return np.sqrt(-np.log1p(-eps))


def memberships(F, d):
    """Communities (lists of topics with F_uc >= d), then every topic in none as a community of its own."""
    found = [sorted(np.flatnonzero(F[:, c] >= d).tolist()) for c in range(F.shape[1])]
    found = [c for c in found if c]
    found = [list(c) for c in dict.fromkeys(tuple(c) for c in found)]  # drop duplicate communities
    covered = {t for c in found for t in c}
    return found + [[t] for t in range(F.shape[0]) if t not in covered]


def choose_k(n, u, v, y, w=None, ks=K_GRID, folds=FOLDS, holdout=HOLDOUT, seed=SEED):
    """Mean hold-out log-likelihood per (weighted) pair of each K over ``folds`` random 80/20 splits of the
    observed pairs."""
    w = np.ones(len(y)) if w is None else w
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(folds):
        test = rng.random(len(y)) < holdout
        tr = ~test
        eps = background(n, (w * y)[tr].sum() / (1 - holdout))
        for K in ks:
            F, _ = fit(initial(n, u[tr], v[tr], y[tr], K, seed=seed + f), u[tr], v[tr], y[tr], eps, w[tr])
            rows.append({"fold": f, "K": K,
                         "heldout_ll": loglik(F, u[test], v[test], y[test], eps, w[test]) / w[test].sum()})
    return pd.DataFrame(rows)


# --- data --------------------------------------------------------------------------------------------------

def far_weight(pairs, n):
    """Inverse inclusion probability of a far pair: pairs outside the candidates over far pairs judged."""
    outside = n * (n - 1) // 2 - (pairs.source == "candidate").sum()
    return outside / (pairs.source == "far").sum()


def observations(pairs, k, far, drop=(), n=None):
    """Judged pairs that enter the likelihood and their weights: candidates within rank ``k``; the far pairs with
    weight 1 if ``far`` is True, weighted by ``far_weight`` if it is "weighted" (k must then be the largest rank,
    so every pair is either observed or represented by the far sample); without the pairs in ``drop`` (a set of
    (min, max) tuples)."""
    from crypto_analysis.validation import NEIGHBOURS

    if far == "weighted" and k != NEIGHBOURS:
        raise ValueError("weighted far pairs stand for the pairs outside every candidate neighbourhood: k must be "
                         f"{NEIGHBOURS}")
    keep = (pairs.source == "candidate") & (pairs["rank"] <= k)
    if far:
        keep |= pairs.source == "far"
    obs = pairs[keep]
    w = np.where(obs.source == "far", far_weight(pairs, n) if far == "weighted" else 1.0, 1.0)
    if drop:
        kept = np.array([(a, b) not in drop for a, b in zip(obs.a, obs.b)])
        obs, w = obs[kept], w[kept]
    return obs.a.to_numpy(int), obs.b.to_numpy(int), obs.same.to_numpy(int), w


def human_sets(out_dir):
    """Development pairs (``pairs_gold`` without the few-shot ones) and test pairs with each annotator's label."""
    from crypto_analysis.narrative_overlap import FEW_SHOT
    from crypto_analysis.validation import validation_dir

    val = validation_dir(Path(out_dir))
    dev = pd.read_parquet(val / "pairs_gold.parquet")
    dev = dev[~dev.pair_id.isin([e[0] for e in FEW_SHOT])][["a", "b", "same"]].reset_index(drop=True)
    test = pd.read_parquet(val / "macro_test_sample.parquet")[["pair_id", "a", "b"]]
    names = []
    for f in sorted(val.glob("macro_test_*.csv")):
        labels = pd.read_csv(f).set_index("pair_id")
        names.append(str(labels.annotator.iloc[0]).split()[0].capitalize())
        test[names[-1]] = test.pair_id.map(labels.same.eq("yes"))
    test["agreed"] = test[names].nunique(axis=1) == 1
    test["same"] = test[names[0]]
    return dev, test, names


def human_pair_keys(dev, test):
    return {(min(a, b), max(a, b)) for f in (dev, test) for a, b in zip(f.a, f.b)}


def n_topics(out_dir):
    cands = pd.read_parquet(Path(out_dir) / "validation" / "candidates.parquet")
    return int(max(cands.a.max(), cands.b.max())) + 1


def bigclam(n, u, v, y, w=None, seed=SEED):
    """K by hold-out, then F on every observed pair; returns the communities, F, K and the K table."""
    w = np.ones(len(y)) if w is None else w
    table = choose_k(n, u, v, y, w, seed=seed)
    K = int(table.groupby("K").heldout_ll.mean().idxmax())
    eps = background(n, (w * y).sum())
    F, ll = fit(initial(n, u, v, y, K, seed=seed), u, v, y, eps, w)
    d = delta(eps)
    return memberships(F, d), F, K, table, d


def fit_one(out_dir, k, far, masked):
    """BigCLAM on the judged pairs within neighbour rank ``k``, with or without the far pairs, with or without
    the human-labelled pairs; returns its communities, F and hold-out table as frames."""
    from crypto_analysis.narrative_overlap import judged_pairs

    pairs, n = judged_pairs(out_dir), n_topics(out_dir)
    drop = human_pair_keys(*human_sets(out_dir)[:2]) if masked else set()
    u, v, y, w = observations(pairs, k, far, drop, n)
    found, F, K, table, d = bigclam(n, u, v, y, w)
    params = json.dumps({"far": far, "K": K}, sort_keys=True)
    print(f"  {'masked' if masked else 'in graph'} k={k} far={far}: {len(y):,} pairs, {y.sum():,} edges "
          f"({(w * y).sum():,.0f} weighted), K={K}, "
          f"delta={d:.3f}, {sum(len(c) > 1 for c in found)} multi-topic communities", flush=True)
    comms = pd.DataFrame([{"k": k, "algorithm": "bigclam", "params": params, "community": i, "topic": t}
                          for i, c in enumerate(found) for t in c])
    affs = pd.DataFrame([{"k": k, "params": params, "topic": t, "community": c, "F": F[t, c], "member": F[t, c] >= d}
                         for t in range(n) for c in range(F.shape[1]) if F[t, c] > 0])
    return masked, comms, affs, table.assign(k=k, far=str(far))


def fit_all(out_dir, workers=16, fars=FAR):
    """BigCLAM for every neighbour rank k and every way of using the far pairs in ``fars`` (weighted only at the
    largest k), in graph and masked, in parallel; returns {masked: (communities, affiliations, hold-out table)}."""
    from concurrent.futures import ProcessPoolExecutor

    from crypto_analysis.narrative_overlap import KS
    from crypto_analysis.validation import NEIGHBOURS

    jobs = [(out_dir, k, far, masked) for masked in (False, True) for k in KS for far in fars
            if far != "weighted" or k == NEIGHBOURS]
    with ProcessPoolExecutor(workers) as pool:
        done = list(pool.map(fit_one, *zip(*jobs)))
    return {m: tuple(pd.concat([r[i] for r in done if r[0] == m], ignore_index=True) for i in (1, 2, 3))
            for m in (False, True)}


def baselines_masked(out_dir):
    """The ``narrative_overlap`` algorithms rerun without the judgments of the human-labelled pairs."""
    from crypto_analysis.narrative_overlap import ALGORITHMS, KS, detect, graph, judged_pairs

    pairs, n = judged_pairs(out_dir), n_topics(out_dir)
    drop = human_pair_keys(*human_sets(out_dir)[:2])
    pairs = pairs[[(a, b) not in drop for a, b in zip(pairs.a, pairs.b)]]
    rows = []
    for k in KS:
        G = graph(pairs, n, k)
        for algorithm, grid in ALGORITHMS.items():
            for params in grid:
                found = detect(G, algorithm, params)
                rows += [{"k": k, "algorithm": algorithm, "params": json.dumps(params, sort_keys=True),
                          "community": i, "topic": t} for i, c in enumerate(found) for t in c]
    return pd.DataFrame(rows)


# --- comparison --------------------------------------------------------------------------------------------

def predict(grp, frame):
    member = grp.groupby("topic").community.agg(set).to_dict()
    return np.array([bool(member.get(a, set()) & member.get(b, set())) for a, b in zip(frame.a, frame.b)])


def structure(grp, G, judged):
    """Size and overlap of a configuration's communities. ``isolated_with_edges`` counts topics alone in their
    community although the LLM linked them to another topic. Over the pairs of topics inside each multi-topic
    community: ``within_judged`` is the share the LLM judged at all, ``within_yes`` the share of those it judged
    the same macro-topic (``judged`` maps every judged (a, b) to that answer)."""
    sizes = grp.groupby("community").size()
    member = grp.groupby("topic").size()
    alone = set(grp[grp.community.isin(sizes[sizes == 1].index)].topic)
    inside = [(min(a, b), max(a, b)) for _, ts in grp.groupby("community").topic
              for i, a in enumerate(ts.tolist()) for b in ts.tolist()[i + 1:]]
    answers = [judged[p] for p in inside if p in judged]
    return {"communities": len(sizes), "multi_topic": int((sizes > 1).sum()), "largest": int(sizes.max()),
            "overlapping_topics": int((member > 1).sum()),
            "isolated_with_edges": sum(1 for t in alone if G.degree(t) > 0),
            "within_judged": len(answers) / len(inside) if inside else np.nan,
            "within_yes": float(np.mean(answers)) if answers else np.nan}


def kappa(t, p):
    """Cohen's kappa of binary labels along the last axis (rows are bootstrap draws)."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    po = (t == p).mean(-1)
    pt, pp = t.mean(-1), p.mean(-1)
    pe = pt * pp + (1 - pt) * (1 - pp)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def draws(n, number=5000, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (number, n))


def scores(truth, pred, prefix):
    """Kappa with its 95% bootstrap interval over the pairs, precision and recall against the humans."""
    from crypto_analysis.validation import binary_agreement

    r = binary_agreement(truth, pred)
    i = draws(len(truth))
    lo, hi = np.nanpercentile(kappa(truth[i], pred[i]), [2.5, 97.5])
    return {f"{prefix} kappa": r["kappa"], f"{prefix} kappa lo": lo, f"{prefix} kappa hi": hi,
            f"{prefix} precision": r["llm_precision"], f"{prefix} recall": r["llm_recall"]}


def paired_difference(truth, a, b):
    """Kappa of ``a`` minus kappa of ``b`` on the same pairs, with its 95% paired bootstrap interval."""
    i = draws(len(truth))
    diffs = kappa(truth[i], a[i]) - kappa(truth[i], b[i])
    return float(kappa(truth, a) - kappa(truth, b)), *np.nanpercentile(diffs, [2.5, 97.5])


def compare(out_dir):
    from crypto_analysis.narrative_overlap import COMMUNITIES, MACRO_CONFIG, graph, judged_pairs, overlap_dir

    out = overlap_dir(out_dir)
    dev, test, names = human_sets(out_dir)
    pairs, n = judged_pairs(out_dir), n_topics(out_dir)
    drop = human_pair_keys(dev, test)
    settings = {
        "in graph": (pd.concat([pd.read_parquet(out / COMMUNITIES), pd.read_parquet(out / "communities_bigclam.parquet")]),
                     pairs),
        "masked": (pd.concat([pd.read_parquet(out / "communities_masked.parquet"),
                              pd.read_parquet(out / "communities_bigclam_masked.parquet")]),
                   pairs[[(a, b) not in drop for a, b in zip(pairs.a, pairs.b)]]),
    }
    truth = dev.same.astype(bool).to_numpy()
    production = (MACRO_CONFIG[0], MACRO_CONFIG[1], json.dumps(MACRO_CONFIG[2], sort_keys=True))
    rows, chosen = [], {}
    for setting, (comms, judged) in settings.items():
        answers = dict(zip(zip(judged.a, judged.b), judged.same))
        graphs = {}
        for (k, alg, params), grp in comms.groupby(["k", "algorithm", "params"]):
            G = graphs.setdefault(k, graph(judged, n, k))
            pred = predict(grp, dev)
            rows.append({"setting": setting, "k": k, "algorithm": alg, "params": params,
                         **scores(truth, pred, "dev"), **structure(grp, G, answers)})
        table = pd.DataFrame([r for r in rows if r["setting"] == setting])
        # each algorithm's configuration chosen on the development pairs: highest kappa, then fewest isolated
        for alg, grp in table.groupby("algorithm"):
            best = grp.sort_values(["dev kappa", "isolated_with_edges"], ascending=[False, True]).iloc[0]
            chosen[(setting, alg)] = (best.k, alg, best.params)
        chosen[(setting, "production")] = production
    table = pd.DataFrame(rows)
    # paired differences and test scores for the chosen configurations
    summary = []
    for (setting, label), (k, alg, params) in chosen.items():
        comms = settings[setting][0]
        grp = comms[(comms.k == k) & (comms.algorithm == alg) & (comms.params == params)]
        pred_dev = predict(grp, dev)
        ref = comms[(comms.k == production[0]) & (comms.algorithm == production[1]) & (comms.params == production[2])]
        diff, lo, hi = paired_difference(truth, pred_dev, predict(ref, dev))
        row = {"setting": setting, "chosen as": label, "k": k, "algorithm": alg, "params": params,
               "dev kappa": table[(table.setting == setting) & (table.k == k) & (table.algorithm == alg)
                                  & (table.params == params)]["dev kappa"].iloc[0],
               "minus production": diff, "minus production lo": lo, "minus production hi": hi}
        pred_test = predict(grp, test)
        agreed = test.agreed.to_numpy()
        row.update(scores(test.same.astype(bool).to_numpy()[agreed], pred_test[agreed], "test agreed"))
        for name in names:
            row.update(scores(test[name].astype(bool).to_numpy(), pred_test, f"test {name}"))
        summary.append(row)
    table.to_parquet(out / "affiliation_comparison.parquet", index=False)
    summary = pd.DataFrame(summary)
    summary.to_parquet(out / "affiliation_summary.parquet", index=False)
    return table, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["fit", "compare"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--far", nargs="+", choices=["true", "false", "weighted"],
                        help="fit: only these uses of the far pairs, replacing their earlier fits and keeping the "
                             "rest (default: all, and the baselines rerun masked)")
    args = parser.parse_args()
    from crypto_analysis.narrative_overlap import overlap_dir

    out = overlap_dir(args.out_dir)
    if args.step == "fit":
        fars = FAR if not args.far else tuple({"true": True, "false": False}.get(f, f) for f in args.far)
        for masked, frames in fit_all(args.out_dir, fars=fars).items():
            suffix = "_masked" if masked else ""
            for name, frame in zip(("communities_bigclam", "affiliations_bigclam", "bigclam_k"), frames):
                path = out / f"{name}{suffix}.parquet"
                if args.far and path.exists():
                    old = pd.read_parquet(path)
                    used = old.far if "far" in old else old.params.map(lambda p: json.loads(p)["far"])
                    frame = pd.concat([old[~used.astype(str).isin([str(f) for f in fars])], frame],
                                      ignore_index=True)
                    if "far" in frame:
                        frame["far"] = frame.far.astype(str)
                frame.to_parquet(path, index=False)
        if not args.far:
            print("baselines, masked", flush=True)
            baselines_masked(args.out_dir).to_parquet(out / "communities_masked.parquet", index=False)
    else:
        pd.set_option("display.width", 250)
        table, summary = compare(args.out_dir)
        cols = ["setting", "k", "algorithm", "params", "dev kappa", "dev kappa lo", "dev kappa hi", "dev precision",
                "dev recall", "communities", "multi_topic", "largest", "overlapping_topics", "isolated_with_edges"]
        for setting, grp in table.groupby("setting"):
            print(f"\n== {setting}: best configuration of each algorithm on the development pairs")
            print(grp.sort_values("dev kappa", ascending=False).groupby("algorithm").head(1)[cols].round(3)
                  .to_string(index=False))
        print("\n== chosen configurations: kappa minus the production configuration (paired bootstrap), test")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
