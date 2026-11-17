"""Link wallets that are likely controlled by the same actor ("entities").

Two kinds of evidence, both from ``bonding_curve_trades`` (see ``trade_features``):

1. **Co-signing** (strong). A Solana transaction is signed by every wallet whose trade it carries,
   so wallets trading in the same transaction share control. This is the account-model analogue of
   the multi-input heuristic for Bitcoin (Meiklejohn et al., 2013).
2. **Co-action similarity** (behavioural). Following Pacheco et al. (2021), "Uncovering Coordinated
   Networks on Social Media", each creator wallet is described by the wallets that buy in the
   creation slot of its tokens (its *bundle*), plus the creator itself. Features are TF-IDF weighted,
   so wallets that bundle for many creators (public snipers) count for little, and creators whose
   vectors have cosine similarity >= ``threshold`` and share >= ``min_shared`` bundle wallets are
   linked. Wallets bundling for more than ``max_creators`` creators are dropped as features: that
   can only remove links, never create them.

Entities are the connected components of the union of both link sets.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def creator_wallet_matrix(bundle, max_creators=50):
    """TF-IDF creator x wallet matrix from creation-slot buys.

    ``bundle`` has columns ``mint, coin_creator, trader`` (slot-0 buys by non-creators). TF is the
    number of the creator's tokens the wallet bundled (the creator's own feature counts its tokens
    with any bundle). Returns ``(X, creators, wallets, dropped_wallets)``, rows L2-normalised.
    """
    tf = bundle.groupby(["coin_creator", "trader"]).mint.nunique().rename("tf").reset_index()
    own = (bundle.groupby("coin_creator").mint.nunique().rename("tf").reset_index()
           .assign(trader=lambda d: d.coin_creator))
    df_per_wallet = tf.groupby("trader").coin_creator.nunique()
    dropped = df_per_wallet[df_per_wallet > max_creators].index
    tf = pd.concat([tf[~tf.trader.isin(dropped)], own], ignore_index=True)
    tf = tf.groupby(["coin_creator", "trader"], as_index=False).tf.sum()

    creators = pd.Index(tf.coin_creator.unique())
    wallets = pd.Index(tf.trader.unique())
    rows = creators.get_indexer(tf.coin_creator)
    cols = wallets.get_indexer(tf.trader)
    n_creators = len(creators)
    doc_freq = np.bincount(cols, minlength=len(wallets))
    idf = np.log((1 + n_creators) / (1 + doc_freq)) + 1
    values = (1 + np.log(tf.tf.to_numpy(float))) * idf[cols]
    X = sparse.csr_matrix((values, (rows, cols)), shape=(n_creators, len(wallets)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sparse.diags(1 / np.where(norms == 0, 1, norms)) @ X
    return X.tocsr(), creators, wallets, dropped


def coaction_links(X, threshold=0.7, min_shared=2):
    """Creator pairs (i < j) with cosine >= threshold and >= min_shared shared features."""
    sim = (X @ X.T).tocoo()
    binary = X.copy()
    binary.data[:] = 1
    shared = (binary @ binary.T).tocsr()
    keep = (sim.row < sim.col) & (sim.data >= threshold)
    i, j, s = sim.row[keep], sim.col[keep], sim.data[keep]
    n_shared = np.asarray(shared[i, j]).ravel().astype(int)
    ok = n_shared >= min_shared
    return pd.DataFrame({"i": i[ok], "j": j[ok], "cosine": s[ok], "shared": n_shared[ok]})


def entity_links(bundle, cosign, threshold=0.7, min_shared=2, max_creators=50):
    """All wallet links as a DataFrame ``source, target, kind, weight, shared``.

    ``kind`` is ``coaction`` (creator pairs; ``weight`` = cosine, ``shared`` = shared bundle wallets)
    or ``cosign`` (wallets co-signing transactions; ``weight`` = number of such transactions).
    Also returns the creators in the co-action matrix and the dropped hub wallets.
    """
    X, creators, _, dropped = creator_wallet_matrix(bundle, max_creators)
    co = coaction_links(X, threshold, min_shared)
    parts = [pd.DataFrame({"source": creators[co.i.to_numpy()], "target": creators[co.j.to_numpy()],
                           "kind": "coaction", "weight": co.cosine.to_numpy(),
                           "shared": co.shared.to_numpy()})]
    if len(cosign):
        signers = cosign[["signature", "trader"]].drop_duplicates()
        pairs = signers.merge(signers, on="signature")
        pairs = pairs[pairs.trader_x < pairs.trader_y]
        counts = pairs.groupby(["trader_x", "trader_y"]).size().rename("weight").reset_index()
        parts.append(pd.DataFrame({"source": counts.trader_x, "target": counts.trader_y,
                                   "kind": "cosign", "weight": counts.weight.astype(float),
                                   "shared": 0}))
    return pd.concat(parts, ignore_index=True), creators, dropped


def resolve_entities(bundle, cosign, threshold=0.7, min_shared=2, max_creators=50):
    """Assign every creator wallet (and exclusive bundle wallets) to an entity.

    Returns ``(creator_entity, wallet_entity, edges, info)``:

    - ``creator_entity``: ``coin_creator -> entity`` for creators seen in ``bundle``;
    - ``wallet_entity``: ``trader -> entity`` for bundle wallets whose creators all share one entity;
    - ``edges``: the links (see ``entity_links``) with the ``entity`` they belong to;
    - ``info``: counts of links and dropped hub wallets.
    """
    edges, creators, dropped = entity_links(bundle, cosign, threshold, min_shared, max_creators)

    # Node space: creators first, then any other wallet that appears in a link
    others = pd.Index(pd.unique(edges[["source", "target"]].to_numpy().ravel())).difference(creators)
    nodes = creators.append(others)
    ei, ej = nodes.get_indexer(edges.source), nodes.get_indexer(edges.target)
    graph = sparse.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(graph, directed=False)
    node_entity = pd.Series(labels, index=nodes, name="entity")
    edges = edges.assign(entity=labels[ei])

    creator_entity = node_entity.loc[creators]
    wallet_creators = bundle[["trader", "coin_creator"]].drop_duplicates()
    wallet_creators = wallet_creators.assign(
        entity=creator_entity.reindex(wallet_creators.coin_creator).to_numpy())
    per_wallet = wallet_creators.groupby("trader").entity.agg(["nunique", "first"])
    wallet_entity = per_wallet.loc[per_wallet["nunique"] == 1, "first"].rename("entity")
    wallet_entity = wallet_entity[~wallet_entity.index.isin(creators)]

    info = {
        "creators": len(creators),
        "coaction_links": int((edges.kind == "coaction").sum()),
        "cosign_links": int((edges.kind == "cosign").sum()),
        "dropped_hub_wallets": len(dropped),
        "entities_with_2plus_creators": int((creator_entity.value_counts() > 1).sum()),
    }
    return creator_entity, wallet_entity, edges, info
