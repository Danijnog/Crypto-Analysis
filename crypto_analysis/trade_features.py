"""Extract per-token trade features and wallet-link evidence from ``bonding_curve_trades``.

The trades (~600M rows, one Parquet file per day) are processed one day at a time, so memory stays
bounded, and each day's outputs are written separately, so the run is resumable. For every day::

    <out>/token_day/<day>.parquet   per (mint, day) partial aggregates, summed later per mint
    <out>/bundle/<day>.parquet      non-creator buys in a token's creation slot (+ the next slots)
    <out>/cosign/<day>.parquet      (signature, trader) for transactions with trades by 2+ wallets

Usage::

    python -m crypto_analysis.trade_features --out-dir data/trade_features --threads 6
"""

import argparse
from pathlib import Path
import time

from crypto_analysis.datalake import DATA_DIR, connect

SNIPE_SLOTS = 3  # slots after creation counted as sniping (~1.2 s at 400 ms per slot)


def load_tokens(con):
    """Token launch facts every day's trades are joined against."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE launch AS
        SELECT mint, coin_creator, created_slot, created_at, created_signature FROM tokens
    """)


def day_outputs(con, trades_file):
    """Return the three per-day DataFrames for one trades file."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE day_trades AS
        SELECT t.signature, t.slot, t.block_time, t.mint, t.trader, t.is_buy, t.sol_amount, t.is_bot,
               l.coin_creator, l.created_slot, l.created_at, l.created_signature
        FROM read_parquet('{trades_file}') t
        JOIN launch l USING (mint)
    """)
    token_day = con.execute(f"""
        SELECT mint,
               count(*)                                                       AS n_trades,
               count(*) FILTER (WHERE is_buy)                                 AS n_buys,
               sum(sol_amount) FILTER (WHERE is_buy)                          AS buy_lamports,
               sum(sol_amount) FILTER (WHERE NOT is_buy)                      AS sell_lamports,
               count(DISTINCT trader) FILTER (WHERE is_buy AND block_time - created_at <= 86400)
                                                                              AS buyers_24h,
               count(*) FILTER (WHERE is_buy AND is_bot)                      AS direct_buys,
               count(*) FILTER (WHERE is_buy AND is_bot IS NOT NULL)          AS known_depth_buys,
               sum(sol_amount) FILTER (WHERE signature = created_signature
                                          AND is_buy AND trader = coin_creator) AS dev_buy_lamports,
               sum(sol_amount) FILTER (WHERE trader = coin_creator AND is_buy)  AS creator_buy_lamports,
               sum(sol_amount) FILTER (WHERE trader = coin_creator AND NOT is_buy)
                                                                              AS creator_sell_lamports,
               min(block_time) FILTER (WHERE trader = coin_creator AND NOT is_buy)
                                                                              AS creator_first_sell,
               count(DISTINCT trader) FILTER (WHERE is_buy AND trader <> coin_creator
                                                 AND slot = created_slot)    AS bundle_wallets,
               sum(sol_amount) FILTER (WHERE is_buy AND trader <> coin_creator
                                          AND slot = created_slot)           AS bundle_lamports,
               count(DISTINCT trader) FILTER (WHERE is_buy AND trader <> coin_creator
                                                 AND slot BETWEEN created_slot + 1
                                                              AND created_slot + {SNIPE_SLOTS})
                                                                              AS sniper_wallets,
               sum(sol_amount) FILTER (WHERE is_buy AND trader <> coin_creator
                                          AND slot BETWEEN created_slot + 1
                                                       AND created_slot + {SNIPE_SLOTS})
                                                                              AS sniper_lamports
        FROM day_trades GROUP BY 1
    """).df()
    bundle = con.execute(f"""
        SELECT mint, coin_creator, trader, slot - created_slot AS slot_offset,
               sum(sol_amount) AS lamports, count(*) AS buys
        FROM day_trades
        WHERE is_buy AND trader <> coin_creator AND slot BETWEEN created_slot AND created_slot + {SNIPE_SLOTS}
        GROUP BY 1, 2, 3, 4
    """).df()
    cosign = con.execute("""
        WITH multi AS (
            SELECT signature FROM day_trades GROUP BY 1 HAVING count(DISTINCT trader) > 1
        )
        SELECT DISTINCT d.signature, d.trader, d.mint FROM day_trades d JOIN multi USING (signature)
    """).df()
    return token_day, bundle, cosign


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit", default="16GB")
    args = parser.parse_args()

    kinds = ("token_day", "bundle", "cosign")
    for kind in kinds:
        (args.out_dir / kind).mkdir(parents=True, exist_ok=True)
    days = sorted((args.data_dir / "parquet" / "bonding_curve_trades").glob("bonding_curve_trades_*.parquet"))
    todo = [f for f in days if not (args.out_dir / "cosign" / f"{f.stem[-10:]}.parquet").exists()]
    print(f"{len(days)} trade days, {len(days) - len(todo)} already done, {len(todo)} to process",
          flush=True)

    con = connect(args.data_dir, threads=args.threads, memory_limit=args.memory_limit)
    load_tokens(con)
    start = time.time()
    for k, trades_file in enumerate(todo, 1):
        day = trades_file.stem[-10:]
        outputs = day_outputs(con, trades_file)
        for kind, df in zip(kinds, outputs):  # cosign last: its file marks the day as done
            tmp = args.out_dir / kind / f"{day}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(args.out_dir / kind / f"{day}.parquet")
        elapsed = time.time() - start
        print(f"[{k}/{len(todo)}] {day}: {len(outputs[0]):,} tokens, {len(outputs[1]):,} bundle buys, "
              f"{len(outputs[2]):,} co-signed rows ({elapsed / 60:.1f} min, "
              f"~{elapsed / k * (len(todo) - k) / 60:.1f} min left)", flush=True)


if __name__ == "__main__":
    main()
