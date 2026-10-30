"""DuckDB access to the pump.fun datalake (Parquet written by pumpfun-crawler).

    con = connect()                  # views: token_images, tokens, token_metadata, graduations, blobs
    build_token_flags(con)           # per-token replication / graduation table: token_flags

Views read the Parquet globs directly, so they always reflect what is on disk.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path("/home/datalake/pumpfun/data")


def connect(data_dir=DATA_DIR, threads=8, memory_limit="16GB"):
    """DuckDB connection with one view per datalake table.

    ``token_images.day`` is the partition date taken from the file name. ``threads`` and
    ``memory_limit`` keep the session polite on a shared machine.
    """
    data_dir = Path(data_dir)
    parquet = data_dir / "parquet"
    con = duckdb.connect()
    con.execute(f"SET threads TO {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(rf"""
        CREATE OR REPLACE VIEW token_images AS
        SELECT *, CAST(regexp_extract(filename, '(\d{{4}}-\d{{2}}-\d{{2}})', 1) AS DATE) AS day
        FROM read_parquet('{parquet}/token_images/*.parquet', filename = true)
    """)
    con.execute(f"CREATE OR REPLACE VIEW tokens AS "
                f"SELECT * FROM read_parquet('{parquet}/tokens/*.parquet')")
    con.execute(f"CREATE OR REPLACE VIEW token_metadata AS "
                f"SELECT mint, meta_name, meta_symbol, description "
                f"FROM read_parquet('{parquet}/token_metadata/*.parquet')")
    con.execute(f"CREATE OR REPLACE VIEW graduations AS "
                f"SELECT * FROM read_parquet('{parquet}/graduations/*.parquet')")
    con.execute(f"CREATE OR REPLACE VIEW blobs AS "
                f"SELECT * FROM read_parquet('{data_dir}/images/shard_*.parquet')")
    return con


def build_token_flags(con, grad_window_hours=24, table="token_flags"):
    """Materialise one row per token with a stored image, placed relative to the first token that
    used the same image (``sha256``), ordered by ``tokens.created_at``.

    Columns added to the token's own fields:

    - ``n_tokens``: tokens using the image; ``first_created``: launch of its first user;
      ``first_grad``: first graduation among its users (NULL if none).
    - ``graduated``: graduated at any time; ``grad_fast``: graduated within ``grad_window_hours``
      of launch (the outcome to compare groups, since recent launches had less time to graduate).
    - ``is_copy``: a later token with the same image; ``is_self_relaunch``: a copy by the creator
      of the first token; ``copies_graduate``: a copy launched after the image's first graduation;
      ``secs_after_graduation``: ``created_at - first_grad``.
    - ``role``: ``original`` | ``self-relaunch`` | ``copy (other creator)`` | ``copy of a graduate``
      (a copy of a graduate takes that role even when it is also a self-relaunch).
    """
    window = int(grad_window_hours * 3600)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE {table} AS
    WITH base AS (
        SELECT i.mint, i.sha256, i.day, t.created_at, t.coin_creator, g.block_time AS grad_time
        FROM token_images i
        JOIN tokens t USING (mint)
        LEFT JOIN graduations g USING (mint)
        WHERE i.status = 'ok'
    ), first_use AS (
        SELECT sha256, count(*) AS n_tokens, min(created_at) AS first_created,
               arg_min(coin_creator, created_at) AS first_creator
        FROM base GROUP BY 1
    ), first_grad AS (
        SELECT sha256, min(grad_time) AS first_grad
        FROM base WHERE grad_time IS NOT NULL GROUP BY 1
    ), flagged AS (
        SELECT b.*, f.n_tokens, f.first_created, g.first_grad,
               b.grad_time IS NOT NULL                                      AS graduated,
               coalesce(b.grad_time - b.created_at <= {window}, false)     AS grad_fast,
               b.created_at > f.first_created                               AS is_copy,
               b.created_at > f.first_created
                   AND b.coin_creator = f.first_creator                     AS is_self_relaunch,
               coalesce(b.created_at > g.first_grad, false)                 AS copies_graduate,
               b.created_at - g.first_grad                                  AS secs_after_graduation
        FROM base b
        JOIN first_use f USING (sha256)
        LEFT JOIN first_grad g USING (sha256)
    )
    SELECT *,
           CASE WHEN NOT is_copy THEN 'original'
                WHEN copies_graduate THEN 'copy of a graduate'
                WHEN is_self_relaunch THEN 'self-relaunch'
                ELSE 'copy (other creator)' END AS role
    FROM flagged
    """)
    return table
