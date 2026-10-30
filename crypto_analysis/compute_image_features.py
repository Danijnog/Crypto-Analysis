"""Compute image features (colour statistics + pHash) for every blob shard.

Writes one Parquet file per input shard, so the run is resumable: shards whose output already
exists are skipped. Usage::

    python -m crypto_analysis.compute_image_features \\
        --images-dir /home/datalake/pumpfun/data/images --out-dir data/image_features --workers 8
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from crypto_analysis.image_features import shard_features


def _process(shard, out_dir):
    features, errors = shard_features(shard)
    tmp = out_dir / f"{shard.stem}.parquet.tmp"
    features.to_parquet(tmp, index=False)
    tmp.rename(out_dir / f"{shard.stem}.parquet")  # atomic: a half-written file never looks done
    if len(errors):
        errors.to_parquet(out_dir / f"{shard.stem}.errors.parquet", index=False)
    return shard.name, len(features), len(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.images_dir.glob("shard_*.parquet"))
    todo = [s for s in shards if not (args.out_dir / s.name).exists()]
    print(f"{len(shards)} shards, {len(shards) - len(todo)} already done, {len(todo)} to process",
          flush=True)

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process, s, args.out_dir) for s in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            name, n_ok, n_err = fut.result()
            elapsed = time.time() - start
            eta = elapsed / k * (len(todo) - k)
            print(f"[{k}/{len(todo)}] {name}: {n_ok} images, {n_err} errors "
                  f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left)", flush=True)


if __name__ == "__main__":
    main()
