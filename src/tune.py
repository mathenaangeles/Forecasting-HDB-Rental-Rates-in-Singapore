"""Random hyper-parameter search, addressable as a SLURM job array.

Each task evaluates exactly one sampled parameter set with the same
rolling-origin protocol as every other ablation. The sampler is seeded by the
task index, so the search space is identical whether it is run locally in a
loop or fanned out across the cluster.

Pass --index with a single task number on the cluster, or --range with a start
and an end to work through the draws locally. A --range run appends to
outputs/tuning.csv. A --index run writes outputs/parts/tuning_<index>.csv
instead, because appending means reading the shared file and writing it back,
which loses whatever a concurrent array task wrote in between. Run --merge once
the array has finished to fold the parts into outputs/tuning.csv.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

from experiment import Config, evaluate

OUT = Path("outputs/tuning.csv")
PARTS = Path("outputs/parts")


def sample(model: str, seed: int) -> dict:
    r = np.random.default_rng(seed)
    if model == "lgbm":
        return dict(
            n_estimators=int(r.choice([800, 1200, 1800, 2500])),
            learning_rate=float(r.choice([0.02, 0.03, 0.04, 0.06])),
            num_leaves=int(r.choice([63, 127, 255, 511])),
            min_child_samples=int(r.choice([20, 40, 80, 160])),
            subsample=float(r.choice([0.7, 0.85, 1.0])),
            colsample_bytree=float(r.choice([0.5, 0.7, 0.9])),
            reg_lambda=float(r.choice([1.0, 5.0, 20.0, 50.0])),
        )
    return dict(
        iterations=int(r.choice([1500, 2000, 3000])),
        learning_rate=float(r.choice([0.03, 0.05, 0.08])),
        depth=int(r.choice([6, 8, 10])),
        l2_leaf_reg=float(r.choice([1.0, 4.0, 10.0, 30.0])),
        random_strength=float(r.choice([0.5, 1.0, 2.0])),
        bagging_temperature=float(r.choice([0.0, 0.5, 1.0])),
    )


def task(i: int, df: pd.DataFrame) -> dict:
    """Even indices tune LightGBM, odd tune CatBoost."""
    model = "lgbm" if i % 2 == 0 else "cat"
    params = sample(model, i)
    cfg = Config(name=f"tune{i:03d}_{model}", model=model, params=params)
    t0 = time.time()
    mean, res = evaluate(cfg, df, verbose=False)
    row = dict(idx=i, model=model, mean_rmse=mean,
               last=res["rmse"].iloc[-1], bias=res["bias"].mean(),
               secs=int(time.time() - t0), **params)
    print(f"[{i:03d}] {model:5s} mean={mean:7.2f} last={row['last']:7.2f}  {params}",
          flush=True)
    return row


def append(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if OUT.exists():
        new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
        new = new.drop_duplicates(subset="idx", keep="last")
    OUT.parent.mkdir(exist_ok=True)
    new.sort_values("mean_rmse").to_csv(OUT, index=False)


def write_part(row: dict) -> None:
    """One file per array task, so concurrent tasks cannot overwrite each other."""
    PARTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(PARTS / f"tuning_{row['idx']:03d}.csv", index=False)


def merge_parts() -> None:
    """Fold the per-task files from a SLURM array into outputs/tuning.csv."""
    files = sorted(PARTS.glob("tuning_*.csv"))
    if not files:
        print(f"no part files in {PARTS}/")
        return
    append(pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_dict("records"))
    print(f"merged {len(files)} part files into {OUT}")


def write_best(out: str = "src/tuned_params.py") -> None:
    """Freeze the best row per model into `src/tuned_params.py`."""
    t = pd.read_csv(OUT)
    drop = {"idx", "model", "mean_rmse", "last", "bias", "secs"}
    best = {}
    for m in ("lgbm", "cat"):
        sub = t[t.model == m]
        if sub.empty:
            best[m] = {}
            continue
        row = sub.sort_values("mean_rmse").iloc[0]
        best[m] = {k: (int(v) if float(v).is_integer() else float(v))
                   for k, v in row.items()
                   if k not in drop and pd.notna(v)}
        print(f"best {m}: {row.mean_rmse:.2f}  {best[m]}")
    Path(out).write_text(
        '"""Winning hyper-parameters, written by `python src/tune.py --write-best`.\n\n'
        "Do not edit by hand: rerun the search to change them.\n"
        '"""\n'
        f"BEST_LGBM: dict = {best['lgbm']!r}\n"
        f"BEST_CAT: dict = {best['cat']!r}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, help="one draw, for a SLURM array task")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                    help="a range of draws, run locally in sequence")
    ap.add_argument("--merge", action="store_true", help="fold outputs/parts/ into outputs/tuning.csv")
    ap.add_argument("--write-best", action="store_true")
    a = ap.parse_args()
    if a.write_best:
        write_best()
        raise SystemExit
    if a.merge:
        merge_parts()
        raise SystemExit
    if a.index is None and a.range is None:
        ap.error("pass --index for one draw, or --range START END to run several")
    df = pd.read_parquet("outputs/base.parquet")
    if a.index is not None:
        # An array task keeps its row to itself, for --merge to collect later.
        write_part(task(a.index, df))
    else:
        append([task(i, df) for i in range(*a.range)])
        print(pd.read_csv(OUT).head(10).to_string(index=False))
