"""Ablation driver.

Configs are named on the command line, or selected with --all for everything in
the registry, with --index for the single config addressed by a SLURM array
task, or with --group for one of the named tables the report quotes.

Named runs and --all append to outputs/ablation_results.csv. An --index run
does not, because array tasks run at the same time and appending means reading
that file and writing it back, which loses the rows written in between. Each
task writes its own file under outputs/parts/ instead, and --merge folds them
into the shared table once the array has finished.

A --group run re-scores exactly the configurations one reported table contains,
which is what src/groups.py is for, and writes them to
outputs/ablation_<group>_rerun.csv. The shipped table is never overwritten,
because a rerun inherits today's defaults. Whether it then returns the original
numbers or only their ordering depends on the batch, and --list-groups says
which is which.

    python src/run_ablation.py --group index_forecast
    python src/run_ablation.py --list-groups
    python src/run_ablation.py --count          # how many array tasks are needed
    python src/run_ablation.py --merge          # after a SLURM array
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import Config, evaluate, FOLDS  # noqa: E402
import configs  # noqa: E402
import groups  # noqa: E402

OUT = "outputs/ablation_results.csv"
PARTS = "outputs/parts"


def run_one(cfg: Config, df, folds=FOLDS, out_csv: str = OUT) -> dict:
    t0 = time.time()
    print(f"### {cfg.name}", flush=True)
    mean, res = evaluate(cfg, df, folds=folds)
    row = dict(name=cfg.name, mean_rmse=round(mean, 2),
               **{f"fold{c}": round(v, 2) for c, v in zip(res.cut, res.rmse)},
               last=round(res.rmse.iloc[-1], 2), bias=round(res.bias.mean(), 1),
               secs=round(time.time() - t0))
    os.makedirs("outputs", exist_ok=True)
    prev = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    prev = prev[prev.get("name", pd.Series(dtype=str)) != cfg.name] if len(prev) else prev
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(out_csv, index=False)
    return row


def run(cfgs, df, out_csv=OUT, folds=None):
    rows = [run_one(c, df, folds or FOLDS, out_csv) for c in cfgs]
    r = pd.DataFrame(rows).sort_values("mean_rmse")
    print("\n=== SUMMARY ===", flush=True)
    print(r.to_string(index=False), flush=True)
    return r


def merge_parts(out_csv: str = OUT) -> pd.DataFrame:
    """Fold the per-task files from a SLURM array into the shared table."""
    # Only this script's own parts. src/tune.py writes tuning_*.csv in here too,
    # and those rows have different columns and no config name.
    files = sorted(f for f in os.listdir(PARTS) if f.startswith("ablation_")
                   and f.endswith(".csv")) if os.path.isdir(PARTS) else []
    if not files:
        print(f"no part files in {PARTS}/")
        return pd.DataFrame()
    new = pd.concat([pd.read_csv(f"{PARTS}/{f}") for f in files], ignore_index=True)
    prev = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    if len(prev):
        prev = prev[~prev["name"].isin(new["name"])]
    out = pd.concat([prev, new], ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"merged {len(files)} part files, {len(new)} rows, into {out_csv}")
    return out


def main(argv):
    if argv and argv[0] == "--list-groups":
        for g in sorted(groups.GROUPS):
            note = groups.PARTIAL.get(g, "complete")
            print(f"{g:26s} {len(groups.GROUPS[g]):2d} configs  {note}")
        return
    names = sorted(configs.REGISTRY)
    if argv and argv[0] == "--count":
        print(len(names))
        return
    if argv and argv[0] == "--merge":
        merge_parts()
        return
    df = pd.read_parquet("outputs/base.parquet")
    out_csv = OUT
    if argv and argv[0] == "--group":
        group = argv[1]
        sel, out_csv = groups.resolve(group)
        note = groups.PARTIAL.get(group, "all rows covered")
        print(f"{group}: {note}.\nWriting {out_csv}. The shipped table "
              f"outputs/ablation_{group}.csv is left as the historical record, "
              f"since these configs inherit today's defaults.")
        if not sel:
            return
        # A group rerun is the whole table, not an addition to an older one.
        if os.path.exists(out_csv):
            os.remove(out_csv)
    elif not argv or argv[0] == "--all":
        sel = names
    elif argv[0] == "--index":
        i = int(argv[1])
        if not 0 <= i < len(names):
            raise SystemExit(f"--index {i} is outside the registry, which holds "
                             f"{len(names)} configs, so the array bound is stale")
        sel = [names[i]]
        # One file per task, so concurrent tasks cannot overwrite each other.
        os.makedirs(PARTS, exist_ok=True)
        out_csv = f"{PARTS}/ablation_{i:03d}_{sel[0]}.csv"
    else:
        sel = argv
    run([configs.get(n) for n in sel], df, out_csv=out_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
