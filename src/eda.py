"""Every exploratory statistic the report quotes, recomputed from the raw files.

The figures in src/figs_eda.py show the shape of the data. This script prints
the numbers that appear in the text of Section II, so that each one can be
checked against its source without reading a plot. It writes nothing except to
standard output and outputs/eda_summary.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

ROWS: list[tuple[str, str]] = []


def say(label: str, value) -> None:
    """Print one statistic and record it for the summary file."""
    text = value if isinstance(value, str) else f"{value}"
    ROWS.append((label, text))
    print(f"{label:<52} {text}")


def main() -> None:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print("\n--- Shape and span")
    say("training rows", f"{len(train):,}")
    say("test rows", f"{len(test):,}")
    say("training months", f"{train.RENT_APPROVAL_DATE.min()} to {train.RENT_APPROVAL_DATE.max()}")
    say("test months", f"{test.RENT_APPROVAL_DATE.min()} to {test.RENT_APPROVAL_DATE.max()}")
    last = train.RENT_APPROVAL_DATE.max()
    after = (test.RENT_APPROVAL_DATE > last).mean()
    say("test rows after the last labelled month", f"{after:.3%}")

    print("\n--- Columns that carry nothing")
    both = pd.concat([train.drop(columns=["MONTHLY_RENT"]), test], ignore_index=True)
    for col in ("FURNISHED", "FEE"):
        say(f"{col}: distinct values over all {len(both):,} rows",
            f"{both[col].nunique()} ({both[col].iloc[0]!r})")

    print("\n--- Flat type labels")
    say("FLAT_TYPE labels in training as written", train.FLAT_TYPE.nunique())
    norm = train.FLAT_TYPE.str.replace(" ", "-", regex=False).str.lower()
    say("FLAT_TYPE labels after normalising the separator", norm.nunique())
    say("rarest flat type", f"{norm.value_counts().index[-1]} ({norm.value_counts().iloc[-1]} rows)")

    print("\n--- The target")
    r = train.MONTHLY_RENT
    say("rent range", f"S${r.min():,.0f} to S${r.max():,.0f}")
    say("mean rent", f"S${r.mean():,.2f}")
    say("median rent", f"S${r.median():,.0f}")
    say("standard deviation", f"S${r.std():,.2f}")
    say("skew of rent", f"{r.skew():+.2f}")
    say("skew of log rent", f"{np.log(r).skew():+.2f}")
    say("rents that are exact multiples of S$50", f"{(r % 50 == 0).mean():.3%}")

    month = pd.PeriodIndex(train.RENT_APPROVAL_DATE, freq="M")
    # The loader lower-cases every string field, so the unit key has to be built
    # the same way or the two spellings of a street split one block into two.
    street = train.STREET.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    block = train.BLOCK.str.lower()
    say("mean rent in the first labelled month", f"S${r[month == month.min()].mean():,.0f}")
    say("mean rent in the last labelled month", f"S${r[month == month.max()].mean():,.0f}")
    first, final = r[month == month.min()].mean(), r[month == month.max()].mean()
    say("rise over the labelled period", f"{final / first - 1:.1%}")
    for year, m in train.groupby(month.year).MONTHLY_RENT.mean().items():
        say(f"  mean rent in {year}", f"S${m:,.0f}")

    print("\n--- Size and the cross-section")
    med = train.groupby(norm).MONTHLY_RENT.median()
    say("median rent, 1-room", f"S${med.get('1-room', float('nan')):,.0f}")
    say("median rent, 5-room", f"S${med.get('5-room', float('nan')):,.0f}")
    area = train.FLOOR_AREA_SQM
    say("variance of rent explained by floor area alone",
        f"{np.corrcoef(area, r)[0, 1] ** 2:.1%}")
    for name, key in (("approval month", month),
                      ("town and flat type", [train.TOWN, norm])):
        grouped = r.groupby(key).transform("mean")
        say(f"variance explained by {name}", f"{1 - grouped.sub(r).var() / r.var():.1%}")
    both_keys = [month, train.TOWN, norm]
    grouped = r.groupby(both_keys).transform("mean")
    say("variance explained by both together", f"{1 - grouped.sub(r).var() / r.var():.1%}")

    print("\n--- Outliers")
    q1, q3 = r.quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flag = (r < lo) | (r > hi)
    say("Tukey fence", f"S${lo:,.0f} to S${hi:,.0f}")
    say("rents outside the fence", f"{flag.sum()} ({flag.mean():.2%})")
    unit = [street, block, norm, month]
    size = r.groupby(unit).transform("size")
    gmean = r.groupby(unit).transform("mean")
    say("flagged rents that have a twin", f"{(size[flag] > 1).mean():.1%}")
    say("median gap from the twin-group mean",
        f"S${(r[flag & (size > 1)] - gmean[flag & (size > 1)]).abs().median():,.0f}")

    print("\n--- The floor set by indistinguishable rows")
    grp = r.groupby(unit)
    n, m = grp.transform("size"), grp.transform("mean")
    twin = n > 1
    say("training rows sharing block, flat type and month", f"{twin.mean():.3%}")
    say("groups of at least two such rows", f"{(grp.size() > 1).sum():,}")
    sse = ((r[twin] - m[twin]) ** 2).sum()
    say("within-group RMSE", f"S${np.sqrt(sse / twin.sum()):,.2f}")
    dof = twin.sum() - (grp.size() > 1).sum()
    say("noise level corrected for fitted group means", f"S${np.sqrt(sse / dof):,.2f}")
    say("  as a share of mean rent", f"{np.sqrt(sse / dof) / r.mean():.0%}")
    # Per year, with the same correction for the group means that were fitted.
    yearly = pd.DataFrame(dict(y=month.year, d=r - m, g=grp.ngroup()))[twin]
    for year, part in yearly.groupby("y"):
        dof = len(part) - part.g.nunique()
        say(f"  within-group SD in {year}", f"S${np.sqrt((part.d ** 2).sum() / dof):,.0f}")

    keyed = pd.DataFrame(dict(street=street, block=block, ft=norm, month=month,
                              rent=r))[twin]
    spread = keyed.groupby(["street", "block", "ft", "month"]).rent.std()
    w = spread.idxmax()
    rents = keyed.loc[(keyed.street == w[0]) & (keyed.block == w[1])
                      & (keyed.ft == w[2]) & (keyed.month == w[3]), "rent"]
    say("most dispersed group", f"{w[1]} {w[0]}, {w[2]}, {w[3]}")
    say("  its rents", ", ".join(f"S${x:,.0f}" for x in sorted(rents, reverse=True)))

    print("\n--- The street-name join")
    blocks = pd.read_csv("data/auxiliary/sg-hdb-block.csv")
    key = ["TOWN", "BLOCK", "STREET"]
    bl = blocks.rename(columns=str.upper)
    bl = bl.rename(columns={"STREET_NAME": "STREET", "BLK_NO": "BLOCK"})
    have = {c for c in key if c in bl.columns}
    if have == set(key):
        raw = train.merge(bl[key].drop_duplicates(), on=key, how="left", indicator=True)
        say("rows matching the block table as written",
            f"{(raw._merge == 'both').mean():.1%}")
        lower = train.assign(STREET=train.STREET.str.lower())
        bl2 = bl.assign(STREET=bl.STREET.str.lower())
        hit = lower.merge(bl2[key].drop_duplicates(), on=key, how="left", indicator=True)
        say("rows matching after lower-casing the street",
            f"{(hit._merge == 'both').mean():.1%}")
    say("rows written in title case", f"{(train.STREET != train.STREET.str.lower()).mean():.1%}")
    say("distinct town, block and street as written",
        f"{train[key].drop_duplicates().shape[0]:,}")
    lowered = train.assign(STREET=train.STREET.str.lower())
    say("distinct town, block and street once lower-cased",
        f"{lowered[key].drop_duplicates().shape[0]:,}")
    spellings = train.assign(S=train.STREET.str.lower()).groupby(
        ["TOWN", "BLOCK", "S"]).STREET.nunique()
    say("blocks appearing under both spellings", f"{(spellings > 1).sum():,}")

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(ROWS, columns=["statistic", "value"]).to_csv(
        "outputs/eda_summary.csv", index=False)
    print("\nwrote outputs/eda_summary.csv")


if __name__ == "__main__":
    main()
