"""Score each confirmed variant inside the shipped three-member blend.

  final = (1 - w) * (0.5 lgbm + 0.5 cat) + w * notebook

where w is predict.NOTEBOOK_WEIGHT. This uses the cached fold predictions of
confirm.py (outputs/confirm_*.pkl) and h2h.py (outputs/h2h_preds*.pkl). The rows
are aligned by construction, since both come from the same base frame under the
same filter, and the alignment is asserted on the target.

    python src/blend_weights.py
    python src/blend_weights.py --inner month

The --inner option names which inner tuning protocol of the notebook member to
weight, and is described in src/h2h.py. The default is the one the reported
table was built from. The repaired protocol writes its own table beside it.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import rmse, clip_rent, FOLDS  # noqa: E402
from h2h import preds_path, suffix_for  # noqa: E402

W = 0.3


def main(inner="rows"):
    # Not named tag, which the column prefix below already uses.
    proto = suffix_for(inner)
    nb = pd.read_pickle(preds_path(inner))
    rows = []
    cached = sorted(glob.glob("outputs/confirm_*.pkl"))
    if not cached:
        raise SystemExit("no outputs/confirm_*.pkl found. Run confirm.py for the "
                         "variants to weight first, for example\n"
                         "  python src/confirm.py Js_flat_type_0.5 Jf_blockinfo Jc_ft05_blockinfo")
    for path in cached:
        name = path[len("outputs/confirm_"):-4]
        p = pd.read_pickle(path)
        row = dict(name=name)
        for w in (0.0, W):
            r = {}
            for c, v in p.items():
                assert np.array_equal(v["y"], nb[c]["y"])
                two = 0.5 * (v["lgbm"] + v["cat"])
                r[c] = rmse(v["y"], clip_rent((1 - w) * two + w * nb[c]["notebook"]))
            tag = "2way" if w == 0 else "3way"
            row[f"{tag}_roll16"] = round(np.mean([r[c] for c in FOLDS]), 2)
            row[f"{tag}_hold12"] = round(r[39], 2)
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("3way_roll16")
    out = f"outputs/ablation_blend_weights{proto}.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv[argv.index("--inner") + 1] if "--inner" in argv else "rows")
