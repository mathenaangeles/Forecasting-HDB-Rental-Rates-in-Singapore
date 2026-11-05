"""Candidate extra blend members (ablation K), scored on the confirmation splits.

Two candidates are supported. `ridge` is a linear model on the shipped detrended
target and features, so it brings a model class that no tree member has.
`nb_plus` is the team notebook (raw rent, lagged block history) given the
block-composition and coordinate features it lacked, but still fitted on raw
rent so that it stays a genuinely different view of the problem. The candidates
to score are named on the command line.

Each one writes outputs/member_<name>.pkl with the same {cut: {y, pred, TM}}
layout, and its rows are aligned with those of confirm.py and h2h.py.

    python src/members.py ridge nb_plus
    python src/members.py nb_plus --inner month

The --inner option reaches the notebook member's own hyperparameter search and
is described in src/h2h.py. It defaults to the protocol the report's tables were
written under, and the repaired one writes member_nb_plus_repaired.pkl beside
them rather than over them.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import Config, fit_predict, rmse, clip_rent, HORIZON, FOLDS  # noqa: E402
from configs import BLOCK_INFO  # noqa: E402
from h2h import NUM, notebook_frame, notebook_fit_predict, suffix_for  # noqa: E402

CUTS = FOLDS + [39]
# The shipped configuration, i.e. configs.Jc_ft05_blockinfo.
SHIPPED = dict(index_by="FLAT_TYPE", index_shrink=0.5, extra_num=list(BLOCK_INFO))
NB_PLUS = NUM + BLOCK_INFO + ["LATITUDE", "LONGITUDE", "DIST_CBD"]


def main(names, inner="rows"):
    tag = suffix_for(inner)
    base = pd.read_parquet("outputs/base.parquet")
    tr_b = base[base._split == "train"]
    nb = notebook_frame(base)
    tr_n = nb[nb._split == "train"]
    for name in names:
        out = {}
        for cut in CUTS:
            t0 = time.time()
            if name == "ridge":
                a, b = tr_b[tr_b.TM < cut], tr_b[(tr_b.TM >= cut) & (tr_b.TM < cut + HORIZON)]
                p = fit_predict(Config(model="ridge", **SHIPPED), a, b, ref_t=cut - 1)
            elif name == "nb_plus":
                a, b = tr_n[tr_n.TM < cut], tr_n[(tr_n.TM >= cut) & (tr_n.TM < cut + HORIZON)]
                p, _ = notebook_fit_predict(a, b, NB_PLUS, inner=inner)
            else:
                raise SystemExit(name)
            y = b["MONTHLY_RENT"].to_numpy()
            out[cut] = dict(y=y, pred=np.asarray(p), TM=b["TM"].to_numpy())
            print(f"  {name} cut={cut} RMSE={rmse(y, clip_rent(p)):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # Ridge has no inner search, so its cache carries no protocol tag.
        pd.to_pickle(out, f"outputs/member_{name}{tag if name == 'nb_plus' else ''}.pkl")
        r = {c: rmse(v["y"], clip_rent(v["pred"])) for c, v in out.items()}
        print(f"{name}: roll16 {np.mean([r[c] for c in FOLDS]):.2f}  hold12 {r[39]:.2f}", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    which = "rows"
    if "--inner" in argv:
        i = argv.index("--inner")
        which = argv[i + 1]
        del argv[i:i + 2]
    main(argv, which)
