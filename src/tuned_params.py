"""Winning hyper-parameters, written by `python src/tune.py --write-best`.

Do not edit by hand: rerun the search to change them.
"""
BEST_LGBM: dict = {'n_estimators': 800, 'learning_rate': 0.03, 'num_leaves': 127, 'min_child_samples': 20, 'subsample': 0.85, 'colsample_bytree': 0.9, 'reg_lambda': 5}
BEST_CAT: dict = {'learning_rate': 0.05, 'iterations': 2000, 'depth': 12, 'l2_leaf_reg': 30, 'random_strength': 1, 'bagging_temperature': 1}
