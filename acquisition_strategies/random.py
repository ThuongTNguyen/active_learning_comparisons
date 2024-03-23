import numpy as np
import pandas as pd
import logging


def acq_random(X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, random_state=None, **kwargs):
    if size >= len(X_U):
        logging.warning(f"Sample of size={size} requested from data with {len(X_U)} elements!")
        acq_idxs = np.arange(len(X_U), dtype=int)
    else:
        acq_idxs = np.random.default_rng(seed=random_state).choice(len(X_U), replace=False, size=size)
    return acq_idxs