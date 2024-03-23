import numpy as np
import pandas as pd
import logging


def init_random(X, y, size, random_state=None):
    if size >= len(X):
        logging.warning(f"Sample of size={size} requested from data with {len(X)} elements!")
        acq_idxs = np.arange(dtype=int)
    else:
        acq_idxs = np.random.default_rng(seed=random_state).choice(len(X), replace=False, size=size)
    return acq_idxs