import numpy as np
import logging


def acq_margin(X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, random_state=None, **kwargs):
    """
    Find the examples with the smallest gaps between the prediction confidences between the most likely class and
    the second most likely class.
    """
    if size >= len(X_U):
        logging.warning(f"Sample of size={size} requested from data with {len(X_U)} elements!")
        acq_idxs = np.arange(len(X_U), dtype=int)
        return acq_idxs

    prob_mat = clf.predict_proba(X_U_trf)
    a = np.sort(prob_mat, axis=1)
    margin = 1 - (a[:, -1] - a[:, -2])
    acq_idxs = np.argsort(margin)[-size:]
    return acq_idxs


def acq_entropy(X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, random_state=None, **kwargs):
    """
    Return the top examples wrt entropy of predicted probabilities per class.
    """
    if size >= len(X_U):
        logging.warning(f"Sample of size={size} requested from data with {len(X_U)} elements!")
        acq_idxs = np.arange(len(X_U), dtype=int)
        return acq_idxs
    # to make this fast, calculate a matrix of log values of the prediction probs, and then
    # perform a row-wise dot product
    log_base = len(set(y_L))
    prob_mat = clf.predict_proba(X_U_trf)
    log_mat = np.log(prob_mat)/np.log(log_base)
    entropies = np.sum(-prob_mat * log_mat, axis=1)
    acq_idxs = np.argsort(entropies)[-size:]
    return acq_idxs
