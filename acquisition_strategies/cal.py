import numpy as np
import re, sys, os, functools
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import clone as sklearn_clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
from scipy.special import kl_div
from matplotlib import pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import logging


class CAL(object):
    def __init__(self, precompute_dist_mat=False, k=3, additional_transform=None):
        """

        Initialize with label-unlabeled distance matrix.
        :param LU_dist_mat: If this is provided, acquisition computations can be fast. But it might not be possible
            to always provide this, e.g., the feature vectors depend on current labeled pool.
        :param k: # neigbours
        :param embeddings: can pass in a dict with structure {key: numpy array} where key can be "L" or "U" and
            the values are embeddings. If this is passed in we can save calls to the encoder (if the encoding is
            data-independent like USE or un-tuned like off-the-shelf mpnet).
        :param additional_transform: this is a function that is called in acquire() - this might be an additional
            transform we want to do to compute the nearest neighbors. It will be passed all object variables and acquire
            args. This is purely for the kNN, so when clf.predict_proba() is used, this transformation is not used.
        :return:
        """
        self.precompute_dist_mat = precompute_dist_mat
        self.LU_dist_mat = None
        self.k = k
        self.embeddings = None
        self.additional_transform = additional_transform

    def acquire(self, X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, **kwargs):
        if self.additional_transform:
            X_L_trf_knn, X_U_trf_knn = self.additional_transform(self, X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size,
                                                                 kwargs)
        else:
            X_L_trf_knn, X_U_trf_knn = X_L_trf, X_U_trf

        if self.precompute_dist_mat:
            if self.LU_dist_mat is None:
                logging.info(f"Computing distance matrix between arrays of sizes {np.shape(X_L)} and {np.shape(X_U)}. "
                             f"This will be computed just once.")
                self.LU_dist_mat = euclidean_distances(X_L_trf_knn, X_U_trf_knn)
        else:  # we need to recompute distance at every function call
            logging.info(f"Computing distance matrix between arrays of sizes "
                         f"{np.shape(X_L_trf_knn)} and {np.shape(X_U_trf_knn)}.")
            self.LU_dist_mat = euclidean_distances(X_L_trf_knn, X_U_trf_knn)

        # this is an additional check for the case the dist. matrix is persisted, ensuring sizes match up per AL iter
        assert np.shape(self.LU_dist_mat)[0] == len(X_L_trf_knn)
        assert np.shape(self.LU_dist_mat)[1] == len(X_U_trf_knn)

        y_L_proba = clf.predict_proba(X_L_trf)
        y_U_proba = clf.predict_proba(X_U_trf)
        s = np.zeros(np.shape(self.LU_dist_mat)[1])
        for c_idx in range(len(X_U)):
            idx_closest = np.argsort(self.LU_dist_mat[:, c_idx])[:self.k]
            neighbour_probs = y_L_proba[idx_closest, :]
            curr_probs = y_U_proba[c_idx,:] #clf.predict_proba([X_U_trf[c_idx]])[0]
            # calculate avg KL div. score and store
            s[c_idx] = sum([sum(kl_div(n, curr_probs)) for n in neighbour_probs])

        # get the ones with the greatest div
        idxs = np.argsort(s)[-size:]

        # adjust the dist. matrix if we want the dist mat to persist
        if self.precompute_dist_mat:
            self.LU_dist_mat = np.delete(self.LU_dist_mat, idxs, axis=1)
            A = X_U_trf_knn[idxs, :]
            B = np.delete(X_U_trf_knn, idxs, axis=0)
            new_dist = euclidean_distances(A, B)
            self.LU_dist_mat = np.vstack((self.LU_dist_mat, new_dist))

        return idxs


