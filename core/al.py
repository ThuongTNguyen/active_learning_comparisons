import numpy as np
import re, sys, os
from collections import Counter
from functools import partial
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
from collections.abc import Iterable
from matplotlib import pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from core.model_selection import select_model, DummyTransformer
import datetime
import acquisition_strategies as acq_strat
import init_strategies as init_strat
import logging
# logging.getLogger()
f1_macro = partial(f1_score, average='macro')



def move_elements(X_to, y_to, X_from, y_from, idxs):
    """
    Helper function that transfers feature vectors and labels from (X_from, y_from) to (X_to, y_to) given indices to
    transfer.
    :param X_to:
    :param y_to:
    :param X_from:
    :param y_from:
    :param idxs:
    :return:
    """
    X_to = np.concatenate((X_to, X_from[idxs]))
    y_to = np.concatenate((y_to, y_from[idxs]))
    X_from = np.delete(X_from, idxs, axis=0)
    y_from = np.delete(y_from, idxs)
    return X_to, y_to, X_from, y_from


def batch_active_learn(X_L, y_L, X_U, y_U, X_test, y_test,
                       clf_class, clf_param_grid, transform_class, trf_param_grid,
                       acq_fn, seed_size, batch_size, num_iters,
                       init_fn=init_strat.init_random, model_selector=select_model,
                       model_search_type='cv', num_cv_folds=3, val_size=0.8, metric=f1_macro,
                       calibrate=True, cal_size=0.2,
                       op_dir=None,
                       random_state=None, model_selector_params=None):
    """

    :param X_L: can be None, if we want to bootstrap from the unlabelled pool
    :param y_L: needs to be None if X_L is None
    :param X_U:
    :param y_U: we need this for eval, ofc, in real-life these are not known
    :param X_test: test set
    :param y_test:
    :param clf_class:
    :param clf_param_grid: scikit-type param grid for the classifier
    :param transform_class: can be None, if no transformation is required
    :param trf_param_grid: scikit-type param grid for the transformation, can be None if default
        params are to be used.
    :param acq_fn: acquisition function, it should accept current X_L, y_L, X_U, y_U
    :param seed_size: data to be put into the labelled set before beginning the AL loop
    :param batch_size: batch size for AL
    :param num_iters: number of times to active learn, will stop early if we exhaust the unlabeled pool
    :param init_function: how to pick the initial seed data, accepts X_U, y_U. Needs to return indices in X_U.
        TODO: is it possible that we have a non-empty X_L, y_L but we still want a seed set?
    :param model_selector: how to perform model selection at an iteration, the default is model_selection.select_model()
        but this can be a custom function, see demo.py.
    :param model_search_type: 'cv' or 'val' - applies
    :return:
    """
    al_op_file = None
    if op_dir:
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        al_op_file = f"{op_dir}/al.csv"
    logging.info(f"Output directory: {op_dir}")

    # set this to an empty dict otherwise it'll error out later
    if model_selector_params is None:
        model_selector_params = dict()

    df = pd.DataFrame()

    if X_L is None or len(X_L) == 0:
        if len(np.shape(X_U)) == 1:
            X_L = np.array([], dtype='<U1')  # dtype for strings
        else:
            X_L = np.empty((0, np.shape(X_U)[1]))
        y_L = np.array([], dtype='int')
    if seed_size > 0:
        seed_idxs = init_fn(X_U, y_U, seed_size, random_state=random_state)
        X_L, y_L, X_U, y_U = move_elements(X_L, y_L, X_U, y_U, seed_idxs)
        invalid_labels = set([k for k, v in Counter(y_L).items() if v < 2])
        if len(invalid_labels) > 0:
            logging.warning(f"At seed acq. (size={len(X_L)}): labels={invalid_labels} have 1 instance only! "
                            f"All label counts: {Counter(y_L)}")

    logging.info(f"Training initial classifier.")

    res_model_search = model_selector(X=X_L, y=y_L,
                       clf_class=clf_class, clf_param_grid=clf_param_grid,
                       transform_class=transform_class,
                       transform_param_grid=trf_param_grid,
                       calibrate=calibrate, cal_size=cal_size,
                       search_type=model_search_type, metric=metric,
                       num_cv_folds=num_cv_folds, val_size=val_size,
                       random_state=random_state,
                       op_file=f"{op_dir}/model_sel_init.csv" if op_dir else None,
                                      **model_selector_params)

    best_clf, best_trf = res_model_search.clf, res_model_search.trf
    if best_trf is None:
        best_trf = DummyTransformer()

    score = metric(y_test, best_clf.predict(best_trf.transform(X_test)))
    df = df.append({'iter_idx': -1, 'train_size': len(X_L), 'score': score,
                    'acq': acq_fn.__qualname__, 'init': init_fn.__qualname__, 'ts': str(datetime.datetime.now())}, ignore_index=True)
    if al_op_file:
        df.to_csv(al_op_file, index=False)
    logging.info(f"Init classifier trained with test score={score}. Will start AL loop.")

    # at the start we have the clf and trf trained on the initial seed data
    for i in range(num_iters):
        print(f"At AL iter={i+1}/{num_iters} ({100.0*(i+1)/num_iters:.2f}%).")

        # since the acq. fn. doesn't understand transformations, we pass it both the raw and transformed features
        X_L_trf, X_U_trf = best_trf.transform(X_L), best_trf.transform(X_U)
        acquire_idxs = acq_fn(X_L, y_L, X_U,
                              X_L_trf, X_U_trf,
                              best_clf, min(batch_size, len(X_U)), random_state=random_state)
        # move elements from the unlabelled to labelled pool
        X_L, y_L, X_U, y_U = move_elements(X_L, y_L, X_U, y_U, acquire_idxs)

        # after acquisition train a new model on the new seed data
        res_model_search = model_selector(X=X_L, y=y_L, clf_class=clf_class,
                                               clf_param_grid=clf_param_grid,
                                               transform_class=transform_class,
                                               transform_param_grid=trf_param_grid,
                                               search_type=model_search_type,
                                               metric=metric, num_cv_folds=num_cv_folds,
                                               calibrate=calibrate, cal_size=cal_size,
                                               val_size=val_size,
                                               random_state=random_state,
                                               op_file=f"{op_dir}/model_sel_al_{i}.csv" if op_dir else None,
                                               **model_selector_params)
        best_clf, best_trf = res_model_search.clf, res_model_search.trf
        if best_trf is None:
            best_trf = DummyTransformer()

        # use this recent model to predict on the test data
        score = metric(y_test, best_clf.predict(best_trf.transform(X_test)))

        df = df.append({'iter_idx': i, 'train_size': len(X_L), 'score': score,
                        'acq': acq_fn.__qualname__, 'init': init_fn.__qualname__, 'ts': str(datetime.datetime.now())},
                       ignore_index=True)

        # write results  *inside* the loop so we've interim results even if prog. crashes
        if al_op_file:
            df.to_csv(al_op_file, index=False)
        if len(X_U) == 0: # we've exhausted the unlabeled pool
            print(f"Exhausted unlabeled pool, exiting active learning loop.")
            break

    if al_op_file:
        df.to_csv(al_op_file, index=False)

    return df