import numpy as np, pandas as pd, json, copy, time
import itertools, collections, datetime
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from functools import partial
from dataclasses import dataclass
import typing
import logging

f1_macro = partial(f1_score, average='macro')


class DummyTransformer:
    def __init__(self):
        pass

    def fit_transform(self, X, *args, **kwargs):
        return X

    def transform(self, X, *args, **kwargs):
        return X

@dataclass
class ModelSearchResult:
    clf: 'typing.Any'  # classifier
    trf: 'typing.Any' = DummyTransformer()  # transformation
    best_clf_params: 'typing.Any' = None  # classifier params
    best_trf_params: 'typing.Any' = None  # transformation params
    best_score: 'typing.Any' = None  # best held-out score
    res_df: 'typing.Any' = None  # results tabulated in a dataframe

class MyCalibratedClassifierCV(CalibratedClassifierCV):
    """
    Why this subclass? Issue with dealing with strings in CalibrationClassifierCV (we'll call this CCCV) - predict()
    breaks because (a) CCCV calls scikit validations check_array() with dtype='numeric' which forces
    the strings in the data to convert to float (and fails). This can be fixed by calling check_array() with dtype=None
    argument, but (b) the next issue is it also checks for the array to be 2D. Most of these problems arise in the
    predict_proba() method of CCCV, so I am overriding it here to bypass those checks.
    """
    def predict_proba(self, X):
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.predict_proba(X)
            mean_proba += proba

        mean_proba /= len(self.calibrated_classifiers_)

        return mean_proba



def flatten_param_grid(param_grid):
    # unpack the grid dict into a list of dict configs
    keys, values = [], []
    # unpack using a loop to ensure the orders are respectively maintained
    for k, v in param_grid.items():
        keys.append(k)
        values.append(v)
    param_configs = []
    for t in itertools.product(*values):
        param_configs.append(dict(zip(keys, t)))
    return param_configs


def dataset_iter(X, y, search_type, num_cv_folds, val_size, val_data=None, random_state=None):
    """
    A helper function that creates train and test subsets for hold-out validation.
    This cleanly subsumes (a) cross-val and (b) testing with one validation set.
    :param X:
    :param y: can be None, since not all transformations need to look at y
    :param search_type:
    :param num_cv_folds: used if search_type=='cv'
    :param val_size:  used if search_type=='val'
    :param val_set_prepended: Used if search_type=='val', but doesn't create a val set, uses the first
        'val_set_appended' indices from X. This value must be an integer if set.
        This is not recommended for practical AL algorithms becuase we wouldn't typically have access to this kind
        of a val set in the real-world. This option is provided to recreate some specific theoretical scenarios.
    :param random_state:
    :return:
    """
    if search_type == 'cv':
        skf = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_state)
        for train_idx, test_idx in skf.split(X, y):
            yield train_idx, test_idx
    elif search_type == 'val':
        if val_data is None:
            train_idx, test_idx = train_test_split(list(range(len(X))), test_size=val_size, stratify=y,
                                                   random_state=random_state)
            yield train_idx, test_idx
        else:
            logging.warning(f"Here be dragons! This is an unrealistic setup!")
            yield list(range(len(X))), None, val_data


def transform_data(X_train, y_train, X_test, transform_class, transform_param_configs):
    """
    Transformation is abstracted into a function because of multiple cases such as no transformation, transformation
    but with no param grid, transform with its own params to search on.
    :param X_train:
    :param y_train:
    :param X_test: this can be None
    :param transform_class:
    :param transform_param_configs:
    :return: possibly transformed data, current config params, fitted transformer object
    """
    # logging.info(f"size X_train = {np.shape(X_train)}, {type(X_train[0])}, "
    #              f"X_test = {np.shape(X_test)} {type(X_test[0])}")

    X_train_trf, X_test_trf = None, None
    if transform_class is None:
        yield X_train, X_test, None, None
    elif len(transform_param_configs) == 0:
        # use the default params, nothing to iterate over
        trf = transform_class()
        if y_train is not None:
            X_train_trf = trf.fit_transform(X_train, y_train)
        else:
            X_train_trf = trf.fit_transform(X_train)
        if X_test is not None:
            X_test_trf = trf.transform(X_test)
        yield X_train_trf, X_test_trf, None, trf
    else:
        for idx, trf_param in enumerate(transform_param_configs):
            # logging.info(f"(YIELD LOOP) size X_train = {np.shape(X_train)}, {type(X_train[0])}, "
            #              f"X_test = {np.shape(X_test)} {type(X_test[0])}")
            trf = transform_class(**trf_param)
            if y_train is not None:
                X_train_trf = trf.fit_transform(X_train, y_train)
            else:
                X_train_trf = trf.fit_transform(X_train)
            if X_test is not None:
                X_test_trf = trf.transform(X_test)
            yield X_train_trf, X_test_trf, trf_param, trf



def select_model(X, y,
                 clf_class, clf_param_grid,
                 transform_class, transform_param_grid,
                 fit_needs_val=False,
                 search_type='cv', metric=f1_macro,
                 num_cv_folds=5, val_size=0.2,
                 calibrate=False, cal_size=0.2,
                 random_state=None, op_file=None, refit=True,
                 *args, **kwargs):
    """
    This is intended to be a replacement for scikit's utilities, so we don't need to r  ely on cloning of objects,
    which often causes problems with  TF or pytorch. The downside is this doesn't parallelize, and so we would need
    to rely on data-parellelism using "GNU parallel" or something similar.
    :param X: data based on which model selection is to be done
    :param y:
    :param clf_class: this is the classifier, needs to have fit(), predict(), and predict_proba(). This need not
        be a scikit class. Can't be None. Initialized with params from clf_param_grid.
    :param clf_param_grid: the param grid to search for clf_class classifier. This is a dict
        i.e., {'param_1': [val1, val2, ...], 'param_2': ...}. If there're params we need to keep fixed for clf_class,
        use a list with just one entry here.
    :param transform_class: the transformation to be learned and applied on the training data, and then applied on the
        test data. This can be None for cases where clf_class internally transforms the data. Instances from X will
        be supplied to the clf_class.
    :param transform_param_grid:  similar in role to `clf_param_grid` but for transformation.
    :param search_type: strategy for model selection. Currently supported strategies:
        * 'cv': cross validation
        * 'val': using a validation set
    :param fit_needs_val: set this to True if your classifier's fit *requires* a validation set, e.g., BERT, LightGBM. Be
        careful about setting this to True, since all calls to fit now would be passed a val set. Esp. beware of the
        case refit=True because currently we don't support passing a val dataset in this step.
    :param num_cv_folds: used only if search_type='cv'
    :param val_size: fraction to use as hold-out set, used only if search_tye='val'
    :param metric: metric which should be maximized for by the model selection strategy
    :param random_state: use for various splits, can be set to a specific int value for reproducibility
    :param op_file: if not None, this should be a file path where results would be written
    :param refit: if a final model needs to be fit on the entire data, this can be False only search_type='val'
        because then we can return a best model. If search_type='cv', this needs to be True.
    :return:
    """
    # some checks for settings
    if fit_needs_val is True and search_type != 'val':
        logging.error("fit_needs_val can be True only if search_type is 'val'!")
        return

    if (refit is False and search_type == 'cv') or (refit is True and fit_needs_val is True):
        logging.error(f"Combination of refit and search_type is invalid or unsupported, aborting!")
        return

    if random_state is None:
        random_state = int(time.time())

    num_orig_labels = len(set(y))
    logging.info(f"Complete data has {num_orig_labels} labels.")
    # create a result dataframe
    res_df = pd.DataFrame()

    # unpack the grid params into list of dicts - but DO NOT merge the clf and transform configs
    # because of potential namespace conflict, e.g., both models may have a parameter called "epochs".
    clf_param_configs, transform_param_configs = [], []
    for param_grid, param_config in [(clf_param_grid, clf_param_configs), (transform_param_grid,
                                                                           transform_param_configs)]:
        if param_grid:
            param_config += flatten_param_grid(param_grid)

    num_trf_configs, num_clf_configs = len(transform_param_configs), len(clf_param_configs)
    logging.info(f"# transformation configs: {num_trf_configs}.")
    logging.info(f"# classifier configs: {num_clf_configs}.")
    # to calculate the total search space size, watch out for 0-sized lists!
    if num_trf_configs == 0 and num_clf_configs == 0:
        overall_space_size = 0
    else:
        overall_space_size = max(1, num_trf_configs) * max(1, num_clf_configs)
    logging.info(f"Overall config search space size: {overall_space_size}")

    # separate out the calibration data first so there is no leakage
    if calibrate:
        invalid_labels = set([k for k, v in Counter(y).items() if v < 2])
        if len(invalid_labels) > 0:
            logging.warning(f"{invalid_labels} labels have 1 instance only!")
        X_cal, X, y_cal, y = train_test_split(X, y, stratify=y, train_size=cal_size, random_state=random_state)
        logging.info(f"Separate out {cal_size} fraction of the data for calibration.")

    # the first level of iteration must be at the data level so that expensive transformers may be reused
    expt_id = -1
    persist_models = {}  # stores models against param idxs, but is only useful when search_type=='val'
    for ds_idx, t in enumerate(dataset_iter(X, y, search_type=search_type, num_cv_folds=num_cv_folds,
                                                                val_size=val_size,
                                                                val_data=kwargs.get('val_data', None),
                                                                random_state=random_state)):
        train_idx, test_idx, *rest = t
        X_train, y_train = X[train_idx], y[train_idx]
        if len(rest) > 0:
            assert test_idx is None, "dataset_iter() can return more than 3 entries only if test_idx is None!"
            X_test, y_test = rest[0]
        else:
            X_test, y_test = X[test_idx], y[test_idx]

        # It is possible that we're thin-slicing the data so much that not all labels are represented. If so, log a
        # warning for post-hoc analysis.
        iter_num_labels_train, iter_num_labels_test = len(set(y_train)), len(set(y_test))
        if iter_num_labels_train < num_orig_labels:
            logging.warning(f"Current iteration data has {iter_num_labels_train} train labels, whereas there were "
                            f"originally {num_orig_labels} labels.")
        if iter_num_labels_test < num_orig_labels:
            logging.warning(f"Current iteration data has {iter_num_labels_test} test labels, whereas there were "
                            f"originally {num_orig_labels} labels.")

        for trf_idx, (X_train_trf, X_test_trf, trf_param, trf_obj) in enumerate(transform_data(X_train, y_train, X_test,
                                                                        transform_class, transform_param_configs)):
            logging.info("Received transformed data.")
            # now iterate over the params of the classifier
            for clf_idx, clf_param in enumerate(clf_param_configs):
                expt_id += 1
                clf = clf_class(**clf_param)
                if fit_needs_val:
                    clf.fit(X_train_trf, y_train, X_test_trf, y_test)
                else:
                    clf.fit(X_train_trf, y_train)
                y_pred = clf.predict(X_test_trf)
                score = metric(y_test, y_pred)
                persist_models[(clf_idx, trf_idx)] = (clf, trf_obj)
                logging.info(f"expt_id:{expt_id}, trf_param={json.dumps(trf_param)}, clf_param={json.dumps(clf_param)},"
                             f"current val. score: {score:.4f}")

                res_df = res_df.append({'id': expt_id, 'score': score, 'transform params': trf_param,
                                        'classifier params': clf_param,
                                        'ds_idx': ds_idx, 'trf_idx': trf_idx, 'clf_idx': clf_idx,
                                        'ts': str(datetime.datetime.now())},
                                       ignore_index=True)

    # find the best params
    temp_df = res_df.groupby(by=['trf_idx', 'clf_idx'], as_index=False).agg(
        avg_score=pd.NamedAgg(column='score', aggfunc='mean'))
    best_score = max(temp_df['avg_score'])
    best_trf_idx, best_clf_idx = temp_df[temp_df['avg_score']==best_score][['trf_idx', 'clf_idx']].values[0]
    temp_dict_1, temp_dict_2 = dict(res_df[['trf_idx', 'transform params']].values),\
                               dict(res_df[['clf_idx', 'classifier params']].values)
    best_trf_params, best_clf_params = temp_dict_1[best_trf_idx], temp_dict_2[best_clf_idx]
    logging.info(f"Best trf params={best_trf_params}, best clf params={best_clf_params}")

    # refit with the best params
    if refit:
        if calibrate:
            X_final, X_cal_trf, _, trf = \
                next(transform_data(X, y, X_cal, transform_class, [best_trf_params] if best_trf_params else []))
        else:
            X_final, _, _, trf = \
                next(transform_data(X, y, None, transform_class, [best_trf_params] if best_trf_params else []))
        clf = clf_class(**best_clf_params)
        clf.fit(X_final, y)
    else:
        logging.info(f"refit is False, will return the best val model.")
        clf, trf = persist_models[(best_clf_idx, best_trf_idx)]
        if calibrate:
            X_cal_trf, _, _, trf = \
            next(transform_data(X_cal, y_cal, None, transform_class, [best_trf_params] if best_trf_params else []))

    if calibrate:
        logging.info(f"Will calibrate model now.")
        clf = MyCalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        clf.fit(X_cal_trf, y_cal)
    if op_file:
        res_df.to_csv(op_file)

    return ModelSearchResult(clf, trf, best_clf_params, best_trf_params, best_score, res_df)


if __name__ == "__main__":
    # print(flatten_param_grid({'C': [1, 2, 3], 'kernel': ['rbf', 'poly']}))
    for m in  [ModelSearchResult(23, 4), ModelSearchResult('hello')]:
        print(m)