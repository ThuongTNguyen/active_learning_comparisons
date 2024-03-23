# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # ,0,1,2,5,6"
# os.environ['HF_HOME'] = '/var/tellme/users/enguyen/hf_cache'
from functools import partial
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from core.model_selection import select_model, ModelSearchResult
from core.hf_model_selection import train_using_val as hf_select_model, BERTLike
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import acquisition_strategies as acq_strat
from acquisition_strategies import acq_random, CAL, DAL, REAL, acq_margin, acq_entropy
from core import al
from init_strategies import init_random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(funcName)s|%(message)s')
# set up the logger
# formatter = logging.Formatter('%(message)s')
# log = logging.getLogger('')
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# ch.setFormatter(formatter)
# log.addHandler(ch)
logging.info("Logger setup complete.")
f1_macro = partial(f1_score, average='macro')


class MiniLM:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit_transform(self, X, *args, **kwargs):
        emb = self.model.encode(X)
        return emb

    def transform(self, X, *args, **kwargs):
        emb = self.model.encode(X)
        return emb


def model_search_demo1(search_type='cv'):
    '''
    Has a transform and a classifier both their own param grid search
    :param search_type:
    :return:
    '''
    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    select_model(np.array(X), np.array(y),
                 clf_class=LinearSVC, clf_param_grid={'C': [1, 0.01]},
                 transform_class=CountVectorizer, transform_param_grid={'ngram_range': [(1, 1), (1, 3)]},
                 calibrate=True, cal_size=0.2,
                 search_type=search_type, num_cv_folds=3, val_size=0.2, op_file='scratch/model_selection.csv'
                 )


def model_search_demo2(search_type='cv'):
    '''
    Has a transform and a classifier, with only search grid for classifier
    :param search_type:
    :return:
    '''
    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X, _, y, _ = train_test_split(X, y, train_size=500, stratify=y)
    select_model(np.array(X), np.array(y),
                 clf_class=RandomForestClassifier, clf_param_grid={'n_estimators': [5, 10, 30],
                                                                   'min_samples_split': [2, 5]},
                 transform_class=MiniLM, transform_param_grid=None,
                 search_type=search_type, num_cv_folds=3, val_size=0.2, op_file='scratch/model_selection.csv'
                 )

def al_demo_1():
    '''
    Run the whole AL loop with our model search using the common settings.
    :return:
    '''
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    X_U, y_U = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_test, y_test = fetch_20newsgroups(subset='test', categories=categories, return_X_y=True)
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LinearSVC, clf_param_grid={'C': [1, 0.01]},
                          transform_class=CountVectorizer, trf_param_grid={'ngram_range': [(1, 1), (1, 3)]},
                          acq_fn=acq_random, seed_size=20, batch_size=10, num_iters=10,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='cv', num_cv_folds=3, val_size=0.8,
                          metric=f1_macro, op_dir=r'scratch/demo_al', random_state=10)


def al_demo_2():
    '''
    Run the AL loop with a custom model selector, there are just two requirements here:
    * it should accept the named params X, y, metric (a callable), random_state, op_file. Make sure to leave in the
    glob params *args and **kwargs so that other codes doesn't break.
    * it needs to return a ModelSearchResult object. All fields except the clf field can be None.
    This setting should be used with models like BERT where we might HF to train. This particular example shows it
    with a scikit classifier.
    :return:
    '''

    def custom_model_search(X, y, metric=f1_macro, random_state=None, op_file=None, *args, **kwargs):
        # note we can ignore as many params we want, like random_state
        clf = Pipeline([('cvec', CountVectorizer(ngram_range=(1, 3))), ('linsvc', LinearSVC())])
        grid_clf = GridSearchCV(clf, param_grid={'linsvc__C': [0.1, 1., 10.]}, scoring=make_scorer(metric), cv=3)
        grid_clf.fit(X, y)
        return ModelSearchResult(grid_clf.best_estimator_)

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    X_U, y_U = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_test, y_test = fetch_20newsgroups(subset='test', categories=categories, return_X_y=True)

    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=None, clf_param_grid=None,
                          transform_class=None, trf_param_grid=None,
                          acq_fn=acq_random, seed_size=20, batch_size=10, num_iters=10,
                          init_fn=init_random, model_selector=custom_model_search,
                          model_search_type=None, num_cv_folds=None, val_size=None,
                          metric=f1_macro, op_dir=r'scratch/demo_al')


def al_demo_3():
    """
    Shows BERT fine-tuning with custom model selection function.
    """
    dataset_name = '20ng'

    if dataset_name == 'sst2':
        train_file = 'scratch/SST2_train.tsv'
        df = pd.read_csv(train_file, sep='\t')
        X, y = df['sentence'].to_numpy(), df['label'].to_numpy()
        X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    if dataset_name == '20ng':
        categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
        X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
        X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    # Here too we use a custom model selector. This time we also use the 'model_selector_params'
    # variable to pass in arguments for the fine-tuning code
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=None, clf_param_grid=None,
                          transform_class=None, trf_param_grid=None,
                          acq_fn=acq_random, seed_size=20, batch_size=10, num_iters=5,
                          init_fn=init_random, model_selector=hf_select_model,
                          model_search_type=None, num_cv_folds=None, val_size=0.2,
                          metric=f1_macro, op_dir=r'scratch/demo_al',
                          model_selector_params={'output_dir': 'scratch/demo_bert', 'save_model': True,
                                                 'max_steps': 2})


def al_demo_4():
    """
    Shows BERT fine-tuning with model_selection.select_model().
    :return:
    """
    dataset_name = '20ng'

    if dataset_name == 'sst2':
        train_file = 'scratch/SST2_train.tsv'
        df = pd.read_csv(train_file, sep='\t')
        X, y = df['sentence'].to_numpy(), df['label'].to_numpy()
        X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    if dataset_name == '20ng':
        categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
        X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
        X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    # We'll use select_model() with val. Note we still need model model_selector_params to pass in some specific values.
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=BERTLike,
                          clf_param_grid={'output_dir': ['scratch/demo_bert'], 'lr': [5e-3, 1e-1],
                                          'save_model': [True],'max_steps': [2]},
                          transform_class=None, trf_param_grid=None,
                          acq_fn=acq_random, seed_size=20, batch_size=10, num_iters=5,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='val', num_cv_folds=None, val_size=0.2,
                          model_selector_params={'refit': False, 'fit_needs_val': True},
                          metric=f1_macro, op_dir=r'scratch/demo_al')


def al_demo_cal_1():
    """
        Shows use of the CAL acq. fn.
        :return:
        """
    categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    cal_obj = CAL(precompute_dist_mat=False, k=5)
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LogisticRegression,
                          clf_param_grid={'C': [0.1, 1, 10]},
                          transform_class=MiniLM, trf_param_grid=None,
                          acq_fn=cal_obj.acquire, seed_size=20, batch_size=10, num_iters=3,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='cv', num_cv_folds=3, val_size=0.2,
                          metric=f1_macro, op_dir=r'scratch/demo_al')


def al_demo_cal_2():
    """
        Shows use of the CAL acq. fn. with custom transformation for the kNN step.
        :return:
        """
    categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    def f(*args):
        # see code for CAL to see what would be passed
        containing_obj, X_L_trf, X_U_trf = args[0], args[4], args[5]
        # the following transformation obviously makes no sense, but shows how the containing object attributes, like
        # the number of nearest neighbors, can be accessed. For something meaningful,
        # you can use a different embedding here.
        new_X_L_trf = X_L_trf + containing_obj.k
        new_X_U_trf = X_U_trf + containing_obj.k
        return new_X_L_trf, new_X_U_trf

    cal_obj = CAL(precompute_dist_mat=True, k=3, additional_transform=f)
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LogisticRegression,
                          clf_param_grid={'C': [0.1, 1, 10]},
                          transform_class=MiniLM, trf_param_grid=None,
                          acq_fn=cal_obj.acquire, seed_size=20, batch_size=10, num_iters=3,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='cv', num_cv_folds=3, val_size=0.2,
                          metric=f1_macro, op_dir=r'scratch/demo_al')


def al_demo_cal_3():
    """
        Shows use of the CAL acq. fn. with custom transformation for the kNN step.
        Also this one uses a fixed val set for model selection.
        :return:
        """
    categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=200, stratify=y)
    X_fixed_val, X_test, y_fixed_val, y_test = train_test_split(X_test, y_test, train_size=0.5, stratify=y_test)

    def f(*args):
        # see code for CAL to see what would be passed
        containing_obj, X_L_trf, X_U_trf = args[0], args[4], args[5]
        # In this example, since the transform class is None, string arrays are passed in. Since this is just a demo,
        # we'll convert them into random 10-dim. vectors.
        new_X_L_trf = np.random.random((len(X_L_trf), 10))
        new_X_U_trf = np.random.random((len(X_U_trf), 10))
        return new_X_L_trf, new_X_U_trf

    cal_obj = CAL(precompute_dist_mat=True, k=3, additional_transform=f)
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=BERTLike,
                          clf_param_grid={'output_dir': ['scratch/demo_bert'], 'lr': [5e-3],
                                          'save_model': [True], 'max_steps': [2]},
                          transform_class=None, trf_param_grid=None,
                          acq_fn=cal_obj.acquire, seed_size=20, batch_size=10, num_iters=3,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='val', num_cv_folds=3, val_size=0.2,
                          model_selector_params={'refit': False, 'fit_needs_val': True,
                                                 'val_data': (X_fixed_val, y_fixed_val)},
                          metric=f1_macro, op_dir=r'scratch/demo_al')

def al_demo_dal():
    """
        Shows use of the DAL acq. fn.
        :return:
        """
    categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    dal_obj = DAL() 
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LogisticRegression,
                          clf_param_grid={'C': [0.1, 1]},
                          transform_class=CountVectorizer, trf_param_grid=None,
                          acq_fn=dal_obj.acquire, seed_size=20, batch_size=10, num_iters=3,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='cv', num_cv_folds=3, val_size=0.2,
                          metric=f1_macro, op_dir=r'scratch/demo_al')


def al_demo_real_1():
    """
        Shows use of the REAL acq. fn.
        :return:
        """
    categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=100, test_size=100, stratify=y)

    real_obj = REAL(num_clusters=5, additional_transform=None)
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LogisticRegression,
                          clf_param_grid={'C': [0.1, 1]},
                          transform_class=MiniLM, trf_param_grid=None,
                          acq_fn=real_obj.acquire, seed_size=20, batch_size=10, num_iters=3,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='cv', num_cv_folds=3, val_size=0.2,
                          metric=f1_macro, op_dir=r'scratch/demo_al')


def al_demo_real_2():
    """
        Shows use of the REAL acq. fn.
        :return:
        """
    categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    X, y = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_U, X_test, y_U, y_test = train_test_split(X, y, train_size=200, test_size=100, stratify=y)

    def f(*args):
        # see code for CAL to see what would be passed
        X_L, X_U, clf = args[1], args[3], args[6]
        # use CLS representation of the current finetuned model
        X_U_trf_kmeans = clf.get_cls_output(X_U, is_last_hidden_state=True, is_normalized=True)
        return X_U_trf_kmeans

    real_obj = REAL(num_clusters=5, additional_transform=f)
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=BERTLike,
                          clf_param_grid={'output_dir': ['scratch/demo_real'], 'lr': [5e-3], #'model_name': ['roberta-base'],
                                          'save_model': [False], 'max_steps': [2]},
                          transform_class=None, trf_param_grid=None,
                          acq_fn=real_obj.acquire, seed_size=20, batch_size=10, num_iters=3,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='val', num_cv_folds=3, val_size=0.2,
                          model_selector_params={'refit': False, 'fit_needs_val': True},
                          metric=f1_macro, op_dir=r'scratch/demo_al')

def al_demo_uncertainty():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    X_U, y_U = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
    X_test, y_test = fetch_20newsgroups(subset='test', categories=categories, return_X_y=True)

    print(f"\n\n===================\nAcquisition function: margin")
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LinearSVC, clf_param_grid={'C': [1, 0.01]},
                          transform_class=CountVectorizer, trf_param_grid=None,
                          acq_fn=acq_margin, seed_size=20, batch_size=10, num_iters=10,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='val', num_cv_folds=3, val_size=0.2,
                          model_selector_params={'refit': True},
                          metric=f1_macro, op_dir=r'scratch/demo_al', random_state=10)

    print(f"\n\n===================\nAcquisition function: entropy")
    al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                          X_test=np.array(X_test), y_test=np.array(y_test),
                          clf_class=LinearSVC, clf_param_grid={'C': [1, 0.01]},
                          transform_class=CountVectorizer, trf_param_grid={'ngram_range': [(1, 3)]},
                          acq_fn=acq_entropy, seed_size=20, batch_size=10, num_iters=10,
                          init_fn=init_random, model_selector=select_model,
                          model_search_type='cv', num_cv_folds=3, val_size=0.2,
                          metric=f1_macro, op_dir=r'scratch/demo_al', random_state=10)


if __name__ == "__main__":
    # model_search_demo1('cv')
    # model_search_demo2('cv')
    # al_demo_1()
    # al_demo_2()
    # al_demo_3()
    # al_demo_4()
    # al_demo_cal_1()
    # al_demo_cal_2()

    # al_demo_cal_3()

    # al_demo_cal_3()
    al_demo_dal()
    # al_demo_real_1()
    # al_demo_real_2()
    # al_demo_uncertainty()

