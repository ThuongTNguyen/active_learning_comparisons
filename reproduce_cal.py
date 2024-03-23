import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # ,0,1,2,5,6"
os.environ['HF_HOME'] = '/var/tellme/users/enguyen/hf_cache'
from functools import partial
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from core.model_selection import select_model, ModelSearchResult
from core.hf_model_selection import train_using_val as hf_select_model, BERTLike
from acquisition_strategies import acq_random, CAL
from core import al
from init_strategies import init_random

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(funcName)s|%(message)s')
logging.info("Logger setup complete.")
f1_macro = partial(f1_score, average='macro')


def reproduce_CAL(dataset_name='agnews', qm='cal'):
    """
        Reproduce results shown in CAL papers.
        :return:
        """

    df_train = pd.read_csv(f'./scratch/{dataset_name}/train.csv')
    X, y = df_train['text'], df_train['label']
    df_test = pd.read_csv(f'./scratch/{dataset_name}/test.csv')
    X_test, y_test = df_test['text'], df_test['label']

    X_U, X_fixed_val, y_U, y_fixed_val = train_test_split(X, y, test_size=0.05, stratify=y)
    print(len(y_U), len(y_fixed_val), len(y_test))

    seed_size = int(len(X_U) * 0.01)
    al_batch_size = int(len(X_U) * 0.02)
    num_iters = 7

    def f(*args):
        # see code for CAL to see what would be passed
        X_L, X_U, clf = args[1], args[3], args[6]
        # use CLS representation of the current finetuned model
        knn_X_L_trf = clf.get_cls_output(X_L, is_last_hidden_state=False, is_normalized=True)
        knn_X_U_trf = clf.get_cls_output(X_U, is_last_hidden_state=False, is_normalized=True)
        return knn_X_L_trf, knn_X_U_trf

    for trial_idx in range(5):       
        if qm == 'cal':
            cal_obj = CAL(precompute_dist_mat=False, k=10, additional_transform=f)
            qm_obj = cal_obj.acquire
    
        else:
            qm_obj = acq_random
        output_dir = f'./rep_cal/{qm}/'
        output_al_dir = os.path.join(output_dir, f'trial_{trial_idx}')
        output_clf_dir = os.path.join(output_dir, f'model_trial_{trial_idx}')

        al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                              X_test=np.array(X_test), y_test=np.array(y_test),
                              clf_class=BERTLike,
                              clf_param_grid={'output_dir': [output_clf_dir], 'lr': [2e-5], 'model_name': ['bert-base-cased'],
                                              'train_batch_size': [16], 'eval_batch_size': [256], 'num_epochs': [3],
                                              'max_length': [128], 'eval_steps': [('num_evals_per_epoch', 5)],
                                              'warmup_steps': [0.1], 'save_model': [False], 'max_steps': [-1]},
                              transform_class=None, trf_param_grid=None,
                              acq_fn=qm_obj, seed_size=seed_size, batch_size=al_batch_size, num_iters=num_iters,
                              init_fn=init_random, model_selector=select_model,
                              model_search_type='val', num_cv_folds=3, val_size=0.2,
                              model_selector_params={'refit': False, 'fit_needs_val': True,
                                                     'val_data': (X_fixed_val, y_fixed_val)},
                              metric=f1_macro, calibrate=False,
                              op_dir=output_al_dir, random_state=trial_idx)


if __name__ == "__main__":
    reproduce_CAL(dataset_name='agnews', qm='cal')

