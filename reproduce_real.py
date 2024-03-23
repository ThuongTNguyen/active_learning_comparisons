import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # ,0,1,2,5,6"
os.environ['HF_HOME'] = '/var/tellme/users/enguyen/hf_cache'
from functools import partial
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from core.model_selection import select_model, ModelSearchResult
from core.hf_model_selection import train_using_val as hf_select_model, BERTLike
from acquisition_strategies import acq_random, REAL
from core import al
from init_strategies import init_random

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(funcName)s|%(message)s')
logging.info("Logger setup complete.")
f1_macro = partial(f1_score, average='macro')


def reproduce_REAL(dataset_name='agnews', qm='real'):
    """
        Reproduce results shown in REAL papers.
        :return:
        """
    path_all = '../ECML_PKDD_23_Real/data355/agnews/'
    # for name in ['test', 'train', 'valid']:
    #     path = os.path.join(path_all, f'{name}.json')
    #     all_rows = []
    #     with open(path, 'r') as f:
    #         data = f 
    #         for x in data:
    #             all_rows.append(json.loads(x))
    
    #     df = pd.DataFrame.from_records(all_rows)
    #     df = df[['txt','lbl']]
    #     df.columns = ['text', 'label']
    #     df.to_csv(os.path.join(path_all,f'{name}.csv'), index=False)
                

    
    df_train = pd.read_csv(os.path.join(path_all,'train.csv'))
    df_test = pd.read_csv(os.path.join(path_all,'test.csv'))
    df_val = pd.read_csv(os.path.join(path_all,'valid.csv'))
    X, y = df_train['text'], df_train['label']
    X_test, y_test = df_test['text'], df_test['label']
    X_fixed_val, y_fixed_val = df_val['text'], df_val['label']

    X_U, y_U = X, y
    print(len(y_U), len(y_fixed_val), len(y_test))

    seed_size = 100
    al_batch_size = 150
    num_iters = 8
    num_clusters = 25

    def f(*args):
        # see code for REAL to see what would be passed
        X_L, X_U, clf = args[1], args[3], args[6]
        # use CLS representation of the current finetuned model
        X_U_trf_kmeans = clf.get_cls_output(X_U, is_last_hidden_state=True, is_normalized=False)
        return X_U_trf_kmeans

    for trial_idx in range(5):
        if qm == 'real':
            real_obj = REAL(num_clusters=num_clusters, additional_transform=f)
            qm_obj = real_obj.acquire
    
        else:
            qm_obj = acq_random
        output_dir = f'./rep_real/{qm}/'
        output_al_dir = os.path.join(output_dir, f'trial_{trial_idx}')
        output_clf_dir = os.path.join(output_dir, f'model_trial_{trial_idx}')
        al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U), 
                              X_test=np.array(X_test), y_test=np.array(y_test),
                              clf_class=BERTLike,
                              clf_param_grid={'output_dir': [output_clf_dir], 'lr': [2e-5], 'model_name': ['roberta-base'],
                                              'train_batch_size': [8], 'eval_batch_size': [512], 'num_epochs': [4],
                                              'max_length': [96], 'eval_steps': [('num_evals_per_epoch', 4)],
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
    reproduce_REAL(dataset_name='agnews', qm='random')


