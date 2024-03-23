import os
os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # ,0,1,2,5,6"
os.environ['HF_HOME'] = '/var/tellme/users/enguyen/hf_cache'
from functools import partial
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from core.model_selection import select_model, ModelSearchResult
from core.hf_model_selection import train_using_val as hf_select_model, BERTLike
from acquisition_strategies import acq_random, REAL, CAL, DAL, acq_margin
from core import al
from core.data_utils import load_data
from init_strategies import init_random
import click
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(funcName)s|%(message)s')
logging.info("Logger setup complete.")
f1_macro = partial(f1_score, average='macro')
tmp_bert_dir = './temp_bert_dir'

def get_cls_unlabeled(*args):
    # see code for REAL to see what would be passed
    X_U, clf = args[3], args[6]
    # use CLS representation of the current finetuned model
    X_U_trf = clf.get_cls_output(X_U, is_last_hidden_state=True, is_normalized=False)
    return X_U_trf


def get_cls_all(*args):
    # see code for CAL to see what would be passed
    X_L, X_U, clf = args[1], args[3], args[6]
    # use CLS representation of the current finetuned model
    X_L_trf = clf.get_cls_output(X_L, is_last_hidden_state=True, is_normalized=False)
    X_U_trf = clf.get_cls_output(X_U, is_last_hidden_state=True, is_normalized=False)
    return X_L_trf, X_U_trf


def run_exps(dsname='agnews', seed_size=500, al_batch_size=500, total_budget=5000, qm_list=['random'], train_size=20000,
             test_size=5000, trial_ids=[1], opdir='./output_bert'):
    """
        Run AL experiments.
        :return:
        """
    X_U, y_U, X_test, y_test = load_data(dataset_name=dsname, max_train_instances=train_size, max_test_instances=test_size)
    print(len(X_U), len(X_test))
    
    num_iters = int((total_budget-seed_size)/al_batch_size)
    # trial_ids = [0]
    print(seed_size, num_iters, trial_ids)
    
    for qm in qm_list:
        for trial_idx in trial_ids:
            output_dir_current = os.path.join(opdir,f'{dsname}_bs{al_batch_size}_{qm}')
            if qm == 'real':
                num_clusters = 25
                real_obj = REAL(num_clusters=num_clusters, additional_transform=get_cls_unlabeled)
                qm_obj = real_obj.acquire
            elif qm == 'cal':
                num_neis = 10
                cal_obj = CAL(precompute_dist_mat=False, k=num_neis, additional_transform=get_cls_all)
                qm_obj = cal_obj.acquire
            elif qm == 'dal':
                dal_obj = DAL(additional_transform=get_cls_all) 
                qm_obj = dal_obj.acquire
            elif qm == 'margin':
                qm_obj = acq_margin
            else:
                qm_obj = acq_random
                
            
            output_al_dir = os.path.join(output_dir_current, f'trial_{trial_idx}')
            output_clf_dir = os.path.join(tmp_bert_dir, f'{dsname}_bs{al_batch_size}_{qm}_trial{trial_idx}_ckpts')
            logging.info(f'Start experiment for ds={dsname}, qm={qm}, trial_id={trial_idx}, output_dir={output_dir_current}.')
            al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U), 
                                  X_test=np.array(X_test), y_test=np.array(y_test),
                                  clf_class=BERTLike,
                                  clf_param_grid={'output_dir': [output_clf_dir], 'lr': [3e-5, 5e-5], 'model_name': ['roberta-base'],
                                                  'train_batch_size': [16], 'eval_batch_size': [32], 'num_epochs': [5,10],
                                                  'max_length': [128], 'eval_steps': [('num_evals_per_epoch', 5)],
                                                  'warmup_steps': [0.1], 'save_model': [False], 'max_steps': [-1]},
                                  transform_class=None, trf_param_grid=None,
                                  acq_fn=qm_obj, seed_size=seed_size, batch_size=al_batch_size, num_iters=num_iters,
                                  init_fn=init_random, model_selector=select_model,
                                  model_search_type='val', num_cv_folds=3, val_size=0.2,
                                  model_selector_params={'refit': False, 'fit_needs_val': True},
                                  metric=f1_macro, calibrate=False,
                                  op_dir=output_al_dir, random_state=trial_idx)

@click.command()
@click.option('--dsname', default='agnews', help="name of dataset")
@click.option('--qm', multiple=True, default=["random"], help="list of acquisition functions")
@click.option('--trial', multiple=True, default=[0], help="trial IDs")
@click.option('--bs', default=500, type=int, help="AL batch size")
def runner(dsname, qm, trial, bs):
    qm_list= list(qm) 
    trial_ids = list(trial)
    budget = 5000
    train_size = 20000
    test_size = 5000
    opdir = './output_bert'
    run_exps(dsname=dsname, seed_size=bs, al_batch_size=bs, total_budget=budget, qm_list=qm_list, train_size=train_size,
             test_size=test_size, trial_ids=trial_ids, opdir=opdir)


if __name__ == "__main__":
    runner()
