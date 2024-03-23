import logging
import os.path
import os, json
import shutil
from pathlib import Path
import glob2, glob

os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numexpr as ne
LOGFILE = f'scratch/logfiles/{os.getpid()}_logfile.txt'

def setup_logger(logfile=LOGFILE):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(process)d|%(levelname)s|%(funcName)s|%(message)s',
                        handlers=[stream_handler, logging.FileHandler(logfile, 'w')])

    logging.info("Logger set up.")
setup_logger()

import collections
import datetime, sys, traceback
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, auc
from core.model_selection import select_model, ModelSearchResult
# from core.hf_model_selection import train_using_val as hf_select_model, BERTLike
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from acquisition_strategies import acq_random, CAL, acq_margin, acq_entropy, DAL, REAL
from core import al
from core import data_utils as du
from init_strategies import init_random
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import spacy

import tensorflow as tf
import tensorflow_hub as hub
USE_module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"


def setup_logger(logfile=LOGFILE):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(process)d|%(levelname)s|%(funcName)s|%(message)s',
                        handlers=[stream_handler, logging.FileHandler(logfile, 'w')])

    logging.info("Logger set up.")
setup_logger()

class MyLightGBM(lgb.LGBMClassifier):
    def fit(self, X_train, y_train, X_val, y_val):
        """This version accepts a val set, in a way required by model_selection.select_model"""
        super().fit(X_train, y_train, eval_set=[(X_val, y_val)])

class MyCountVectorizer(CountVectorizer):
    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y).astype(dtype=np.float32).toarray()
    def transform(self, raw_documents):
        return super().transform(raw_documents).astype(dtype=np.float32).toarray()

f1_macro = partial(f1_score, average='macro')


class USE:
    def __init__(self):
        self.model = hub.load(USE_module_url)

    def fit_transform(self, X, *args, **kwargs):
        emb = np.array(self.model(X))
        return emb

    def transform(self, X, *args, **kwargs):
        emb = np.array(self.model(X))
        return emb


class SentTransformer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, X, *args, **kwargs):
        emb = self.model.encode(X, batch_size=100)
        return emb

    def transform(self, X, *args, **kwargs):
        emb = self.model.encode(X)
        return emb


class SpacyWordVecs:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')

    def avg_wordvecs(self, X_text):
        X_wv = None
        for i, s in enumerate(X_text):
            doc = self.nlp(str(s))
            if X_wv is None:
                dims = len(doc.vector)
                X_wv = np.empty((len(X_text), dims))
            X_wv[i, :] = doc.vector
        return X_wv

    def fit_transform(self, X, *args, **kwargs):
        emb = self.avg_wordvecs(X)
        return emb

    def transform(self, X, *args, **kwargs):
        emb = self.avg_wordvecs(X)
        return emb


def worker(t):
    try:
        al.batch_active_learn(**t)
    except:
        # without this errors in processes will disappear
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.info(''.join('!! ' + line for line in lines))

def get_formatter(a, precision):
    s = f"{a:.{precision}f}"
    # get unique digits after decimal
    t = set(list(s.split(".")[-1]))
    if len(t)==1 and list(t)[0]=='0':
        # we are only able to show zeros
        if float(s) < a:
            return f"{precision}E"
    return f"{precision}f"


def process_expt_results(expt_name, expt_output_file):
    # one plot per daatset
    output_base_path = "scratch/"
    height, width = 8, 14
    random_acq_name = 'random'
    random_acq_fn_name = 'acq_random'
    actual_acq_fn_name = 'CAL.acquire'
    presentable_names = {'acq_random': 'random', 'CAL.acquire': 'CAL'}
    score_display_precision = 2 # beyond this we'll switch to scientific notation
    if type(expt_output_file) == str:
        expt_output_file = [expt_output_file]
    # df = pd.concat([pd.read_csv(f) for f in expt_output_file])
    all_file_dfs = []
    for f in expt_output_file:
        f_dir = os.path.dirname(f)
        temp_df = pd.read_csv(f)
        temp_df['op_dir'] = [f"{f_dir}{os.sep}{i}" for i in temp_df['op_dir']]
        all_file_dfs.append(temp_df)
    df = pd.concat(all_file_dfs)

    clfs, trfs, acqs = sorted(set(df['clf'])), sorted(set(df['trf'])), sorted(set(df['acq']))
    if len(acqs) !=2:
        print(f"# acq. functions should be exactly 2, one random, one sthg else, here it is {acqs}. Aborting!")
        return
    if random_acq_name not in acqs:
        print(f"One of the acq. functions must be {random_acq_name}! Here we only have {acqs}. Aborting!")
        return

    for ds in set(df['dataset']):
        auc_scores = collections.defaultdict(list)
        fig, axes = plt.subplots(nrows=len(clfs), ncols=len(trfs), figsize=(width * len(trfs), height * (len(clfs))))
        print(f"Processing dataset {ds}.")
        ds_df = df[df['dataset'] == ds]

        for (c_idx, c), (t_idx, t) in itertools.product(enumerate(clfs), enumerate(trfs)):
            temp_df = ds_df.query(f'clf=="{c}" and trf=="{t}"')
            # not all combinations might be present, e.g., BERT
            if len(temp_df) == 0:
                continue
            ax = axes[c_idx, t_idx]

            ax.set_xlabel("# instances"); ax.set_ylabel("F1-macro")
            ax.set_ylim(0, 1)

            # TODO: do we drop scores for the seed set? (yes for now)

            specific_res_df = pd.concat([pd.read_csv(f"{r['op_dir']}/al.csv") for _, r in temp_df.iterrows()])
            specific_res_df = specific_res_df[specific_res_df['iter_idx'] != -1]
            acq_fn_names = set(specific_res_df['acq'])
            res_aggr = specific_res_df.groupby(by=['acq', 'train_size'], as_index=False).agg(
                avg_score=pd.NamedAgg(column='score', aggfunc='mean'))
            label_auc_map = {}
            for curr_acq in acq_fn_names:
                x, y = res_aggr[res_aggr['acq'] == curr_acq]['train_size'], \
                       res_aggr[res_aggr['acq'] == curr_acq]['avg_score']
                temp_auc = auc(x, y)
                auc_scores[curr_acq].append(temp_auc)
                label_auc_map[curr_acq] = f"{presentable_names[curr_acq]}, AUC={temp_auc:.2f}"

            scores_actual_acq = np.array(specific_res_df[specific_res_df['acq']==actual_acq_fn_name].
                                         sort_values(by='train_size')['score'])
            scores_random_acq = np.array(specific_res_df[specific_res_df['acq']==random_acq_fn_name].
                                         sort_values(by='train_size')['score'])
            _, local_p_val = wilcoxon(scores_actual_acq-scores_random_acq, alternative='greater')
            _, local_p_val_t = ttest_rel(scores_actual_acq, scores_random_acq, alternative='greater')

            specific_res_df['acq'] = specific_res_df['acq'].map(label_auc_map)
            sns.lineplot(specific_res_df, x='train_size', y='score', hue='acq', ax=ax, marker='o')

            ax.legend()
            fmt = get_formatter(local_p_val, 2)
            fmt_t = get_formatter(local_p_val_t, 2)
            ax.set_title(f"{t}$ \\rightarrow ${c}, F1-macro, $p_W$  = {local_p_val:.{fmt}}, "
                         f"$p_t$  = {local_p_val_t:.{fmt_t}}", size=20)

        # add texts to clarify transforms and classifiers
        for trf_idx, trf in enumerate(trfs):
            x_left, y_bottom, x_right, y_top = axes[0, trf_idx].get_position().bounds
            fig.text(x_left, 0.98, trf, size=40, rotation=0,
                     bbox=dict(boxstyle="Round, pad=0.2",
                               ec="none",
                               fc="wheat", alpha=0.8
                               ))
        for clf_idx, clf in enumerate(clfs):
            x_left, y_bottom, x_right, y_top = axes[clf_idx, 0].get_position().bounds
            fig.text(x_left-0.05, y_bottom+0.15, clf, size=40, rotation=90,
                     bbox=dict(boxstyle="Round, pad=0.2",
                               ec="none",
                               fc="wheat", alpha=0.8
                               ))

        plt.subplots_adjust(top=0.95)

        d = np.array(auc_scores[actual_acq_fn_name]) - np.array(auc_scores[random_acq_fn_name])
        _, p_val = wilcoxon(d, alternative='greater')
        _, p_val_t = ttest_rel(auc_scores[actual_acq_fn_name], auc_scores[random_acq_fn_name], alternative='greater')
        print(f"dataset={ds}, Wilcoxon, non-random is better, p_val ={p_val}\nNote: smaller values "
              f"indicate the acq. fn. is much better than random.")

        fmt = get_formatter(p_val, 2)
        fmt_t = get_formatter(p_val_t, 2)
        fig.suptitle(f"{expt_name}, dataset={ds}, AUC: $p_W$ = {p_val:.{fmt}}, "
                     f"$p_t$ = {p_val_t:.{fmt_t}}", verticalalignment='bottom', y=1.05, size=40)
        fig.savefig(f"{output_base_path}/{expt_name}_{ds}_comparison_plots.png", bbox_inches='tight', pad=0.1)


def run_expts():
    # set overall details here
    num_processes, num_trials = 4, 2
    seed_size, batch_size, num_al_iters = 200, 100, 2
    num_unlabeled, num_test = 1000, 5000
    acq_fns = ['random', 'cal', 'margin', 'dal']
    datasets = ['sst2']#, 'sst2', 'dbpedia', 'agnews', 'imdb', 'pubmed']
    # the last entry is for transform params but in this case, this is only applicable to cvec, and here too,
    # we're just going to use ngrams from 1-3 sizes.
    transforms = [('wordvecs', SpacyWordVecs, None),
                    #('MiniLM', 'all-MiniLM-L6-v2', None)]#,
                  ('MPNet', 'all-mpnet-base-v2', None),
                  ('USE', None, None)]

    expt_output_base_path = f"scratch/cal_seed_{seed_size}_batch_size_{batch_size}_iters_{num_al_iters}"
    if not os.path.exists(expt_output_base_path) or not os.path.isdir(expt_output_base_path):
        os.makedirs(expt_output_base_path)

    # this is the file we'll use to make our plots
    df_expt_details, expt_details_file = pd.DataFrame(), f"{expt_output_base_path}/expt_details.csv"
    # for lightgbm we want to run a single process job - so that it doesn't interfere with high-level parallelization
    models = [
        ('LinearSVC', LinearSVC, {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                    'class_weight': ['balanced']}),
        # ('GBM', MyLightGBM, {'class_weight': ['balanced'], 'max_depth': [3, 5], 'learning_rate': [0.001, 0.01, 0.1],
        #                      'n_estimators': [100], 'min_child_samples': [2, 5], 'early_stopping_round': [3],
        #                      'n_jobs': [1]}),
        ('RF', RandomForestClassifier, {'max_features': ['sqrt'], 'class_weight': ['balanced'],
                       'min_samples_leaf': list(range(1, 10, 4)),
                       'n_estimators': [5] + list(range(10, 51, 10)),
                       'max_depth': [5] + list(range(10, 31, 5))})]

    # preload all data here, and transform them since they can be precomputed in in the non-BERT-like cases
    preloaded_data_trf, preloaded_data = {}, {}
    logging.info(f"We'll preload all the transformed data first.")
    for idx, (ds_name, (trf_name, trf_class, trf_params)) in enumerate(itertools.product(datasets, transforms)):
        if ds_name not in preloaded_data:
            X_train, y_train, X_test, y_test = du.load_data(ds_name, max_train_instances=num_unlabeled,
                                                        max_test_instances=num_test)
            preloaded_data[ds_name] = (X_train, y_train, X_test, y_test)
            logging.info(f"Preloaded data={ds_name}, shape train={np.shape(X_train)}, shape test={np.shape(X_test)}.")
        if (ds_name, trf_name) not in preloaded_data_trf:
            print(f"Preloading data for {(ds_name, trf_name)} combination.")
            X_train, y_train, X_test, y_test = preloaded_data[ds_name]
            if 'minilm' in trf_name.lower() or 'mpnet' in trf_name.lower():
                st_obj = SentTransformer(trf_class)
                X_train_trf = st_obj.fit_transform(X_train)
                X_test_trf = st_obj.fit_transform(X_test)
                del st_obj
            elif 'use' == trf_name.lower():
                use_obj = USE()
                X_train_trf = use_obj.fit_transform(X_train)
                X_test_trf = use_obj.fit_transform(X_test)
                del use_obj
            elif 'ngrams' == trf_name.lower():
                cvec = MyCountVectorizer(ngram_range=(1, 3))
                X_train_trf = cvec.fit_transform(X_train)
                X_test_trf = cvec.transform(X_test)
            elif 'wordvecs' == trf_name.lower():
                sp = SpacyWordVecs()
                X_train_trf = sp.fit_transform(X_train)
                X_test_trf = sp.fit_transform(X_test)
                del sp
            else:
                logging.error(f"Cannot understand transform {trf_name}, aborting!")
                return
            preloaded_data_trf[(ds_name, trf_name)] = (X_train_trf, y_train, X_test_trf, y_test)
            print(f"Preloaded data for {(ds_name, trf_name)} combination, shape train={np.shape(X_train_trf)}, "
                  f"shape test={np.shape(X_test_trf)}")

    all_kwargs = []
    for idx, (trial_idx, ds_name, (model_name, model_class, model_param_grid), (trf_name, trf_class, trf_param_grid), acq_name) in \
            enumerate(itertools.product(range(num_trials), datasets, models, transforms, acq_fns)):
        op_dir = f"{expt_output_base_path}/{ds_name}_{model_name}_{trf_name}_{acq_name}_trial_{trial_idx}"
        print(f"overall id={idx}, trial idx={trial_idx}, settings: {(ds_name, model_name, trf_name)}, op_dir={op_dir}")
        df_expt_details = df_expt_details.append(
            {'trial_idx': trial_idx, 'dataset': ds_name, 'clf': model_name, 'trf': trf_name, 'acq': acq_name,
             'op_dir': os.path.basename(op_dir), 'ts': datetime.datetime.now()},
            ignore_index=True)
        X_U, y_U, X_test, y_test = preloaded_data_trf[(ds_name, trf_name)]

        if acq_name == 'random':
            acq_fn = acq_random
        elif acq_name == 'cal':
            cal_obj = CAL(precompute_dist_mat=True, k=10, additional_transform=None)
            acq_fn = cal_obj.acquire
        elif acq_name == 'margin':
            acq_fn = acq_margin
        elif acq_name == 'entropy':
            acq_fn = acq_entropy
        elif acq_name == 'dal':
            dal = DAL()
            acq_fn = dal.acquire
        elif acq_name == 'real':
            real = REAL(num_clusters=25)
            acq_fn = real.acquire
        else:
            logging.error(f"Can't understand acq. fn. {acq_name}, aborting!")
            return

        kwargs = {'X_L':None, 'y_L':None, 'X_U':np.array(X_U), 'y_U':np.array(y_U),
                          'X_test': np.array(X_test), 'y_test': np.array(y_test),
                          'clf_class': model_class, 'clf_param_grid': model_param_grid,
                          'transform_class': None, 'trf_param_grid': None,
                          'acq_fn': acq_fn, 'seed_size': seed_size, 'batch_size': batch_size, 'num_iters': num_al_iters,
                          'init_fn': init_random, 'model_selector': select_model,
                          'model_search_type': 'val', 'num_cv_folds': 3, 'val_size': 0.2,
                          'calibrate': False if model_name=='LR' or acq_name=='random' else True,
                          'cal_size': 0.2,
                          'metric': f1_macro, 'op_dir': op_dir,
                          'model_selector_params': {"refit": False, "fit_needs_val": True
                          if model_name=="GBM" else False},
                          'random_state': trial_idx}

        all_kwargs.append(kwargs)

    logging.info(f"Beginning experiments with multiple threads.")
    with Pool(num_processes) as p:
        print(p.map(worker, all_kwargs))

    df_expt_details.to_csv(expt_details_file, index=False)


def run_expt_nomp(config_file):
    """
    Doesnt use multiprocessing at all - parallelization should be done at code call level. This is to avoid weird
    library interactions.
    :param config:
    :return:
    """
    with open(config_file) as fr:
        config = json.loads(fr.read())
    # set overall details here
    supported_transforms = {'wordvecs': ('wordvecs', SpacyWordVecs, None),
                            'MPNet': ('MPNet', 'all-mpnet-base-v2', None),
                            'USE': ('USE', None, None)}

    num_trials = config['num_trials']
    seed_size, batch_size, num_al_iters = config['seed_size'], config['batch_size'], config['num_al_iters']
    num_unlabeled, num_test = config['num_unlabeled'], config['num_test']
    acq_fns = config['acq_fns']
    datasets = config['datasets']  # , 'sst2', 'dbpedia', 'agnews', 'imdb', 'pubmed']
    transforms = [supported_transforms[ct] for ct in config['transforms']]
    expt_output_base_path = f"scratch/seed_{seed_size}_batch_size_{batch_size}_iters_{num_al_iters}"

    if not os.path.exists(expt_output_base_path) or not os.path.isdir(expt_output_base_path):
        os.makedirs(expt_output_base_path)

    # this is the file we'll use to make our plots
    df_expt_details, expt_details_file = pd.DataFrame(), f"{expt_output_base_path}/expt_details.csv"

    models = [
        ('LinearSVC', LinearSVC, {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                  'class_weight': ['balanced']}),
        ('RF', RandomForestClassifier, {'max_features': ['sqrt'], 'class_weight': ['balanced'],
                                        'min_samples_leaf': list(range(1, 10, 4)),
                                        'n_estimators': [5] + list(range(10, 51, 10)),
                                        'max_depth': [5] + list(range(10, 31, 5))})]


    for idx, (
    trial_idx, ds_name, (model_name, model_class, model_param_grid), (trf_name, trf_class, trf_param_grid), acq_name) in \
            enumerate(itertools.product(range(num_trials), datasets, models, transforms, acq_fns)):
        op_dir = f"{expt_output_base_path}/{ds_name}_{model_name}_{trf_name}_{acq_name}_trial_{trial_idx}"
        print(f"overall id={idx}, trial idx={trial_idx}, settings: {(ds_name, model_name, trf_name)}, op_dir={op_dir}")
        df_expt_details = df_expt_details.append(
            {'trial_idx': trial_idx, 'dataset': ds_name, 'clf': model_name, 'trf': trf_name, 'acq': acq_name,
             'op_dir': os.path.basename(op_dir), 'ts': datetime.datetime.now()},
            ignore_index=True)
        stored_data_file_path = config['data'][ds_name][trf_name]
        X_U, y_U, X_test, y_test = load_dataset(stored_data_file_path,
                                                expected_train_size=config['data'][ds_name].get('train_size', None),
                                                expected_test_size=config['data'][ds_name].get('test_size', None))

        if acq_name == 'random':
            acq_fn = acq_random
        elif acq_name == 'cal':
            cal_obj = CAL(precompute_dist_mat=True, k=10, additional_transform=None)
            acq_fn = cal_obj.acquire
        elif acq_name == 'margin':
            acq_fn = acq_margin
        elif acq_name == 'entropy':
            acq_fn = acq_entropy
        elif acq_name == 'dal':
            dal = DAL()
            acq_fn = dal.acquire
        elif acq_name == 'real':
            real = REAL(num_clusters=25)
            acq_fn = real.acquire
        else:
            logging.error(f"Can't understand acq. fn. {acq_name}, aborting!")
            return

        kwargs = {'X_L': None, 'y_L': None, 'X_U': np.array(X_U), 'y_U': np.array(y_U),
                  'X_test': np.array(X_test), 'y_test': np.array(y_test),
                  'clf_class': model_class, 'clf_param_grid': model_param_grid,
                  'transform_class': None, 'trf_param_grid': None,
                  'acq_fn': acq_fn, 'seed_size': seed_size, 'batch_size': batch_size, 'num_iters': num_al_iters,
                  'init_fn': init_random, 'model_selector': select_model,
                  'model_search_type': 'val', 'num_cv_folds': 3, 'val_size': 0.2,
                  'calibrate': False if model_name == 'LR' or acq_name == 'random' else True,
                  'cal_size': 0.2,
                  'metric': f1_macro, 'op_dir': op_dir,
                  'model_selector_params': {"refit": False, "fit_needs_val": True
                  if model_name == "GBM" else False},
                  'random_state': trial_idx}

        al.batch_active_learn(**kwargs)
    #logging.info(f"Finished all runs!!!")


def generate_config_json(op_dir, preprocessed_data_dir, script_dir, logfile_dir,
                         load_data=True, python_path='python'):
    script_path = f"{script_dir}/runner.sh"
    f_script = open(script_path, 'w')
    data_locs = dict()
    for work_dir in [op_dir, preprocessed_data_dir, logfile_dir]:
        if not os.path.exists(work_dir) or not os.path.isdir(work_dir):
            os.makedirs(work_dir)

    # only perform trial level splitting
    datasets = ['sst2', 'dbpedia5', 'agnews', 'imdb', 'pubmed']
    acq_fns = ['dal']#['random', 'cal', 'margin', 'dal', 'real']
    reps = ['wordvecs', 'MPNet', 'USE']
    num_processes, num_trials = 4, 3
    seed_size, batch_size, num_al_iters = 500, 500, 9
    num_unlabeled, num_test = 20000, 5000

    supported_transforms = {'wordvecs': ('wordvecs', SpacyWordVecs, None),
                            'MPNet': ('MPNet', 'all-mpnet-base-v2', None),
                            'USE': ('USE', None, None),
                            'ngrams': ('ngrams', None, None)}
    transforms = [supported_transforms[ct] for ct in reps]

    preloaded_data_trf, preloaded_data = {}, {}
    logging.info(f"We'll preload all the transformed data first.")
    for idx, (ds_name, (trf_name, trf_class, trf_params)) in enumerate(itertools.product(datasets, transforms)):
        data_fname = f"{preprocessed_data_dir}/{ds_name}_{trf_name}.npz"
        if ds_name not in data_locs:
            data_locs[ds_name] = dict()
        data_locs[ds_name][trf_name] = data_fname

        if load_data and ds_name not in preloaded_data:
            X_train, y_train, X_test, y_test = du.load_data(ds_name, max_train_instances=num_unlabeled,
                                                            max_test_instances=num_test)
            preloaded_data[ds_name] = (X_train, y_train, X_test, y_test)
            data_locs[ds_name]['train_size'] = np.shape(X_train)[0]
            data_locs[ds_name]['test_size'] = np.shape(X_test)[0]
            logging.info(f"Preloaded data={ds_name}, shape train={np.shape(X_train)}, shape test={np.shape(X_test)}.")

        if load_data and (ds_name, trf_name) not in preloaded_data_trf:
            logging.info(f"Preloading data for {(ds_name, trf_name)} combination.")
            print(f"Preloading data for {(ds_name, trf_name)} combination.")
            X_train, y_train, X_test, y_test = preloaded_data[ds_name]
            if 'minilm' in trf_name.lower() or 'mpnet' in trf_name.lower():
                st_obj = SentTransformer(trf_class)
                X_train_trf = st_obj.fit_transform(X_train)
                X_test_trf = st_obj.fit_transform(X_test)
                del st_obj
            elif 'use' == trf_name.lower():
                use_obj = USE()
                X_train_trf = use_obj.fit_transform(X_train)
                X_test_trf = use_obj.fit_transform(X_test)
                del use_obj
            elif 'ngrams' == trf_name.lower():
                cvec = CountVectorizer(ngram_range=(1, 3))
                X_train_trf = cvec.fit_transform(X_train)
                X_test_trf = cvec.transform(X_test)
            elif 'wordvecs' == trf_name.lower():
                sp = SpacyWordVecs()
                X_train_trf = sp.fit_transform(X_train)
                X_test_trf = sp.fit_transform(X_test)
                del sp
            else:
                logging.error(f"Cannot understand transform {trf_name}, aborting!")
                return
            preloaded_data_trf[(ds_name, trf_name)] = (X_train_trf, y_train, X_test_trf, y_test)
            np.savez(data_fname, X_train_trf=X_train_trf, y_train=y_train, X_test_trf=X_test_trf, y_test=y_test)
            logging.info(f"Preloaded data for {(ds_name, trf_name)} combination, shape train={np.shape(X_train_trf)}, "
                  f"shape test={np.shape(X_test_trf)}")

    for idx, (ds, acq_fn, rep) in enumerate(itertools.product(datasets, acq_fns, reps)):
        config = dict()
        config['num_trials'] = num_trials
        config['seed_size'], config['batch_size'], config['num_al_iters'] = seed_size, batch_size, num_al_iters
        config['num_unlabeled'], config['num_test'] = num_unlabeled, num_test
        config['acq_fns'] = [acq_fn]
        config['datasets'] = [ds]
        config['transforms'] = [rep]
        config['data'] = data_locs
        config['logfile'] = f"{logfile_dir}/log_{ds}_{acq_fn}_{rep}.txt"
        config_file_path = f"{op_dir}/{ds}_{acq_fn}_{rep}.json"

        with open(config_file_path, 'w') as fw:
            fw.write(json.dumps(config, indent=4))
        f_script.write(f"{python_path} cmd_runner.py {config_file_path}\n")
        logging.info(f"Created config  file={config_file_path}")
    f_script.close()

def load_dataset(npz_file, expected_train_size=None, expected_test_size=None):
    npzfile = np.load(npz_file)
    X_train_trf, y_train, X_test_trf, y_test = npzfile['X_train_trf'], npzfile['y_train'], \
                                               npzfile['X_test_trf'], npzfile['y_test']
    if expected_train_size:
        assert np.shape(X_train_trf)[0] == expected_train_size
    if expected_test_size:
        assert np.shape(X_test_trf)[0] == expected_test_size
    return X_train_trf, y_train, X_test_trf, y_test


def is_al_file_complete(fpath, last_idx):
    df = pd.read_csv(fpath)
    last_idx_in_file = int(list(df['iter_idx'])[-1])
    if last_idx_in_file < last_idx:
        return False
    elif last_idx_in_file == last_idx:
        return True
    else:
        print(f"Unknown situation, expected last idx={last_idx}, actual last idx={last_idx_in_file}")
        return None


def differ(script, op_dir, new_runner_script_path, safe_move_path_script, safe_move_dir, rm_script_path):
    """
    Look at the putput dir and script that was used to generate those outputs, and enumerate incomplete or
    not-yet-started runs.
    :param script:
    :param op_dir:
    :param new_script_path:
    :return:
    """
    clfs = ['LinearSVC', 'RF']
    df = pd.read_csv(script, sep=" ", header=None, names=['python', 'runner', 'config_json'])
    assert len(set(df['python']))==1, "There are different Python runtimes mentioned!"
    assert len(set(df['runner'])) == 1, "There are different runners mentioned!"

    if not os.path.exists(safe_move_dir) or not os.path.isdir(safe_move_dir):
        os.makedirs(safe_move_dir)
    else:
        print(f"Safe move dir: {safe_move_dir} already exists. We'll use that.")

    configs = list(df['config_json'])
    redo, rm_cmds, safe_move_cmds = [], [], []
    for _, r in df.iterrows():
        c = r['config_json']

        config_dict = None
        with open(c) as f:
            config_dict = json.loads(f.read())
        num_trials = int(config_dict["num_trials"])
        num_al_iters = int(config_dict["num_al_iters"])
        dataset, acq, transform = config_dict['datasets'][0], config_dict['acq_fns'][0], config_dict['transforms'][0]
        print(f"\n\n{(dataset, acq, transform)}")
        op_c_dirnames = [f"{op_dir}/{dataset}_{clf}_{transform}_{acq}_trial_{i}" for i, clf in itertools.product(range(num_trials), clfs)]
        print(op_c_dirnames)
        op_c_filenames = [f"{i}/al.csv" for i in op_c_dirnames]
        status_exist = [True if os.path.exists(i) and os.path.isfile(i) else False for i in op_c_filenames]
        if status_exist.count(True) < len(status_exist):
            if status_exist.count(True) == 0:
                redo.append((f"{' '.join([r['python'], r['runner'], r['config_json']])}", 'No files exist.'))
                print(f"No files exist.\nredo:{redo[-1][0]}")
            else:

                redo.append((f"{' '.join([r['python'], r['runner'], r['config_json']])}", "Some files dont exist."))
                print(f"Some files do not exist.\nredo:{redo[-1][0]}")
                # need to delete the ones that do exist
                temp_rm = [f"rm -rf {j}" for i, j in zip(status_exist, op_c_dirnames) if i is True]
                rm_cmds += temp_rm
                print(f"del: {temp_rm}")

        else:
            status_complete = [is_al_file_complete(i, num_al_iters-1) for i in op_c_filenames]
            if status_complete.count(True) < len(status_complete):

                redo.append((f"{' '.join([r['python'], r['runner'], r['config_json']])}", 'Incomplete files.'))
                print(f"All files exist but some were not run to completion.\nredo:{redo[-1][0]}")
                temp_rm = [f"rm -rf {j}" for i, j in zip(status_exist, op_c_dirnames) if i is True]
                rm_cmds += temp_rm
                print(f"del: {temp_rm}")
            else:
                temp_safe = [f"cp -r {i} {safe_move_dir}" for i in op_c_dirnames]
                safe_move_cmds += temp_safe
                print(f"Safe move: {temp_safe}")
    redo = np.array(redo)
    print(f"Total initial instructions: {len(df)}.")
    print(f"Count of redo: {len(redo)}.")
    newline = '\n'
    if len(redo) > 0:
        print(f"Some redo examples:{newline}{redo[:5, :]}")

    with open(new_runner_script_path, 'w') as fw:
        fw.write(newline.join(redo[:, 0]))

    with open(safe_move_path_script, 'w') as fw:
        fw.write(newline.join(safe_move_cmds))

    with open(rm_script_path, 'w') as fw:
        fw.write(newline.join(rm_cmds))


def check_runs(op_dirs, final_iter_idx, num_trials, copy_to=None):
    clf_names = ['LinearSVC', 'RF']
    acq_names =['dal', 'random', 'cal', 'real', 'margin']
    datasets = ['agnews', 'dbpedia5', 'sst2', 'pubmed', 'imdb']
    reps = ['wordvecs', 'MPNet', 'USE']

    expected_runs = []
    for dataset, clf, rep, acq, trial_num in itertools.product(datasets, clf_names, reps, acq_names, range(num_trials)):
        exp_dir_name = f"{dataset}_{clf}_{rep}_{acq}_trial_{trial_num}"
        exp_fname = os.sep.join((exp_dir_name, 'al.csv'))
        expected_runs.append(exp_fname)

    completed_runs = dict()
    for op_dir in op_dirs:
        print(f"Currently inspecting {op_dir}.")
        for fname in glob.iglob(f"{op_dir}/**/al.csv", recursive=True):
            path_comps = fname.split(os.sep)
            parent_dir = path_comps[-2]
            temp = parent_dir.split("_")
            # dataset, clf, rep, acq, trial_num = temp[0], temp[1], temp[2], temp[3], int(temp[-1])
            df = pd.read_csv(fname)
            last_iter_idx = int(list(df['iter_idx'])[-1])
            is_complete = False
            if last_iter_idx == final_iter_idx:
                is_complete = True
            # store the full path name for debugging
            # it is possible we might be double counting - in that case, favor the one with the completed status
            identifier_path = os.sep.join((parent_dir, 'al.csv'))
            if identifier_path not in completed_runs:
                completed_runs[identifier_path] = (is_complete, fname)
            else:
                if is_complete:
                    completed_runs[identifier_path] = (is_complete, fname)

    # compare
    counts = {'complete': 0, 'incomplete': 0}
    completes,  incompletes = [], []
    for r in expected_runs:
        is_complete, fname = completed_runs.get(r, (False, None))
        if is_complete:
            counts['complete'] += 1
            print(f"[Complete] Expected={r}, actual={fname}")
            completes.append((r, fname))
        else:
            counts['incomplete'] += 1
            print(f"[Incomplete] Expected={r}, actual={fname}")
            incompletes.append((r, fname))

    print(f"Total expected: {len(expected_runs)}")
    print(counts)

    print("\n".join([f"{i}, {j}" for i, j in incompletes]))
    if copy_to is not None:
        print(f"Will copy the completed files into {copy_to}.")
        if not os.path.exists(copy_to) or not os.path.isdir(copy_to):
            os.makedirs(copy_to)
            print(f"Created directory {copy_to}.")
        for _, fname in completes:
            p = Path(fname)
            src_dir = str(p.parent)
            last_dir = p.parts[-2]
            dest = f"{copy_to}{os.sep}{last_dir}"
            print(f"Copying {src_dir} to {dest}.")
            shutil.copytree(src_dir, dest)

if __name__ == "__main__":
    # setup_logger()
    # run_expts()
    # process_expt_results('CAL, k=3', [r'scratch/cal/expt_details.csv',
    #                                   r'scratch/cal_2/expt_details.csv',
    #                                   r'scratch/cal_1/expt_details.csv'])
    # USE
    # du.show_sample_data()
    # sp = SpacyWordVecs()
    # print(np.shape(sp.fit_transform(['hi i am abhishek', 'whats a good book on economics?'])))
    generate_config_json(op_dir=r'scratch/configs', preprocessed_data_dir=r'scratch/preprocessed_dont_delete',
                         script_dir=r'./', load_data=False, logfile_dir=r'scratch/logfiles',
                         python_path='/media/aghose/DATA/anaconda39/bin/python')
    # run_expt_nomp(r'scratch/configs/27_dbpedia_dal_USE.json')
    # load_dataset(r'scratch/preprocessed_data/agnews_MPNet.npz', 1000, 5000)
    # differ(r'./runner.sh', r'scratch/partial/seed_200_batch_size_200_iters_24',
    #        new_runner_script_path=r'./new_runner.sh',
    #        safe_move_path_script=r'./safe_move.sh',
    #        safe_move_dir=r'scratch/safe',
    #        rm_script_path=r'./rm_script.sh')
    # check_runs([r'/media/aghose/DATA/sources/active_learning_baselines_with_data/active_learning_baselines/'
    #             r'scratch/seed_200_batch_size_200_iters_24',
    #             r'/media/aghose/DATA/sources/active_learning_baselines_with_data/linode_active_learning/'
    #             r'scratch/safe_v1',
    #             r'/media/aghose/DATA/sources/active_learning_baselines_with_data/linode_active_learning/'
    #             r'scratch/safe',
    #             r'/media/aghose/DATA/sources/active_learning_baselines_with_data/linode_active_learning/'
    #             r'scratch/seed_200_batch_size_200_iters_24'
    #             ], final_iter_idx=23, num_trials=3,
    #            copy_to=r'/media/aghose/DATA/sources/active_learning_baselines_with_data/active_learning_baselines/'
    #             r'scratch/final_seed_200_batch_size_200_iters_24')
    # check_runs([r'/media/aghose/DATA/sources/active_learning_baselines_with_data/active_learning_baselines/'
    #             r'scratch/final_seed_200_batch_size_200_iters_24'], final_iter_idx=23, num_trials=3, copy_to=None)