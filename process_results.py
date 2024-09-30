import itertools
import json
import re, sys, os, pandas as pd, glob, numpy as np, math
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from itertools import product
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import results_utils

TEMP_WORKING_DIR = r'scratch/temp'  # these wont be checked in
RESULTS_DIR = r'results' # these will be checked in

# make sure the latest results are in this directory - the expanded non-BERT zips have a different name, change them
# manually to match these :-/
AL_RESULTS_DIR = defaultdict(dict)
AL_RESULTS_DIR['BERT'][200] = r'scratch/current_results/final_BERT_seed_200_batch_size_200_iters_24'
AL_RESULTS_DIR['BERT'][500] = r'scratch/current_results/final_BERT_seed_500_batch_size_500_iters_9'
AL_RESULTS_DIR['non_BERT'][200] = r'scratch/current_results/final_non_BERT_seed_200_batch_size_200_iters_24'
AL_RESULTS_DIR['non_BERT'][500] = r'scratch/current_results/final_non_BERT_seed_500_batch_size_500_iters_24'

# map the function names that show up in the results files to short readable names
acq_fn_name_map = {'CAL.acquire': 'cal',
                   'DAL.acquire': 'dal',
                   'REAL.acquire': 'real',
                   'acq_margin': 'margin',
                   'acq_random': 'random'}

clf_name_map = {'LinearSVC': 'LinSVC',
                'RF': 'RF',
                'BERT': 'RoBERTa'}

rep_name_map = {'MPNet': 'MP',
                'USE': 'USE',
                'wordvecs': 'WV',
                'BERT': 'RoBERTa'}

# preferred display ordering
pipeline_order = [f"{'-'.join(i)}" for i in list(itertools.product(['LinSVC','RF'], ['WV', 'USE',  'MP'])) +
                  [('RoBERTa',)]]

eff_size_map = {200: 500, 400: 500, 600: 500, 800: 1000, 1000: 1000, 1200: 1000, 1400: 1500, 1600: 1500,
                   1800: 2000, 2000: 2000, 2200: 2000, 2400: 2500, 2600: 2500, 2800: 3000, 3000: 3000, 3200: 3000,
                   3400: 3500, 3600: 3500, 3800: 4000, 4000: 4000, 4200: 4000, 4400: 4500, 4600: 4500,
                   4800: 5000, 5000: 5000}
eff_size_map.update({i:i for i in np.arange(500, 5001, 500)})

def dirname_parse(pathname, name_type='BERT'):
    """
    parse dir names and al file to find dataset and AL setting.
    Experiments by Abhishek (non_BERT) and Emma (BERT) follow different conventions.
    :param pathname: str, path all the way to al.csv
    :param name_type: 'BERT' or 'non_BERT'
    :return:
    """
    path = Path(pathname)
    gparent, parent = path.parts[-3], path.parts[-2]
    parsed = {'dataset': None, 'batch_size': None, 'seed_size': None, 'acq_name': None, 'clf': None, 'rep': None,
              'trial_idx': None}

    # extract seed and batch size from the actual file
    with open(pathname) as f:
        lines = f.readlines()
        # convert to float first since some ints might get logged as float and direct int
        # casting would fail
        seed_size = int(float(lines[1].split(",")[1]))
        batch_size = int(float(lines[2].split(",")[1])) - seed_size

    if name_type == 'BERT':
        parsed['clf'] = 'BERT'
        parsed['rep'] = 'BERT'
        toks_parent,toks_gparent = parent.split("_"), gparent.split("_")
        parsed['dataset'], parsed['acq_name'] = toks_gparent[0], toks_gparent[-1]
        batch_size_from_name = int(toks_gparent[1][2:])
        seed_size_from_name =batch_size_from_name
        assert seed_size_from_name == seed_size, f"seed size from name={seed_size_from_name} " \
                                                 f"and file={seed_size} don't match!"
        assert batch_size_from_name == batch_size, f"batch size from name={batch_size_from_name} " \
                                                 f"and file={batch_size} don't match!"
        parsed['seed_size'] = seed_size
        parsed['batch_size'] = batch_size
        parsed['trial_idx'] = int(toks_parent[-1])

    elif name_type == 'non_BERT':
        toks = parent.split("_")
        parsed['dataset'], parsed['clf'], parsed['rep'], parsed['acq_name'], parsed['trial_idx'] = toks[0:4] + \
                                                                                              [int(toks[-1])]
        parsed['seed_size'] = seed_size
        parsed['batch_size'] = batch_size
    else:
        print(f"Can't recognize name type ={name_type}, aborting!")
    return parsed


def collate_data(BERT_dirname, non_BERT_dirname, op_dir=None, name_suffix=None):
    df = pd.DataFrame()
    for name_type, dirname in (('BERT', BERT_dirname), ('non_BERT', non_BERT_dirname)):
        for t in glob.glob(f"{dirname}/**/al.csv", recursive=True):
            print(f"\n{t}")
            metadata = dirname_parse(t, name_type=name_type)
            print(f"Extracted metadata: {metadata}")
            # parent_dir_name = t.split(os.sep)[-2]
            # dataset, clf, transform, acq, *trial_str = parent_dir_name.split("_")
            # trial_idx = int(trial_str[-1])
            # print(dataset, clf, transform, acq, trial_idx)
            df_al = pd.read_csv(t)

            # create a temp df with meta data that needs to be associated with the AL results
            # metadata = {'dataset': dataset, 'clf': clf, 'transform': transform, 'trial_idx': trial_idx}
            df_metadata = pd.concat([pd.DataFrame([metadata] * len(df_al))], axis=0, ignore_index=True)

            # horizontally concat them
            df_al_meta = pd.concat([df_metadata, df_al], axis=1)

            # this goes into the overall result df
            df = pd.concat([df, df_al_meta], ignore_index=True)

    # there're two acq. names, one comes from the file name, and is was in the file
    assert list(df['acq_name']) == list(map(lambda x: acq_fn_name_map[x], df['acq'])), "Acquisition functions derive " \
                                                                                       "from file name and in file don't match!"
    df.drop(columns=['acq'], inplace=True)
    df = df.astype({"iter_idx": float, 'train_size': float})
    df = df.astype({"iter_idx": int, 'train_size': int})

    # some checks
    if len(set(df['batch_size'])) > 1 or len(set(df['seed_size'])) > 1:
        print(f"We might be looking a two different settings wrt batch/seed size! Aborting!")
        return

    # aggregate over trial_idx
    aggr_df = df.groupby(by=['dataset', 'clf', 'rep', 'acq_name', 'iter_idx', 'train_size'], as_index=False).agg(
                                                avg_score=pd.NamedAgg(column='score', aggfunc='mean'),
                                                stdev=pd.NamedAgg(column='score', aggfunc='std'),
                                            )
    if op_dir:
        if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
            os.makedirs(op_dir)
            print(f"Created new dir for output: {op_dir}")
        df_file, aggr_df_file = f"{op_dir}/all_data{'_'+name_suffix if name_suffix else ''}.csv", \
                                f"{op_dir}/aggr_data{'_'+name_suffix if name_suffix else ''}.csv"
        print(f"Writing to files: {(df_file, aggr_df_file)}.")
        df.to_csv(df_file, index=False)
        aggr_df.to_csv(aggr_df_file, index=False)
    return df, aggr_df

def factor_plot(result_df, aggr_df):
    datasets, acq_fns, clfs, transforms = set(aggr_df['dataset']), set(aggr_df['acq']), set(aggr_df['clf']), \
                                          set(aggr_df['transform'])
    for ds, clf, transform in product(datasets, clfs, transforms):
        print(f"Processing: {(ds, clf, transform)}")
        # temp_df = aggr_df[aggr_df['dataset']==ds & aggr_df['clf']==clf]
        temp_df = result_df.query(f'(dataset=="{ds}") and (clf=="{clf}") and (transform=="{transform}")')

        if len(temp_df) == 0:
            continue
        print(len(temp_df))
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_ylim(0.6, 1)
        ax.set_xlabel('# training instances')
        ax.set_ylabel('F1-macro on test')
        ax.set_title(f"dataset={ds}, {clf}$ \\rightarrow ${transform}")
        sns.lineplot(temp_df, x='train_size', y='score', hue='acq', ax=ax)
        plt.show()

def dal_times(fname):
    df = pd.read_csv(fname)
    times = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f') for i in df['ts']]
    train_sizes = df['train_size'].to_numpy()
    X, y = [], []

    for i in range(len(df) - 1):
        ds_curr, ts_curr = train_sizes[i], times[i]
        ds_next, ts_next = train_sizes[i+1], times[i+1]
        dur = (ts_next-ts_curr).total_seconds()
        X.append(ds_next)
        y.append(dur)
    plt.plot(X, y, marker='o')
    plt.xlabel('train size')
    plt.ylabel('time in sec')
    plt.title(f"(DAL AL iters) Total time={sum(y)/3600.:.2f} hr, #iters={len(df)-1}.")
    plt.show()


def avg_friedman(grp_df):
    df_piv = pd.pivot_table(grp_df, values='avg_score', index=['train_size'], columns='acq_name', aggfunc='mean')
    res = friedmanchisquare(*np.array(df_piv).T)
    return pd.Series({'value': res.pvalue})


def std_auc(grp_df):
    """
    Auc so far.
    :param grp_df:
    :return:
    """
    df_piv = pd.pivot_table(grp_df, values='avg_score', index=['train_size'], columns='acq_name', aggfunc='mean')
    x = df_piv.index.tolist()
    aucs = []
    for c in df_piv.columns:
        aucs.append(auc(x, df_piv[c]))
    auc_std = np.std(aucs)
    return pd.Series({'value': auc_std})


def std_val(grp_df):
    df_piv = pd.pivot_table(grp_df, values='avg_score', index=['train_size'], columns='acq_name', aggfunc='mean')
    x = df_piv.index.tolist()
    vals = []
    for c in df_piv.columns:
        vals.append(df_piv[c].tolist()[-1])
    vals_std = np.std(vals)
    return pd.Series({'value': vals_std})


def cumulative_aggr_acq_loss(grp_df, name="AUC std"):
    if name == "Friedman":
        return avg_friedman(grp_df)
    elif name == "AUC std":
        return std_auc(grp_df)
    elif name == "val std":
        return std_val(grp_df)
    else:
        print(f"Can't understand stat type={name}, aborting!")


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a-b)))


def diff(new_arr, ref_arr):
    """
    For cases where the sign matters. Here it is expected that new_arr > ref_arr, so a high +ve result is good,
    and a high -ve positive result is bad.
    :param new_arr:
    :param ref_arr:
    :return:
    """
    return np.sum(new_arr - ref_arr)


def combinatorial_error(grouped_rmse, ref_rmse, error_fn=rmse):
    """

    :param grouped_rmse: should be dict with key as an acq name, and value numpy array containing accuracies, e.g.,
        {'cal': [[0.2], [0.3], [0.22], [0.31], [0.12], [0.23]], 'dal':.....}.
        The multiple rows in the array represent different trials.
    :param ref_rmse: just one numpy array against which the rmses will be computed.
    :return:
    """
    df = pd.DataFrame()
    for k, v in grouped_rmse.items():
        for trial_v in v:
            temp = error_fn(trial_v, ref_rmse)
            df = pd.concat([df, pd.DataFrame([{'name': k, 'error': temp}])], ignore_index=True)
    avg_rmse = np.mean(df['error'])
    df_avg = df.groupby(by='name').agg('mean')
    return avg_rmse, df_avg


def multipop_tests(df_all, df_aggr, op_dir, remove_seed_step=True, stat_name='AUC std', suffix=None):
    """
    Multi-population tests, i.e., which analyze the behavior of all acquisition functions together to look for
    overall patterns. Friedman test p-values are an option, and so is looking at the standard deviation of AUC values.
    The Friedman test can be problematic here since the blocks might not be considered independent - the score at
    200 training size correlates with that at 400 training size.

    :param df_all:
    :param df_aggr:
    :param remove_seed_step:
    :return:
    """
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    print(f"Total rows in all: {len(df_all)}")
    print(f"Total rows in aggr: {len(df_aggr)}")
    # this iter is from the seed set, not AL
    if remove_seed_step:
        df_aggr = df_aggr.query('iter_idx != -1')
        df_all = df_all.query('iter_idx != -1')

    train_sizes = sorted(list(set(df_aggr['train_size'])))
    print(f"{len(train_sizes)} train sizes [{train_sizes[0]}-{train_sizes[-1]}] found.")

    # iterate based on train_size, at each point consider all sizes up to the current one
    all_stat = []
    for i in range(len(train_sizes)):
        cumulative_sizes = train_sizes[0:i+1]
        # if len(cumulative_sizes) < 2:
        #     continue
        print(f"\nProcessing size {train_sizes[i]}, cumulative: [{cumulative_sizes[0]} - {cumulative_sizes[-1]}].")

        # subset the dataframes
        curr_df_aggr = df_aggr[df_aggr['train_size'].isin(cumulative_sizes)]
        curr_df_all = df_all[df_all['train_size'].isin(cumulative_sizes)]

        # get the friedman scores, use the aggr frame, group by prediction pipeline
        df_stat = curr_df_aggr.groupby(by=['clf', 'rep', 'dataset'],
                                       as_index=False).apply(cumulative_aggr_acq_loss, name=stat_name)
        df_stat['train_size'] = cumulative_sizes[-1]
        print(df_stat)
        all_stat.append(df_stat)
        pass
    df_all_res = pd.concat(all_stat)
    df_piv = pd.pivot_table(df_all_res, values='value', index=['clf', 'rep', 'dataset'], columns='train_size',
                            aggfunc='mean')
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(211)
    df_plot = df_piv.transpose()
    df_plot['train size'] = df_plot.index
    df_plot = df_plot.melt(id_vars='train size')
    if stat_name == 'Friedman':
        new_col_name = 'p value (avg. over datasets), Friedman test'
        title = 'p value, Friedman test (avg. over datasets)'
    elif stat_name == 'AUC std':
        new_col_name = 'AUC std'
        title = "std. dev. of AUCs for different QS (avg. over datasets)"
    elif stat_name == 'val std':
        new_col_name = 'val std'
        title = 'std. dev. of F1 macro values (avg. over datasets)'
    else:
        print(f"Can't understand name type={stat_name}, aborting!")
        return

    df_plot.rename(columns={'value': new_col_name}, inplace=True)
    df_plot['predictor'] = ["RoBERTa" if i =='BERT' else f"{i}_{j}" for i, j in zip(df_plot['clf'], df_plot['rep'])]
    sns.lineplot(df_plot, x='train size', y=new_col_name, hue='predictor',
                 marker='o', err_style='band', ax=ax)
    ax.set_title(title)
    fname = f"{stat_name}_{suffix}" if suffix else f"{stat_name}"
    plt.savefig(f"{op_dir}/{fname}.png", bbox_inches='tight')
    plt.savefig(f"{op_dir}/{fname}.pdf", bbox_inches='tight')


def datawise_plots(df, op_dir, suffix=None):
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    df.rename(columns={'score': 'F1 macro', 'acq_name': 'QS'}, inplace=True)
    datasets = set(df['dataset'])
    unique_pipelines = set([tuple(i) for i in np.array(df[['clf', 'rep']])])
    for ds in datasets:
        print(f"Processing for dataset={ds}.")
        fig, axes, name_posn_map = results_utils.get_plot_template(title=f"dataset: {ds}, batch size={suffix}",
                                                                   return_name_position_map=True)

        for p in unique_pipelines:
            fig_posn = name_posn_map[p]
            temp_df = df.query(f'dataset=="{ds}" and clf=="{p[0]}" and rep=="{p[1]}"')
            sns.lineplot(temp_df, x='train_size', y='F1 macro', hue='QS', ax=axes[fig_posn])

            ax = axes[fig_posn]
            ax.set_xlabel(ax.get_xlabel(), fontsize=22)
            ax.set_ylabel(ax.get_ylabel(), fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=22)
            ax.legend(fontsize=22)

        fname = f"{ds}_{suffix}" if suffix else f"{ds}"
        for extn in ['png', 'pdf', 'svg']:
            plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
        plt.clf()


def generate_all():
    collate_data(BERT_dirname=r'scratch/current_results/final_BERT_seed_200_batch_size_200_iters_24',
                 non_BERT_dirname=r'scratch/current_results/final_non_BERT_seed_200_batch_size_200_iters_24',
                 op_dir=f"{RESULTS_DIR}/collated", name_suffix='200')
    collate_data(BERT_dirname=r'scratch/current_results/final_BERT_seed_500_batch_size_500_iters_9',
                 non_BERT_dirname=r'scratch/current_results/final_non_BERT_seed_500_batch_size_500_iters_9',
                 op_dir=f"{RESULTS_DIR}/collated", name_suffix='500')

    for b in [200, 500]:
        df_all_results = pd.read_csv(f"{RESULTS_DIR}/collated/all_data_{b}.csv")
        df_aggr_results = pd.read_csv(f"{RESULTS_DIR}/collated/aggr_data_{b}.csv")
        datawise_plots(df_all_results, op_dir=r'results/datawise_plots', suffix=f"{b}")
        multipop_tests(df_all_results, df_aggr_results, remove_seed_step=True, op_dir=r'results/stat_tests',
                       stat_name='val std', suffix=f"{b}")
        multipop_tests(df_all_results, df_aggr_results, remove_seed_step=True, op_dir=r'results/stat_tests',
                       stat_name='AUC std', suffix=f"{b}")
        acq_wilcoxon_and_rmse(df_all=df_all_results, df_aggr=df_aggr_results, batch_size=b,
                              op_dir=r'results/stat_tests',
                              min_sample_for_stat=3)


def problem_with_wilcoxon(op_dir):
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    X = np.linspace(500, 5000, 10)
    y = np.array([0.5, 0.6, 0.7, 0.77, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86 ])
    y1, y2 = y + 0.03 + np.random.normal(0, 0.001, len(y)), y + 0.1 + np.random.normal(0, 0.001, len(y))
    d1, d2 = y1 - y, y2 - y

    fig = plt.figure(figsize = (6, 4))

    ax = fig.add_subplot(111)
    ax.plot(X, y, marker='o', label='random')

    _, p = wilcoxon(d1, alternative='greater')
    ax.plot(X, y1, marker='o', label=f'QS 1, p={p:.2E}')
    _, p = wilcoxon(d2, alternative='greater')
    ax.plot(X, y2, marker='o', label=f'QS 2, p={p:.2E}')
    ax.set_xlabel('train size', fontsize=16)
    ax.set_ylabel('F1 macro', fontsize=16)
    ax.set_title('p-value, Wilcoxon test, for a QS being better than random',fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=16)
    # ax.legend()
    # ax.set_ylim(0, 1)
    fname = f"wilcoxon_limitation"
    for extn in ['png', 'pdf']:
        plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')


def get_acq_plot_template(title_1=None, title_2=None):
    name_position_map = {}
    label_fontsize = 16
    title_fontsize = 20
    if title_1 is None:
        title_1 = 'p-val (Wilcoxon)'
    if title_2 is None:
        title_2 = 'rmse'

    fig = plt.figure(figsize=(32, 22), constrained_layout=True)

    widths = [1, 1]
    heights = [1, 1, 0.2, 1, 1]
    spec = fig.add_gridspec(ncols=2, nrows=5, width_ratios=widths,
                            height_ratios=heights)
    ax = fig.add_subplot(spec[2, :])
    ax.axis("off")

    for qs, coord_1, coord_2 in [('cal', (0, 0), (1, 0)),
                                 ('dal', (0, 1), (1, 1)),
                                 ('real', (3, 0), (4, 0)),
                                 ('margin', (3, 1), (4, 1))]:
        ax = fig.add_subplot(spec[coord_1[0], coord_1[1]])
        ax.set_title(f'{qs}: {title_1}', fontsize=title_fontsize)
        ax.set_xlabel('train size', fontsize=label_fontsize)
        ax.set_ylabel('p-val', fontsize=label_fontsize)
        name_position_map[(qs, title_1)] = ax

        ax = fig.add_subplot(spec[coord_2[0], coord_2[1]])
        ax.set_title(f'{qs} - {title_2}', fontsize=title_fontsize)
        ax.set_xlabel('train size', fontsize=label_fontsize)
        ax.set_ylabel('rmse', fontsize=label_fontsize)
        name_position_map[(qs, title_2)] = ax

    return fig, name_position_map


def acq_wilcoxon_and_rmse(df_all, op_dir, remove_seed_step=True, min_sample_for_stat=2):
    """
    Currently only the common plot (towards the end of this function, is used in the paper)
    :param df_all:
    :param op_dir:
    :param remove_seed_step:
    :param min_sample_for_stat:
    :return:
    """
    title_1 = 'p-val (Wilcoxon)'
    title_2 = 'rmse'
    batch_sizes = sorted(set(df_all['batch_size']))
    fig, name_position_map = get_acq_plot_template(title_1, title_2)

    # calculate aggr. properties of various acq. fns., compared to random, plotted against training size -
    # rolling up both pipelines and dataset
    unique_acq = sorted(set(df_all['acq_name']) - {'random'})
    if remove_seed_step:
        df_all = df_all.query('iter_idx != -1')
    overall_df = pd.DataFrame()
    # fname = f"acq_wilcoxon_rmse_common"
    for batch_size in batch_sizes:
        df_all_b = df_all.query(f"batch_size=={batch_size}")
        for acq in unique_acq:
            print(f"Currently processing acq={acq}, batch size = {batch_size}.")
            temp_df = df_all_b.query(f"acq_name=='{acq}' or acq_name=='random'")
            df_piv = pd.pivot_table(temp_df, values='score', index=['train_size'], columns='acq_name', aggfunc='mean')
            df_piv['train size'] = df_piv.index
            X_plot = np.sort(np.array(df_piv['train size']))
            y_plot_stat, y_plot_err = [], []
            # for the y values, rem. we need to perform computation up till a point
            for i in range(len(X_plot)):
                valid_sizes = X_plot[:i+1]
                trunc_df = df_piv[df_piv['train size'].isin(valid_sizes)]
                diffs = trunc_df[acq] - trunc_df['random']
                if i+1 < min_sample_for_stat:
                    y_plot_stat.append(np.nan)
                else:
                    _, p = wilcoxon(diffs, alternative='greater', zero_method='zsplit')
                    y_plot_stat.append(p)
                err = rmse(trunc_df[acq], trunc_df['random'])
                y_plot_err.append(err)

            ax_stat = name_position_map[(acq, title_1)]
            ax_err = name_position_map[(acq, title_2)]
            ax_stat.plot(X_plot, y_plot_stat)
            ax_err.plot(X_plot, y_plot_err)
            # overall[acq] = {'stat': y_plot_stat, 'err': y_plot_err, 'X': X_plot}
            overall_df = pd.concat([overall_df, pd.DataFrame.from_records({'p-val, Wilcoxon': y_plot_stat,
                                                                           'rmse': y_plot_err, 'train size': X_plot,
                                                                           'QS': [acq] * len(X_plot),
                                                                           'batch_size': [batch_size] * len(X_plot)
                                                                           })], ignore_index=True)

        fname = f"acq_wilcoxon_rmse_{batch_size}"
        for extn in ['png', 'pdf']:
            fig.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')

    # make a common plot as well
    plt.clf()
    fig = plt.figure(figsize=(14, 5))

    ax_err = fig.add_subplot(122)
    ax_err.set_title(f"RMSE",fontsize='16')
    ax_stat = fig.add_subplot(121)
    ax_stat.set_title(f"p-val (greater, min. sample={min_sample_for_stat}), Wilcoxon signed-rank test",fontsize='16')
    linestyle = {200: "-", 500: "--"}
    for batch_size in batch_sizes:
        overall_df_b = overall_df.query(f"batch_size=={batch_size}")
        overall_df_b['QS'] = [f"{i}, {batch_size}" for i in overall_df_b['QS']]
        sns.lineplot(overall_df_b, x='train size', y='rmse', hue='QS',
                     linestyle=linestyle[batch_size],  ax=ax_err)
        # need to explicitly set the starting lim since the wilcoxon values are not computed for some initial samples
        # due to low sample size
        ax_stat.set_xlim(ax_err.get_xlim())
        sns.lineplot(overall_df_b, x='train size', y='p-val, Wilcoxon',
                     hue='QS', linestyle=linestyle[batch_size], ax=ax_stat)

    for ax in [ax_err, ax_stat]:
        handles, labels = ax.get_legend_handles_labels()
        for h in handles:
            batch_size = int(h._label.split(",")[1])
            h.set_linestyle(linestyle[batch_size])
        valid_lines = [ line for line in ax.lines if "_"!=line._label[0]]
        ax.legend(handles=[(line_hand, point_hand) for line_hand, point_hand in zip(valid_lines, handles)],
                  labels=labels, title=ax.legend_.get_title().get_text(), handlelength=3)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.legend(fontsize=16)
        # plt.setp(ax.get_legend().get_texts(), fontsize='16')  # for legend text
        # plt.setp(ax.get_legend().get_title(), fontsize='16')  # for legend title
    plt.setp(ax_err.get_legend().get_texts(), fontsize='14')  # for legend text
    plt.setp(ax_err.get_legend().get_title(), fontsize='14')  # for legend title
    plt.setp(ax_stat.get_legend().get_texts(), fontsize='14')  # for legend text
    plt.setp(ax_stat.get_legend().get_title(), fontsize='14')  # for legend title

    fig.suptitle(f"Comparison to random, avg. over datasets & pipelines",fontsize='16')
    fname = f"acq_wilcoxon_rmse_common"
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')


def pdp(df_results, op_dir):
    """
    A fairly basic version of partial dependence plots. PD to be shown wrt:
    (a) pipeline, i.e, clf+rep together and then clf and rep separately
    (b) acq
    (c) batch size
    ALWAYS avg over the dataset here. Don't need to average over trials, since the averaging during PD would take
    care of it.


    :param df_results: list with elements (batch_size, all_results, aggr_results)
    :param op_dir:
    :return:
    """
    figsize = (10, 6)
    num_train_bins = 4

    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    batch_size_col = np.concatenate([[t[0]] * len(t[1]) for t in df_results])
    df_all = pd.concat([t[1] for t in df_results])
    df_all['batch_size'] = batch_size_col
    df_all['pipeline'] = [f"{r['clf']}-{r['rep']}" if r['clf']!='BERT' else 'RoBERTa' for _, r in df_all.iterrows()]
    df_all.rename(columns={'score': 'F1 macro', 'acq_name': 'QS'}, inplace=True)

    # if we want to distinguish random and everything else in one bucket, we need this column
    df_all['QS (binary)'] = ['random' if r['QS'] == 'random' else 'non-random' for _, r in df_all.iterrows()]

    # if we want the displays to be split by batch sizes, we can set the num bins here
    # note that this includes the seed set, otherwise it becomes a bit complicated

    bin_edges = np.linspace(0, 5000, num_train_bins + 1)
    bin_dict = dict([(i+1, f"({bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}]") for i in range(num_train_bins)])
    print(bin_dict)
    bin_idxs = np.digitize(df_all['train_size'], bin_edges, right=True)
    bin_names = [bin_dict[i] for i in bin_idxs]
    df_all['train size bin'] = bin_names

    # now we're ready to create our PD plots as barplots
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal effect of QS (random vs non-random)")
    sns.barplot(data=df_all, x='train size bin', y='F1 macro', hue='QS (binary)', ax=axs)
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/acq_binary_train_size_bins.{extn}", bbox_inches='tight')

    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal effect of QS")
    sns.barplot(data=df_all, x='train size bin', y='F1 macro', hue='QS',
                hue_order=['cal', 'dal', 'real', 'margin', 'random'], ax=axs)
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/acq_train_size_bins.{extn}", bbox_inches='tight')

    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal effect of prediction pipeline")
    sns.barplot(data=df_all, x='train size bin', y='F1 macro', hue='pipeline',
                hue_order= ['LinearSVC-wordvecs', 'LinearSVC-USE', 'LinearSVC-MPNet', 'RF-wordvecs',
                            'RF-USE', 'RF-MPNet', 'RoBERTa'],
                ax=axs)
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/pipeline_train_size_bins.{extn}", bbox_inches='tight')

    # we'll try to plot incremental effects over random
    # this will need us calculate these inc. gains first
    piv_cols = ['dataset', 'batch_size', 'pipeline', 'trial_idx', 'train_size', 'train size bin', 'clf', 'rep']
    temp_df = pd.pivot_table(df_all, values='F1 macro', index=piv_cols, columns='QS', aggfunc='mean')
    # temp_df.rename(columns=dict([(i, f"{i}_orig") for i in ('cal', 'real', 'dal', 'margin')]), inplace=True)

    # here's how we calculate the relative gain, "new" means something non-random
    rel_inc = lambda old, new: 100. *(new -old)/old
    # compare to random
    for i in ('cal', 'real', 'dal', 'margin'):
        temp_df[(i, 'random')] = [rel_inc(old, new) for new, old in zip(temp_df[i], temp_df['random'])]

    # compare to margin
    for i in ('cal', 'real', 'dal', ):
        temp_df[(i, 'margin')] = [rel_inc(old, new) for new, old in zip(temp_df[i], temp_df['margin'])]

    temp_df.drop(columns=[f"{i}" for i in ('cal', 'real', 'dal', 'margin')] + ['random'], inplace=True)
    # this actually restores the orig. columns that got bundled in pivoting
    temp_df.reset_index(inplace=True)
    # we want the acq. fns. in a single col., so we can use the "hue" param
    temp_df = pd.melt(temp_df, id_vars=piv_cols)
    temp_df.rename(columns={'value': 'rel. improve. in F1 macro'}, inplace=True)

    # finally plot, phew!
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal rel. improve. in F1-macro, over random, for a prediction pipeline",fontsize=16)
    temp_df['mask'] = [1 if r['QS'][1]=='random' else 0 for _, r in temp_df.iterrows()]
    sns.barplot(data=temp_df.query('mask==1'), x='train size bin', y='rel. improve. in F1 macro', hue='pipeline',
                hue_order=['LinearSVC-wordvecs', 'LinearSVC-USE', 'LinearSVC-MPNet', 'RF-wordvecs',
                           'RF-USE', 'RF-MPNet', 'RoBERTa'],
                ax=axs)
    axs.set_xlabel(axs.get_xlabel(), fontsize=16)
    axs.set_ylabel(axs.get_ylabel(), fontsize=16)
    axs.tick_params(axis='both', which='major', labelsize=14)
    # axs.legend(fontsize=16)
    plt.setp(axs.get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(axs.get_legend().get_title(), fontsize='16')  # for legend title
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/pipeline_incremental_train_size_bins.{extn}", bbox_inches='tight')

    # also wrt just QS
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal rel. improve. in F1-macro, over random, for query strategies",fontsize=16)
    temp_df['mask'] = [1 if r['QS'][1] == 'random' else 0 for _, r in temp_df.iterrows()]
    qsb_df = temp_df.query('mask==1')
    qsb_df['QS, batch size'] = [f"{r['QS']}, {r['batch_size']}" for _, r in qsb_df.iterrows()]
    print(set(qsb_df['QS, batch size']))
    palette = {"('margin', 'random'), 200": '#6897bb',
                "('margin', 'random'), 500": '#bfd3e2',
                "('cal', 'random'), 200": '#f28500',
                "('cal', 'random'), 500": '#fad6a5',
                "('dal', 'random'), 200": '#088F8F',
                "('dal', 'random'), 500": '#AFE1AF',
                "('real', 'random'), 200": '#D70040',
                "('real', 'random'), 500": '#F88379' }
    sns.barplot(data=qsb_df, x='train size bin', y='rel. improve. in F1 macro', hue='QS, batch size', ax=axs,
                palette=palette)
    axs.set_xlabel(axs.get_xlabel(), fontsize=16)
    axs.set_ylabel(axs.get_ylabel(), fontsize=16)
    axs.tick_params(axis='both', which='major', labelsize=14)
    # axs.legend(fontsize=16)
    plt.setp(axs.get_legend().get_texts(), fontsize='14')  # for legend text
    plt.setp(axs.get_legend().get_title(), fontsize='14')  # for legend title
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/acq_incremental_train_size_bins.{extn}", bbox_inches='tight')

    # just for BERT but with all acq. against margin
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal rel. improve. in F1-macro, for RoBERT, against 'margin' query strategy",fontsize=16)
    temp_df['mask'] = [1 if r['QS'][1] == 'margin' and r['pipeline']=='RoBERTa' else 0 for _, r in temp_df.iterrows()]
    sns.barplot(data=temp_df.query('mask==1'), x='train size bin', y='rel. improve. in F1 macro', hue='QS', ax=axs)
    axs.set_xlabel(axs.get_xlabel(), fontsize=16)
    axs.set_ylabel(axs.get_ylabel(), fontsize=16)
    axs.tick_params(axis='both', which='major', labelsize=14)
    # axs.legend(fontsize=16)
    plt.setp(axs.get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(axs.get_legend().get_title(), fontsize='16')  # for legend title
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/BERT_acq_incremental_train_size_bins.{extn}", bbox_inches='tight')

    # we'll use the created df to understand the effect of clf and rep as well, but this will have to be for
    # non-BERT setups only, since otherwise we can't effectively do a cross product
    temp_df_nonBERT = temp_df.query('clf != "BERT"')
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal rel. improve. in F1-macro, over random, for representations (BERT excl.)")
    sns.barplot(data=temp_df_nonBERT, x='train size bin', y='rel. improve. in F1 macro', hue='rep',
                hue_order=['wordvecs', 'USE', 'MPNet'],
                ax=axs)
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/rep_nonBERT_incremental_train_size_bins.{extn}", bbox_inches='tight')

    fig, axs = plt.subplots(1, figsize=figsize)
    axs.set_title("Expected marginal rel. improve. in F1-macro, over random, for classifiers (BERT excl.)")
    sns.barplot(data=temp_df_nonBERT, x='train size bin', y='rel. improve. in F1 macro', hue='clf',
                hue_order=['LinearSVC', 'RF'],
                ax=axs)
    for extn in ['png', 'pdf']:
        fig.savefig(f"{op_dir}/clf_nonBERT_incremental_train_size_bins.{extn}", bbox_inches='tight')


def avg_acc(df_all, op_dir, remove_seed_step=True):
    """
    Redoing the variance plot - this is simpler to understand
    :param df_all:
    :param op_dir:
    :return:
    """
    if remove_seed_step:
        df_all = df_all.query('iter_idx != -1')

    figsize = (6, 4)
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    # common
    # df_all['pipeline'] = ["RoBERTa" if i == 'BERT' else f"{i}_{j}" for i, j in
    #                           zip(top_combos['clf'], top_combos['rep'])]
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111)
    temp_df = df_all.groupby(by=['train_size', 'batch_size', 'clf', 'rep', 'dataset'], as_index=False).agg(
        score_var=pd.NamedAgg(column='score', aggfunc=np.var))
    # sns.set_palette("PuBuGn_d")
    # sns.lineplot(data=temp_df, x='train_size', y='score_var', hue='batch_size', marker='o', ax=ax, palette="tab10")
    sns.lineplot(data=temp_df, x='train_size', y='score_var', hue='batch_size', marker='o', ax=ax, palette="tab10")
    ax.set_ylabel(f'Expected var. of F1 macro', fontsize=16)
    ax.set_title(f'Expected var. of F1 macro scores', fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(ax.get_legend, fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='16')  # for legend title
    fname = f"avg_score_common"
    for extn in ['png', 'pdf']:
        plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
    plt.clf()

    batch_sizes = sorted(set(df_all['batch_size']))
    for b in batch_sizes:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sns.lineplot(data=df_all.query(f"batch_size=={b}"), x='train_size', y='score', hue='acq_name', ax=ax, marker='o')
        ax.set_ylabel('F1 macro')
        fname = f"avg_score_{b}"
        for extn in ['png', 'pdf']:
            plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
        plt.clf()

def get_auc(grp_df):
    temp_df = grp_df.groupby(by=['train_size', 'clf', 'rep', 'acq_name'], as_index=False).\
        agg(avg_score=pd.NamedAgg(column='score', aggfunc='mean'))
    print('Get AUC, temp_df\n', temp_df)
    res = auc(temp_df['train_size'], temp_df['avg_score'])
    return pd.Series({'partial_auc': res})


def best_combination(df_all, num_train_bins, op_dir):
    """
    Create a latex table with train size bin wise ranking of pipeline+acq. fn.
    :param df_all:
    :param num_train_bins:
    :param op_dir:
    :return:
    """

    top_k = 10
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    bin_edges = np.linspace(0, 5000, num_train_bins + 1)
    bin_dict = dict([(i + 1, f"({bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}]") for i in range(num_train_bins)])
    print(bin_dict)
    bin_idxs = np.digitize(df_all['train_size'], bin_edges, right=True)
    bin_names = [bin_dict[i] for i in bin_idxs]
    df_all['train size bin'] = bin_names
    best_df = df_all.groupby(by=['train size bin', 'clf', 'rep', 'acq_name'], as_index=False).apply(get_auc)
    # best_df.groupby(by=['train size bin'], as_index=False).agg(rank=pd.NamedAgg(column='partial_auc', aggfunc='rank'))
    best_df['rank'] =  best_df.groupby(by=['train size bin'], as_index=False)['partial_auc'].rank(ascending=False, method='dense')
    top_combos = best_df[best_df['rank'].isin(list(range(1, top_k+1)))].sort_values(['train size bin', 'rank'])
    top_combos['pipeline'] = ["RoBERTa" if i =='BERT' else f"{i}_{j}" for i, j in zip(top_combos['clf'], top_combos['rep'])]
    top_combos.rename(columns={'acq_name': 'QS'}, inplace=True)

    # clean up some formatting
    top_combos['train size bin'] = [re.sub('\.00', '', i) for i in top_combos['train size bin']]
    top_combos['partial_auc'] = [f"{i:.2f}" for i in top_combos['partial_auc']]
    top_combos = top_combos[['train size bin', 'pipeline', 'QS', 'partial_auc']]
    top_combos.to_latex(f"{op_dir}/top_combos.txt", index=False)

    # lets do a side-by-side plot too, for compactness - right now it can only handle even values for top_k!
    a = int(math.ceil(top_k/2.))
    b = top_k - a
    all = []
    for tr in sorted(set(top_combos['train size bin'])):
        print(tr)
        temp_df = top_combos[top_combos['train size bin']==tr]
        part_a_df = temp_df.head(a)
        part_a_df['pipeline'] = [f"{i}. {j}" for i, j in  zip(range(1, a+1), part_a_df['pipeline'])]
        part_b_df = temp_df.tail(b)
        part_b_df['pipeline'] = [f"{i}. {j}" for i, j in zip(range(a+1, top_k + 1), part_b_df['pipeline'])]
        side_by_side_df = pd.concat([part_a_df.reset_index(), part_b_df.reset_index()], axis=1, ignore_index=True)
        all.append(side_by_side_df)
    overall_side_df = pd.concat(all)
    overall_side_df = overall_side_df[[1, 2, 3, 4, 7, 8, 9]]
    overall_side_df.columns = ['train size', 'pipeline', 'QS', 'AUC', 'pipeline', 'QS', 'AUC']
    overall_side_df.to_latex(f"{op_dir}/top_combos_side_by_side.txt", index=False)


def auc_heatmap_non_random_vs_random(df_all, num_train_bins, op_dir, diff_type='relative'):
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    # print(df_all.head())
    # print(df_all.tail())
    bin_edges = np.linspace(0, 5000, num_train_bins + 1)
    bin_dict = dict([(i + 1, f"({bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}]") for i in range(num_train_bins)])
    print(bin_dict)
    bin_idxs = np.digitize(df_all['train_size'], bin_edges, right=True)
    bin_names = [bin_dict[i] for i in bin_idxs]
    df_all['train_size_bin'] = bin_names
    print(df_all.head())
    print(df_all.tail())
    bin_list = []
    # subplot_list = [211, 212, 221, 222]
    # fig = plt.figure(figsize=(15, 10))
    fig, axn = plt.subplots(1, 4, sharex=True, sharey=True,figsize=(20, 6))
    fig.suptitle(f"Relative improvement in AUC over random", fontsize=18)
    cbar_ax = fig.add_axes([.91, .3, .015, .4])

    for idx, ax in enumerate(axn.flat): #bin_idx, bin_name in bin_dict.items():
        bin_name = bin_dict[idx+1]
        bin_list.append(bin_name)
        print('=====', bin_list)
        df_bin = df_all[df_all['train_size_bin'].isin(bin_list)].copy()
        print(df_bin.shape)
        #TODO: Need to fix the auc calculation: CHANGE from average F1 -> compute AUC -> compute relative improv. AUC
        # TO compute AUC per trial, batch size, dataset, pipeline-> get relative improv AUC -> average
        df_bin = df_bin.groupby(by=['clf', 'rep', 'acq_name'], as_index=False).apply(get_auc)
        df_bin['pipeline'] = [f"{r['clf']}-{r['rep']}" if r['clf'] != 'BERT' else 'RoBERTa' for _, r in
                              df_bin.iterrows()]
        df_bin.rename(columns={'partial_auc': 'AUC', 'acq_name': 'QS'}, inplace=True)
        # print(df_bin)
        df_bin['AUC_random'] = [
            df_bin[(df_bin['pipeline'] == r['pipeline']) & (df_bin['QS'] == 'random')]['AUC'].values[0]
            for _, r in df_bin.iterrows()]
        if diff_type!='relative':
            df_bin['nr_r_diff'] = [r['AUC'] - r['AUC_random'] for _, r in df_bin.iterrows()]
            # print(df_bin['nr_r_diff'])
        else:
            df_bin['nr_r_diff'] = [(r['AUC'] - r['AUC_random'])/r['AUC_random'] for _, r in df_bin.iterrows()]
        df_bin = df_bin[df_bin['QS']!='random']
        print(df_bin)
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(subplot_list[idx])
        df_bin = df_bin.pivot(index='pipeline', columns='QS', values='nr_r_diff')
        sns.heatmap(data=df_bin, cmap="PiYG", vmax=0.05, vmin=-0.05, annot=True,
                    ax=ax, cbar= idx == 0, cbar_ax=None if idx else cbar_ax)  # cmap="crest"
        # ax.set_ylabel(f'Expected var. of F1 macro', fontsize=16)
        bin_max = int(bin_name.split('-')[1].split('.')[0])
        ax.set_title(f'Train size: {bin_max}', fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        if idx in [0]:
            ax.set_ylabel('Prediction Pipeline', fontsize=16)
        else:
            ax.set_ylabel('')
        # if idx in [2, 3]:
        #     ax.set_xlabel('QS', fontsize=16)
        # else:
        #     ax.set_xlabel('')

        ax.tick_params(axis='both', which='major', labelsize=14)
        # # ax.legend(ax.get_legend, fontsize=16)
        # plt.setp(ax.get_legend().get_texts(), fontsize='16')  # for legend text
        # plt.setp(ax.get_legend().get_title(), fontsize='16')  # for legend title
        # fname = f"auc_{bin_name}"
        # for extn in ['png']:#, 'pdf']:
        #     plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
        # plt.clf()
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fname = f"auc_heatmap"
    for extn in ['png' , 'pdf','svg']:
        plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
    plt.clf()

    # best_df = df_all.groupby(by=['train size bin', 'clf', 'rep', 'acq_name'], as_index=False).apply(get_auc)
    # # best_df.groupby(by=['train size bin'], as_index=False).agg(rank=pd.NamedAgg(column='partial_auc', aggfunc='rank'))
    # best_df['rank'] =  best_df.groupby(by=['train size bin'], as_index=False)['partial_auc'].rank(ascending=False, method='dense')


def kendall_w(scores):
    """
    :param scores: rows correspond to category levels, e.g., for QS this would be unc. sampling, CAL etc, and the
        columns correspond to the scores.
    :return:
    """
    if scores.ndim!=2:
        raise 'scores  matrix must be 2-dimensional'
    m = scores.shape[0]  # raters/levels
    n = scores.shape[1]  # items rated
    score_ranks = np.argsort(scores, axis=1)
    denom = m**2*(n**3-n)
    rating_sums = np.sum(score_ranks, axis=0)
    S = n*np.var(rating_sums)
    return 12*S/denom


def friedman_test(df_all, op_dir):
    df_all['bs'] = [f"({r['batch_size']},{r['seed_size']})" for _, r in df_all.iterrows()]
    print('===== Friedman test')
    df_all_nr = df_all[df_all['QS'] !='random']
    df_all_nr_pipeline = df_all_nr.pivot_table(index=['dataset', 'bs','QS','train_size'],
                                 columns=['pipeline'], values='rel_improv', aggfunc='mean')#.reset_index()

    pval_df = pd.DataFrame(columns=['test_name',  'friedman_stat', 'pval', 'kendall_w'])
    temp_stat, temp_pval = friedmanchisquare(*np.array(df_all_nr_pipeline).T)
    kendall_w_pipeline = kendall_w(np.array(df_all_nr_pipeline).T)
    pval_df = pd.concat([pval_df, pd.DataFrame({'test_name': ['pipeline'], 'friedman_stat': [temp_stat],
                                                          'pval': [temp_pval],
                                                'kendall_w': [kendall_w_pipeline]})], ignore_index=True)

    df_all_nr_qs = df_all_nr.pivot_table(index=['dataset', 'bs', 'pipeline', 'train_size'],
                                                        columns=['QS'], values='rel_improv',
                                                        aggfunc='mean')#.reset_index()
    temp_stat, temp_pval = friedmanchisquare(*np.array(df_all_nr_qs).T)
    kendall_w_qs = kendall_w(np.array(df_all_nr_qs.T))
    pval_df = pd.concat([pval_df, pd.DataFrame({'test_name': ['QS'], 'friedman_stat': [temp_stat],
                                                'pval': [temp_pval],
                                                'kendall_w': [kendall_w_qs]})], ignore_index=True)
    fname = f"friedman_pvals_rel_improv.csv"
    pval_df.to_csv(os.path.join(op_dir, fname), index=False)


def wilcoxon_test_rel_improv(df_all_orig, op_dir):
    df_all_orig['bs'] = [f"({r['batch_size']},{r['seed_size']})" for _, r in df_all_orig.iterrows()]
    print('===== Wilcoxon test: batch/seed size')
    df_all = df_all_orig[df_all_orig['QS'] != 'random'].copy()

    # average each subgroup in small size
    df_all = df_all.groupby(by=['dataset', 'bs', 'QS', 'pipeline', 'eff_train_size'], as_index=False). \
        agg({'rel_improv': 'mean'})
    df_bs = df_all.pivot_table(index=['eff_train_size', 'dataset', 'QS', 'pipeline'], columns=['bs'], values='rel_improv',
                               aggfunc='mean').reset_index()

    # Test whether two batch/seed size have same effect
    # order from small->large so we know what the direction of the test means
    bs_uniq = sorted(np.unique(df_all['bs']))
    print(bs_uniq)
    pval_bsize_df = pd.DataFrame(columns=['test_name', 'test_type', 'alternative', 'wcx_stat', 'pval'])
    test_type = 'batch_seed_size'

    method = 'auto'  # change this for accuracy vs speed tradeoff
    for alter, cat in itertools.product(['two-sided', 'less', 'greater'], ['pipeline', 'QS', None]):
        if cat is None:
            temp_stat, temp_pval = wilcoxon(x=df_bs[bs_uniq[0]], y=df_bs[bs_uniq[1]], alternative=alter, method=method)
            pval_bsize_df = pd.concat([pval_bsize_df, pd.DataFrame({'test_name': ['all'], 'test_type': [test_type],
                                                                    'alternative': [alter], 'wcx_stat': [temp_stat],
                                                                    'pval': [temp_pval]})], ignore_index=True)

        else:
            for cat_val in np.unique(df_all[cat]):
                temp_df = df_bs[df_bs[cat] == cat_val]
                print(f'{cat}: {cat_val}')

                temp_stat, temp_pval = wilcoxon(x=temp_df[bs_uniq[0]], y=temp_df[bs_uniq[1]], alternative=alter,
                                                method=method)
                pval_bsize_df = pd.concat([pval_bsize_df, pd.DataFrame({'test_name': cat_val, 'test_type': [test_type],
                                                                        'alternative': [alter], 'wcx_stat': [temp_stat],
                                                                        'pval': [temp_pval]})], ignore_index=True)

    fname = f"wilcoxon_batch_sizes_rel_improv.csv"
    pval_bsize_df.to_csv(os.path.join(op_dir, fname), index=False)
    print(pval_bsize_df)


def compute_feature_importance(df_all, op_dir):
    df_all['bs'] = [f"({r['batch_size']},{r['seed_size']})" for _, r in df_all.iterrows()]
    print('===== Feature importance')
    df_all_nr = df_all[df_all['QS'] != 'random']
    # print(df_all_nr)
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor()  #ExplainableBoostingClassifier()
    features = ['dataset', 'bs', 'QS', 'pipeline', 'train_size']
    X = df_all_nr[features]
    y = df_all_nr['rel_improv']
    print(y.min(), y.max(), df_all_nr[df_all_nr['rel_improv']==y.min()], '\n',df_all_nr[df_all_nr['rel_improv']==y.max()])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    ebm.fit(X_train, y_train)
    ebm_exp = ebm.explain_global()
    plotly_fig = ebm_exp.visualize()
    plotly_fig.write_image(f"{op_dir}/ebm.png")
    y_pred = ebm.predict(X_test)
    print(y_test.min(), y_test.max())
    results = {}
    results['RMSE'] = root_mean_squared_error(y_test, y_pred)
    results['MSE'] = mean_squared_error(y_test, y_pred)
    results['R2 score'] = r2_score(y_test, y_pred)
    imp_scores = {name: score for name, score in zip(ebm_exp.data()['names'], ebm_exp.data()['scores'])}

    results['importance_scores'] = {k:v for k, v in sorted(imp_scores.items(), key=lambda item: item[1])[::-1]}
    fname = f"ebm_feature_importance.json"
    with open(os.path.join(op_dir, fname),'w') as f:
        json.dump(results, f, indent=4)


def relative_improv_non_random_vs_random(df_all, op_dir, num_plots=4, heatmap_annot=True):
    '''For a given batch/seed size'''
    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    # Average over trial first
    df_all.drop(['init', 'ts', 'iter_idx'], axis=1, inplace=True)
    # print(df_all.shape)
    # print(df_all.head(5))
    df_all = df_all.groupby(by=['dataset', 'batch_size', 'seed_size', 'acq_name', 'clf', 'rep','train_size'], as_index=False). \
        agg({'score': 'mean'}) #(avg_score=pd.NamedAgg(column='score', aggfunc='mean'))
    # print(df_all[(df_all['dataset']=='agnews') & (df_all['train_size']==200)])

    # compute relative improvement F1
    rel_f1 = []
    f1_random = []
    for _, r in df_all.iterrows():
        temp_f1_random = df_all[(df_all['dataset']==r['dataset']) &
                                (df_all['batch_size']==r['batch_size']) &
                                (df_all['seed_size']==r['seed_size']) &
                                (df_all['acq_name']=='random') &
                                (df_all['clf']==r['clf']) &
                                (df_all['rep']==r['rep']) &
                                (df_all['train_size']==r['train_size'])]['score'].values[0]
        f1_random.append(temp_f1_random)
        rel_f1.append(100*(r['score']-temp_f1_random)/temp_f1_random)
    df_all['rel_improv'] = rel_f1
    df_all['F1_random'] = f1_random

    # Rename clf, get pipeline names, effective train sizes
    df_all['clf'] = df_all['clf'].map(clf_name_map)
    df_all['rep'] = df_all['rep'].map(rep_name_map)
    df_all['eff_train_size'] = df_all['train_size'].map(eff_size_map)
    df_all.rename(columns={'acq_name': 'QS'}, inplace=True)
    df_all['pipeline'] = [f"{r['clf']}-{r['rep']}" if r['clf'] != 'RoBERTa' else 'RoBERTa' for _, r in
                          df_all.iterrows()]
    # print(df_all.head())
    # print(df_all.tail())

    # Effect size: QS vs prediction pipline
    compute_feature_importance(df_all.copy(), op_dir)

    # Friedman test and Kendall's W
    friedman_test(df_all.copy(), op_dir)

    # Wilcoxon test
    wilcoxon_test_rel_improv(df_all.copy(), op_dir)

    # Compute NR-R stats for "ALWAYS ON"
    always_on_stats_df = pd.DataFrame(columns=['test_name', 'frac_less_than_random',
                                               'avg_rel_improve_geq_random', 'std_rel_improve_geq_random',
                                               'avg_rel_improve','std_rel_improve'])
    df_alwayson = df_all[df_all['QS'] != 'random'].copy()
    frac = df_alwayson[df_alwayson['rel_improv'] < 0].shape[0] / df_alwayson.shape[0]
    avg_rel_improve = np.mean(df_alwayson['rel_improv'])
    std_rel_improve = np.std(df_alwayson['rel_improv'])
    avg_rel_improve_geq_random = np.mean(df_alwayson[df_alwayson['rel_improv'] >= 0]['rel_improv'])
    std_rel_improve_geq_random = np.std(df_alwayson[df_alwayson['rel_improv'] >= 0]['rel_improv'])
    always_on_stats_df = pd.concat([always_on_stats_df, pd.DataFrame({'test_name': ['all'],
                                                                      'frac_less_than_random': [frac],
                                                                      'avg_rel_improve_geq_random': [
                                                                          avg_rel_improve_geq_random],
                                                                      'std_rel_improve_geq_random': [
                                                                          std_rel_improve_geq_random],
                                                                      'avg_rel_improve': [avg_rel_improve],
                                                                      'std_rel_improve': [std_rel_improve],
                                                                      })],
                                   ignore_index=True)

    for cat in ['pipeline', 'QS']:
        for cat_val in np.unique(df_all[cat]):
            if cat=='QS' and cat_val=='random':
                continue

            temp_df = df_alwayson[df_alwayson[cat] == cat_val]
            print('====', cat, cat_val)
            temp_frac = temp_df[temp_df['rel_improv'] < 0].shape[0] / temp_df.shape[0]
            temp_avg_rel_improve = np.mean(temp_df['rel_improv'])
            temp_std_rel_improve = np.std(temp_df['rel_improv'])
            temp_avg_rel_improve_geq_random = np.mean(temp_df[temp_df['rel_improv'] >= 0]['rel_improv'])
            temp_std_rel_improve_geq_random = np.std(temp_df[temp_df['rel_improv'] >= 0]['rel_improv'])
            always_on_stats_df = pd.concat([always_on_stats_df, pd.DataFrame({'test_name': [cat_val],
                                                                              'frac_less_than_random': [temp_frac],
                                                                              'avg_rel_improve_geq_random': [
                                                                                  temp_avg_rel_improve_geq_random],
                                                                              'std_rel_improve_geq_random': [
                                                                                  temp_std_rel_improve_geq_random],
                                                                              'avg_rel_improve': [temp_avg_rel_improve],
                                                                              'std_rel_improve': [temp_std_rel_improve]})],
                                           ignore_index=True)
    print('ALWAYS ON \n',always_on_stats_df)
    always_on_stats_df.to_csv(os.path.join(op_dir, 'always_on_stats_before_avg.csv'), index=False)

    fig, axn = results_utils.rel_improv_plot_template(n_heatmaps=num_plots)
    print('axes template', axn)
    # fig, axn = plt.subplots(1, num_plots, sharex=True, sharey=True,figsize=(20, 6))
    # fig.suptitle(f"Relative improvement in F1-macro over random", fontsize=18)
    # cbar_ax = fig.add_axes([.91, .3, .015, .4])
    eff_size_plot_map = {3: [2000, 3500, 5000],
                        4: [1500, 2500, 3500, 5000],
                         5: [1000, 2000, 3000, 4000, 5000]}
    eff_size_list = eff_size_plot_map[num_plots]
    heatmap_axes = {i: axn[i] for i in range(num_plots)}
    vmax = 0
    for idx, ax in heatmap_axes.items():
        temp_eff_size = eff_size_list[idx]
        df_size = df_all[df_all['eff_train_size']==temp_eff_size].copy()
        df_size = df_size.groupby(by=['pipeline', 'QS'], as_index=False).agg({'rel_improv': 'mean'})
        df_size = df_size[df_size['QS']!='random']

        vmax = max(vmax, max(-df_size['rel_improv'].min(), df_size['rel_improv'].max()))
        df_size = df_size.pivot(index='pipeline', columns='QS', values='rel_improv')
        sns.heatmap(data=df_size.sort_values(by='pipeline', key=lambda col: col.map(lambda c: pipeline_order.index(c))),
                    cmap="PiYG", vmax=vmax, vmin=-vmax, annot=heatmap_annot, annot_kws={'fontsize':14},
                    ax=ax, cbar= idx == num_plots-1) #, cbar_ax=None if idx else cbar_ax)  # cmap="crest"
        if idx == num_plots-1:
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=14)

        # ax.set_ylabel(f'Expected var. of F1 macro', fontsize=16)
        # bin_max = int(bin_name.split('-')[1].split('.')[0])
        ax.set_title(f'Train size: {temp_eff_size}', fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        if idx in [0]:
            ax.set_ylabel('Prediction Pipeline', fontsize=16)
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='both', which='major', labelsize=14)
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    # fname = f"rel_improv_f1"
    # for extn in ['png' , 'pdf','svg']:
    #     plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
    # plt.clf()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    sizes = np.arange(500, 5001, 500)
    # df_pipeline_sizes = df_all.groupby(by=['pipeline','eff_train_size'], as_index=False).agg({'rel_improv': 'mean'})
    # print(df_pipeline_sizes)
    # df_pipeline_sizes = df_pipeline_sizes.pivot(index='eff_train_size', columns='QS', values='rel_improv')
    sns.lineplot(data=df_all[df_all['QS']!='random'].sort_values(by='pipeline', key=lambda col: col.map(lambda c: pipeline_order.index(c))),
                 x="eff_train_size", y="rel_improv", hue="pipeline", ax=axn[num_plots])
    axn[num_plots].set_xlabel('train size', fontsize=16)
    axn[num_plots].set_ylabel('$\delta$ for a Prediction Pipeline', fontsize=16)
    axn[num_plots].set_title('Rel. improvement over random for Prediction Pipelines', fontsize=16)
    axn[num_plots].tick_params(axis='both', which='major', labelsize=14)
    plt.setp(axn[num_plots].get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(axn[num_plots].get_legend().get_title(), fontsize='16')  # for legend title

    df_qs_sizes = df_all[df_all['QS']!='random'].copy() #.groupby(by=['QS', 'eff_train_size'], as_index=False).agg({'rel_improv': 'mean'})
    sns.lineplot(data=df_qs_sizes.sort_values(by='pipeline', key=lambda col: col.map(lambda c: pipeline_order.index(c))),
                 x="eff_train_size", y="rel_improv", hue="QS", ax=axn[num_plots+1])
    axn[num_plots+1].set_xlabel('train size', fontsize=16)
    axn[num_plots + 1].set_ylabel('$\delta$ for a QS', fontsize=16)
    axn[num_plots + 1].set_title('Rel. improvement over random for QSes', fontsize=16)
    axn[num_plots + 1].tick_params(axis='both', which='major', labelsize=14)
    plt.setp(axn[num_plots + 1].get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(axn[num_plots + 1].get_legend().get_title(), fontsize='16')  # for legend title

    fname = f"all_rel_improv_f1"
    for extn in ['png', 'pdf','svg']:
        plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
    plt.clf()

    # fig, ax = plt.figure( figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df_all[(df_all['QS'] != 'random') &
                             (df_all['pipeline'] != 'RoBERTa')].sort_values(by='pipeline', key=lambda col: col.map(
        lambda c: pipeline_order.index(c))),
                 x="eff_train_size", y="rel_improv", hue="rep", ax=ax)
    ax.set_xlabel('train size', fontsize=16)
    ax.set_ylabel('$\delta$ for a Representation', fontsize=16)
    ax.set_title('Rel. improvement over random for Representations', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='16')  # for legend title
    fname = f"rel_improv_f1_representations"
    for extn in ['png', 'pdf', 'svg']:
        plt.savefig(f"{op_dir}/{fname}.{extn}", bbox_inches='tight')
    plt.clf()


def wilcoxon_NR_R(df_all,op_dir):
    method = 'auto'   # change this for accuracy vs speed tradeoff

    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        os.makedirs(op_dir)
    # Average over trial first
    df_all.drop(['init', 'ts', 'iter_idx'], axis=1, inplace=True)
    df_all = df_all.groupby(by=['dataset', 'batch_size', 'seed_size', 'acq_name', 'clf', 'rep','train_size'], as_index=False). \
        agg({'score': 'mean'})
    # Rename clf, get pipeline names, effective train sizes
    df_all['clf'] = df_all['clf'].map(clf_name_map)
    df_all['rep'] = df_all['rep'].map(rep_name_map)
    df_all['eff_train_size'] = df_all['train_size'].map(eff_size_map)
    df_all.rename(columns={'acq_name': 'QS'}, inplace=True)
    df_all['pipeline'] = [f"{r['clf']}-{r['rep']}" if r['clf'] != 'RoBERTa' else 'RoBERTa' for _, r in
                          df_all.iterrows()]
    df_all['bs'] = [f"({r['batch_size']},{r['seed_size']})" for _, r in df_all.iterrows()]

    # average each subgroup in small size
    df_all = df_all.groupby(by=['dataset', 'bs', 'QS', 'pipeline', 'eff_train_size'], as_index=False). \
        agg({'score': 'mean'})
    df_bs = df_all.pivot_table(index=['eff_train_size', 'dataset', 'QS', 'pipeline'], columns=['bs'], values='score',
                       aggfunc='mean').reset_index()
    # df_bs = df_all.pivot_table(index=['eff_train_size'], columns=['bs'], values='score',aggfunc='mean')
    print(df_bs.head())

    # Test whether two batch/seed size have same effect
    # order from small->large so we know what the direction of the test means
    bs_uniq = sorted(np.unique(df_all['bs']))
    print(bs_uniq)
    pval_bsize_df = pd.DataFrame(columns = ['test_name','test_type','alternative','wcx_stat', 'pval'])
    test_type = 'batch_seed_size'


    for alter, cat in itertools.product(['two-sided', 'less', 'greater'], ['pipeline','QS', None]):
        if cat is None:
            temp_stat, temp_pval = wilcoxon(x=df_bs[bs_uniq[0]], y=df_bs[bs_uniq[1]], alternative=alter, method=method)
            pval_bsize_df = pd.concat([pval_bsize_df, pd.DataFrame({'test_name': ['all'], 'test_type': [test_type],
                                                                    'alternative': [alter], 'wcx_stat': [temp_stat],
                                                                    'pval': [temp_pval]})], ignore_index=True)

        else:
            for cat_val in np.unique(df_all[cat]):
                temp_df = df_bs[df_bs[cat] == cat_val]
                print(f'{cat}: {cat_val}')

                temp_stat, temp_pval = wilcoxon(x=temp_df[bs_uniq[0]], y=temp_df[bs_uniq[1]], alternative=alter,method=method)
                pval_bsize_df = pd.concat([pval_bsize_df, pd.DataFrame({'test_name': cat_val, 'test_type': [test_type],
                                                                        'alternative': [alter], 'wcx_stat': [temp_stat],
                                                                        'pval': [temp_pval]})], ignore_index=True)

    fname = f"wilcoxon_batch_sizes.csv"
    pval_bsize_df.to_csv(os.path.join(op_dir, fname), index=False)
    print(pval_bsize_df)

    # Test if NR < R, i.e., "Always ON" mode.
    print('===== TEST NR < R, Always ON mode')
    df_all['nr_r'] = ['nr' if r['QS']!='random' else 'r' for _, r in df_all.iterrows()]

    # NOTE: we can't use QS in the index because QS can be "nr" or "r" but not both, which means one of those cols
    # will end up with nan.
    df_nr_r = df_all.pivot_table(index=['eff_train_size','dataset', 'pipeline', 'bs'],
                                 columns=['nr_r'], values='score', aggfunc='mean').reset_index()
    df_nr_r['rel_improve'] = 100. * (df_nr_r['nr'] - df_nr_r['r'])/df_nr_r['r']
    pval_nr_r_df = pd.DataFrame(columns=['test_name', 'test_type', 'alternative', 'wcx_stat', 'pval'])
    always_on_stats_df = pd.DataFrame(columns=['test_name', 'frac_less_than_random', 'avg_rel_improve_geq_random',
                                                'avg_rel_improve'])
    alter = 'less'
    test_type = 'nr_vs_r'

    temp_stat, temp_pval = wilcoxon(x=df_nr_r['nr'], y=df_nr_r['r'], alternative=alter,method=method)
    pval_nr_r_df = pd.concat([pval_nr_r_df, pd.DataFrame({'test_name': ['all'], 'test_type': [test_type],
                                                          'alternative': [alter], 'wcx_stat': [temp_stat],
                                                          'pval': [temp_pval]})], ignore_index=True)
    frac = np.sum(df_nr_r['nr'] < df_nr_r['r'])/ len(df_nr_r)
    avg_rel_improve = np.mean(df_nr_r['rel_improve'])
    avg_rel_improve_geq_random = np.mean(df_nr_r[df_nr_r['r'] <= df_nr_r['nr']]['rel_improve'])
    always_on_stats_df = pd.concat([always_on_stats_df, pd.DataFrame({'test_name': ['all'],
                                                            'frac_less_than_random': [frac],
                                                            'avg_rel_improve_geq_random': [avg_rel_improve_geq_random],
                                                            'avg_rel_improve': [avg_rel_improve]})], ignore_index=True)


    for cat in ['pipeline', 'QS']:
        for cat_val in np.unique(df_all[cat]):
            print('=========', cat_val)
            if cat=='QS' and cat_val=='random':
                continue

            if cat == 'QS':
                temp_df = df_all[df_all[cat].isin([cat_val, 'random'])]
                temp_df = temp_df.pivot_table(index=['eff_train_size', 'dataset', 'pipeline', 'bs'],
                                             columns=['nr_r'], values='score', aggfunc='mean').reset_index()

            else:
                temp_df = df_nr_r[df_nr_r[cat] == cat_val]
            temp_stat, temp_pval = wilcoxon(x=temp_df['nr'], y=temp_df['r'], alternative=alter,method=method)
            pval_nr_r_df = pd.concat([pval_nr_r_df, pd.DataFrame({'test_name': [cat_val], 'test_type': [test_type],
                                                                  'alternative': [alter], 'wcx_stat': [temp_stat],
                                                                  'pval': [temp_pval]})], ignore_index=True)

    fname = f"wilcoxon_nonrandom_vs_random.csv"
    pval_nr_r_df.to_csv(os.path.join(op_dir, fname), index=False)
    always_on_stats_df.to_csv(os.path.join(op_dir, 'always_on_stats.csv'), index=False)


    print(pval_nr_r_df)



if __name__ == "__main__":
    pass
    # result_df, aggr_df = collate_data(dirname=r'/media/aghose/DATA/sources/active_learning_baselines_with_data/'
    #                      r'active_learning_baselines/scratch/partial/seed_200_batch_size_200_iters_24')
    # factor_plot(result_df, aggr_df)
    # dal_times('/media/aghose/DATA/sources/active_learning_baselines_with_data/active_learning_baselines/'
    #           'scratch/seed_200_batch_size_200_iters_24/sst2_LinearSVC_MPNet_dal_trial_0/al.csv')

    # print(dirname_parse(r'scratch/current_results/final_non_BERT_seed_500_batch_size_500_iters_9/'
    #                     r'dbpedia5_LinearSVC_USE_dal_trial_2/al.csv', name_type='non_BERT'))
    # print(dirname_parse(r'scratch/current_results/final_BERT_seed_200_batch_size_200_iters_24/'
    #                     r'agnews_bs200_cal/trial_0/al.csv', name_type='BERT'))
    # collate_data(BERT_dirname=r'scratch/current_results/final_BERT_seed_200_batch_size_200_iters_24',
    #              non_BERT_dirname=r'scratch/current_results/final_non_BERT_seed_200_batch_size_200_iters_24',
    #              op_dir=f"{RESULTS_DIR}/collated", name_suffix='200')
    # collate_data(BERT_dirname=r'scratch/current_results/final_BERT_seed_500_batch_size_500_iters_9',
    #              non_BERT_dirname=r'scratch/current_results/final_non_BERT_seed_500_batch_size_500_iters_9',
    #              op_dir=f"{RESULTS_DIR}/collated", name_suffix='500')

    for b in [200, 500]:
        df_all_results = pd.read_csv(f"{RESULTS_DIR}/collated/all_data_{b}.csv")
        metric = 'F1'

        # df_aggr_results = pd.read_csv(f"{RESULTS_DIR}/collated/aggr_data_{b}.csv")
        # avg_acc(df_all_results, op_dir=f"{RESULTS_DIR}/stat_tests")
        # datawise_plots(df_all_results, op_dir=r'results/datawise_plots', suffix=f"{b}")
        # multipop_tests(df_all_results, df_aggr_results, remove_seed_step=True, op_dir=r'results/stat_tests',
        #                stat_name='val std', suffix=f"{b}")
        # multipop_tests(df_all_results, df_aggr_results, remove_seed_step=True, op_dir=r'results/stat_tests',
        #                stat_name='AUC std', suffix=f"{b}")
        # acq_wilcoxon_and_rmse(df_all=df_all_results, df_aggr=df_aggr_results,
        #                       op_dir=r'results/stat_tests',
        #                       min_sample_for_stat=3)

    # combinatorial_error({'dal': np.random.random((3, 4)), 'cal': np.random.random((2, 4))},
    #                    np.random.random((5, 4)), error_fn=rmse)

    # problem_with_wilcoxon(op_dir=r'results/assets')
    # pdp([(200, pd.read_csv(f"{RESULTS_DIR}/collated/all_data_200.csv"),
    #       pd.read_csv(f"{RESULTS_DIR}/collated/aggr_data_200.csv")),
    #      (500, pd.read_csv(f"{RESULTS_DIR}/collated/all_data_500.csv"),
    #       pd.read_csv(f"{RESULTS_DIR}/collated/aggr_data_500.csv"))
    #      ], op_dir=r'results/pdp')

    df_all_both_batches = pd.concat([pd.read_csv(f"{RESULTS_DIR}/collated/all_data_200.csv"),
                                     pd.read_csv(f"{RESULTS_DIR}/collated/all_data_500.csv")])

    # wilcoxon_NR_R(df_all_both_batches.copy(), op_dir=f"{RESULTS_DIR}/wilcoxon_pvals")
    relative_improv_non_random_vs_random(df_all_both_batches.copy(),
                                         op_dir=f"{RESULTS_DIR}/rel_improv_f1",
                                         num_plots=5, heatmap_annot=True)

    # auc_heatmap_non_random_vs_random(df_all_both_batches, num_train_bins=4, op_dir=f"{RESULTS_DIR}/auc_heatmap",
    #                                   diff_type='relative')

    # # avg_acc(df_all_both_batches, op_dir=f"{RESULTS_DIR}/stat_tests", remove_seed_step=False)
    # best_combination(df_all_both_batches, num_train_bins=4, op_dir=f"{RESULTS_DIR}/misc")

    # acq_wilcoxon_and_rmse(df_all=df_all_both_batches,
    #                       op_dir=r'results/stat_tests',
    #                       min_sample_for_stat=3,
    #                       remove_seed_step=True)
