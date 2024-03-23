import logging
import re, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

DATA_BASE_PATH = r'data'
DATA_FILE_MAP = {
   # dbpedia, source: https://drive.google.com/uc?id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&export=download
    'dbpedia': {'train': 'dbpedia/train.csv', 'test': 'dbpedia/test.csv'},

    # Source https://nlp.stanford.edu/sentiment/
    'sst2': {'train': 'SST-2/train.tsv', 'test': 'SST-2/test.tsv', 'dev': 'SST-2/dev.tsv'},

    # Source https://ai.stanford.edu/~amaas/data/sentiment/
    'imdb': {'train': 'imdb/train.csv', 'test': 'imdb/test.csv'},

    # Source https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn
    # /rnn_bi_multilayer_lstm_own_csv_agnews.ipynb
    # https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
    'agnews': {'train': 'agnews/train.csv', 'test': 'agnews/test.csv'},

    'pubmed': {'train': 'pubmed/train.csv', 'test': 'pubmed/test.csv', 'dev': 'pubmed/dev.csv'},

    'searchsnippets': {'train': 'data-web-snippets/train.txt', 'test': 'data-web-snippets/test.txt'}

}


def load_data(dataset_name, max_train_instances=None, max_test_instances=None, random_state=1, min_class_rep=5):
    """
    A data loading function that works based on the dataset name. max_train_instances & max_test_instances
    denote the max train/test dataset sizes that's to be returned for the dataset: a subset is selected that is
    stratified wrt labels for each of train and test separately. Cleanup, massaging
    to a common format, etc, are performed here. min_class_rep affects the training subset only, and removes from this
    subset text for labels
    dataset_name: name of dataset, to be loaded based on DATA_FILE_MAP.
        Current available:
        * sst2
        * dbpedia
        * agnews
        * pubmed
        * imdb
    max_instances: max datasize to be returned; if dataset is larger, a subset is selected. None implies the dataset
        is returned as-is, i.e., size is determined by # instances in the file.
    random_state: for reproducibility.
    min_class_rep: If a class doesn't have these many instances in the selected subset, its instances will be dropped.
        Note that this # is ensured in the returned subset and not the whole dataset.
    """
    if dataset_name == 'sst2':
        # like in the CAL paper, we will use the dev set as test, because I'm not sure if the test set labels floating
        # around on the internet are correct!
        d_train, d_test = [pd.read_csv(os.sep.join((DATA_BASE_PATH, DATA_FILE_MAP[dataset_name][d])), sep='\t')
                                  for d in ('train', 'dev')]
        X_train, y_train = d_train['sentence'].to_numpy(), d_train['label'].to_numpy(dtype=int)
        X_test, y_test = d_test['sentence'].to_numpy(), d_test['label'].to_numpy(dtype=int)

    elif dataset_name == 'pubmed':
        d_train, d_dev, d_test = [pd.read_csv(os.sep.join((DATA_BASE_PATH, DATA_FILE_MAP[dataset_name][d])))
                                  for d in ('train', 'dev', 'test')]
        d_train = pd.concat((d_train, d_dev))
        X_train, y_train = d_train['text'].to_numpy(), d_train['label'].to_numpy(dtype=int)
        X_test, y_test = d_test['text'].to_numpy(), d_test['label'].to_numpy(dtype=int)

    elif dataset_name == 'agnews':
        d_train, d_test = [pd.read_csv(os.sep.join((DATA_BASE_PATH, DATA_FILE_MAP[dataset_name][d])))
                                  for d in ('train', 'test')]
        d_train['concat_text'] = [f"{title} {desc}" for title, desc in zip(d_train['Title'], d_train['Description'])]
        d_test['concat_text'] = [f"{title} {desc}" for title, desc in zip(d_test['Title'], d_test['Description'])]
        X_train, y_train = d_train['concat_text'].to_numpy(), d_train['Class Index'].to_numpy(dtype=int)
        X_test, y_test = d_test['concat_text'].to_numpy(), d_test['Class Index'].to_numpy(dtype=int)

    elif dataset_name in ['dbpedia', 'dbpedia5']:
        dataset_key = 'dbpedia'
        d_train, d_test = [pd.read_csv(os.sep.join((DATA_BASE_PATH, DATA_FILE_MAP[dataset_key][d])),
                                       header=None, names=['label', 'title', 'desc'])
                                       for d in ('train', 'test')]
        d_train['concat_text'] = [f"{title} {desc}" for title, desc in zip(d_train['title'], d_train['desc'])]
        d_test['concat_text'] = [f"{title} {desc}" for title, desc in zip(d_test['title'], d_test['desc'])]
        X_train, y_train = d_train['concat_text'].to_numpy(), d_train['label'].to_numpy(dtype=int)
        X_test, y_test = d_test['concat_text'].to_numpy(), d_test['label'].to_numpy(dtype=int)
        if dataset_name == 'dbpedia5':
            valid_labels = [1, 2, 3, 4, 5]
            X_train, y_train, X_test, y_test =  X_train[np.isin(y_train, valid_labels)],\
                                                y_train[np.isin(y_train, valid_labels)],\
                                                X_test[np.isin(y_test, valid_labels)],\
                                                y_test[np.isin(y_test, valid_labels)]

    elif dataset_name == 'imdb':
        d_train, d_test = [pd.read_csv(os.sep.join((DATA_BASE_PATH, DATA_FILE_MAP[dataset_name][d])))
                           for d in ('train', 'test')]
        X_train, y_train = d_train['text'].to_numpy(), d_train['label'].to_numpy(dtype=int)
        X_test, y_test = d_test['text'].to_numpy(), d_test['label'].to_numpy(dtype=int)

    elif dataset_name == 'searchsnippets':
        label_to_id = dict()
        X_train, y_train, X_test, y_test = None, None, None, None
        for d in ('train', 'test'):
            fpath = os.sep.join((DATA_BASE_PATH, DATA_FILE_MAP[dataset_name][d]))
            with open(fpath, encoding="utf8") as fr:
                lines =  fr.readlines()

            texts, labels = [], []
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                toks = re.split('\s+', line)
                text, label = " ".join(toks[:-1]), toks[-1]
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id) + 1
                label_id = label_to_id[label]
                texts.append(text)
                labels.append(label_id)

            if d == 'train':
                X_train, y_train = np.array(texts), np.array(labels, dtype=int)
            if d == 'test':
                X_test, y_test = np.array(texts), np.array(labels, dtype=int)

    else:
        logging.error(f"Can't process dataset {dataset_name}.")
        return

    logging.info(f"Loaded dataset={dataset_name}.")

    # calculate the class ratio difference between train and test
    n_train, n_test = len(X_train), len(X_test)
    test_class_ratios = dict([(k, 1.*v/n_test) for k, v in Counter(y_test).items()])
    train_class_ratios = dict([(k, 1. * v / n_train) for k, v in Counter(y_train).items()])
    rmse_class_ratios = 0
    for k in set(train_class_ratios.keys()).union(test_class_ratios.keys()):
        rmse_class_ratios += (train_class_ratios.get(k, 0) - test_class_ratios.get(k, 0))**2
    rmse_class_ratios = np.sqrt(rmse_class_ratios)
    print(f"\ndataset={dataset_name}\ntrain classes (size={len(X_train)}): {train_class_ratios}\n"
          f"test classes (size={len(X_test)}): {test_class_ratios}\n"
          f"RMSE train/test class ratios: {rmse_class_ratios:.2f}")

    # honor min_class_rep and max_instances
    if max_train_instances is None:
        max_train_instances = len(X_train)
    if max_test_instances is None:
        max_test_instances = len(X_test)

    labels_to_keep = [k for k, v in train_class_ratios.items() if int(v * max_train_instances) >= min_class_rep]
    keep_idxs = np.isin(y_train, labels_to_keep)
    X_train, y_train = X_train[keep_idxs], y_train[keep_idxs]
    if max_train_instances < len(X_train):
        X_train, _, y_train, _ = train_test_split(X_train, y_train, stratify=y_train, train_size=max_train_instances,
                                                  random_state=random_state)
    if max_test_instances < len(X_test):
        X_test, _, y_test, _ = train_test_split(X_test, y_test, stratify=y_test, train_size=max_test_instances,
                                                random_state=random_state)
    return X_train, y_train, X_test, y_test


def multiple_splits(X, y, split_fractions, random_state):
    """
    scikit's train_test_split only provides 2 splits, here it is used recursively to provide multiple splits.
    X: instances
    y: labels
    split fractions: a list of +ve fractions that sum to 1.
    random_state: for reproducibility in splits
    *NOTE*: As opposed to scikit this generates data splits in the form [(X1, y1), (X2, y2), (X3, y3), ...].
    """
    splits = []
    X_next, y_next = X, y
    for i in range(len(split_fractions) - 1):
        total_remaining = sum(split_fractions[i:])
        train_pct = split_fractions[i] / total_remaining
        Xi, X_next, yi, y_next = train_test_split(X_next, y_next, stratify=y_next, train_size=train_pct,
                                                  random_state=random_state)
        splits.append((Xi, yi))
    splits.append((X_next, y_next))
    return splits


def generate_datasets_for_multiple_trials(X, y, num_trials=1, split_fractions=None):
    """
    This yields the splits, so you can use it within the experiment loop.
    *NOTE*: As opposed to scikit this generates data splits in the form [(X1, y1), (X2, y2), (X3, y3), ...].
    """
    if split_fractions is None:
        split_fractions = [0.6, 0.2, 0.2]
    for trial_idx in range(num_trials):
        splits = multiple_splits(X, y, split_fractions, random_state=trial_idx)
        yield trial_idx, splits


def demo_usage():
    """
    Note: Although '20newsgroups' is supported it has a lot of classes, and might make training time-consuming for
    cases with one-vs-rest classifier.
    """
    max_instances = 10000  # the dataset size we'd like to work with
    min_class_rep = 20  # for a class to be present at least these many instances must be present
    for dataset_name in ['sst2', 'sst5', 'ag_news', 'imdb', 'fakenews', 'yelp_review_polarity', 'sentiment140']:
        X, y = load_data(dataset_name, max_instances=max_instances, min_class_rep=min_class_rep)
        c = Counter(y)
        h = dict([(k, 1. * v / len(X)) for k, v in c.items()])
        label_count_str = ", ".join([f"{k}:{v}" for k, v in sorted(c.items())])
        distr_str = ", ".join([f"{k}:{v:0.02f}" for k, v in sorted(h.items())])

        # calculate the average string length using top 1000 samples
        avg_str_length = np.mean([len(i) for i in X[:1000]])
        print(f"Loaded dataset {dataset_name}.\n\t# instances={len(X)}\n\tunique labels={set(y)}"
              f"\n\tlabel distr.={distr_str}\n\tlabel counts={label_count_str}"
              f"\n\tavg. string length (in chars)={avg_str_length}")

    # let's see how multiple trials work
    N = 5000
    dataset_name = 'sentiment140'
    print(f"Fitting classifier over multiple trials on dataset={dataset_name}.")
    X, y = load_data(dataset_name, max_instances=N, min_class_rep=min_class_rep)

    # It is OK to use CountVectorizer() on the whole dataset since it doesn't use dataset level statistics. tf-idf
    # shouldn't be used in this manner since idf is a dataset level statistic that should be derived only from the
    # training data and therefore, something like a Pipeline object should be used.
    X_vec = CountVectorizer().fit_transform(X)
    print(f"Shape of X: {np.shape(X_vec)}")

    search_space = {'max_depth': [2, 5, 10]}
    scores, best_params = [], []
    num_trials = 3
    for trial_idx, splits in generate_datasets_for_multiple_trials(X_vec, y, num_trials=num_trials,
                                                                   split_fractions=[0.8, 0.2]):
        print(f"# trial={trial_idx + 1}:")
        X_train, y_train = splits[0]
        X_test, y_test = splits[1]
        clf = DecisionTreeClassifier(class_weight='balanced')
        grid_clf = GridSearchCV(clf, param_grid=search_space, refit=True, cv=3, verbose=10)
        grid_clf.fit(X_train, y_train)
        score = f1_score(y_test, grid_clf.best_estimator_.predict(X_test), average='macro')
        print(f"Test score: {score:.02f}")
        scores.append(score)
        best_params.append(str(grid_clf.best_params_))
    temp_sep = "\n\t\t"
    print(f"Finished {num_trials} trials:\n\tAvg F1={np.mean(scores):0.4f}"
          f"\n\tstddev={np.std(scores):0.04f}\n\tbest params:{temp_sep}{temp_sep.join(best_params)}")


def show_sample_data():
    # print some examples from each dataset
    num_examples = 5
    for dataset in ['searchsnippets', 'sst2', 'dbpedia5', 'agnews', 'imdb', 'pubmed']:
        print(f"\n\n=============\ndataset={dataset}")
        X_train, y_train, X_test, y_test = load_data(dataset, max_train_instances=None, max_test_instances=None,
                                                     min_class_rep=1)

        print(f"train size={np.shape(X_train)}, test size={np.shape(X_test)}, num classes={len(set(y_train))}")
        print(f"\nTrain examples:")
        print("\n".join([f"[{z + 1}] label={j}::{i}" for z, (i, j) in
                         enumerate(zip(X_train[:num_examples], y_train[:num_examples]))]))
        print(f"\nTest examples:")
        print("\n".join([f"[{z + 1}] label={j}::{i}" for z, (i, j) in
                         enumerate(zip(X_test[:num_examples], y_test[:num_examples]))]))


if __name__ == "__main__":
    pass
