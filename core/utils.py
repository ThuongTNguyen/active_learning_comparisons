import datetime
import string

from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import os, joblib, scipy, itertools, logging
import sqlite3, json
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import seaborn.objects as so
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from collections import Counter
from itertools import product
import numpy as np, pandas as pd, datetime
USE_module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class PersistentStringTransformer(BaseEstimator, TransformerMixin):
    """
    Create a transformer class which does not need to refit if the passed in data is already fit to it.
    It avoids the refit by looking up a db.
    """
    def __init__(self, dbfile_path, *args, **kwargs):
        self.dbfile_path = dbfile_path
        super(PersistentStringTransformer, self).__init__(*args, **kwargs)
        self.table_name = "train_data_models"

    def fit(self, X, y=None):
        """
        :param X: 
        :param y: 
        :return: 
        """
        con = sqlite3.connect(self.dbfile_path)
        cur = con.cursor()
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}';")
        res = cur.fetchone()
        if res is None:
            cur.execute(f"CREATE TABLE {self.table_name}(ID integer, train_text text, model blob)")
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = np.random.randint(0, 10, X.shape[0])
        return X



class SVCWithEmbeddings(BaseEstimator):
    '''Class for embeddings+classifier type of model. Fairly flexible, so it can be used in a lot of scenarios.'''
    def __init__(self, detect_ood=False, detect_rejection=False, embedding_technique='ngrams', **params):
        """
        NOTE: do not instantiate any model in init! scikit learn doesn't honor them - do it in fit.
        :param detect_ood: do we need to detect out of distrbution errors? Currently, if True, this is done
            only using Isolation Forest.
        :param detect_rejection: you can detect if all classifiers have rejected an instance, and declare an outlier
            based on that. This test is applied before the ood test if both flags are true.
        :param embedding_model: what model to embed text with. Ideally I would have liked to pass in the embedding
        model itself but there seems to be some issue with running TF function within GridSearchCV. For now, I'm
        selecting a model based on a keyword string (see fit()).
        :param params:
        """
        self.detect_ood = detect_ood
        self.detect_rejection = detect_rejection
        self.embedding_technique = embedding_technique
        self.embedding_model = None
        self.model_filename = 'use_svc.joblib'
        self.params_dict = params
        self.ood_model =None

    def load_model(self, model_dir='scratch'):
        with open(os.path.join(model_dir, self.model_filename), 'rb') as f:
            model = joblib.load(f)
        self.model = model
        self.class_names = self.model.classes_
        self.params_dict = self.model.get_params()

    def vectorize(self, X, mode=None):
        """
        Move all vectorization steps in here, so we don't have to track them separaely for fit(), predict(), etc.
        :param mode:
        :return:
        """
        X_vec = None
        if self.embedding_technique == 'ngrams':
            if mode == "fit_transform":
                self.embedding_model = CountVectorizer(ngram_range=(1, 3))
                X_vec = self.embedding_model.fit_transform(X)
            elif mode == "transform":
                X_vec = self.embedding_model.transform(X)
            else:
                logging.error(f"Can't parse mode(=={mode}).")
        elif self.embedding_technique == 'muse':  # mode doesn't matter since this isn't being fitted
            self.embedding_model = muse_model
            X_vec = self.embedding_model(X.flatten())
        else:
            logging.error(f"Cant parse model technique=={self.embedding_technique}.")
        return X_vec

    def fit(self, X, y):
        """
        Remember to handle the OOD case here. Also detect_rejection doesn't require special handling.
        :param X:
        :param y:
        :return:
        """
        X_vec = self.vectorize(X, mode='fit_transform')
        self.model = LinearSVC(**self.params_dict)
        self.model.fit(X_vec, y)
        # the calibration functions check if a class is fitted based on the presence of variables with an underscore
        # at the end - so I am creating a new variable called classes_ .
        self.classes_  = self.model.classes_
        if self.detect_ood:
            self.ood_model = IsolationForest().fit(X_vec)

    def predict(self, X):
        X_vec = self.vectorize(X, mode='transform')
        y = self.model.predict(X_vec)

        if self.detect_rejection:
            dec_matrix = self.model.decision_function(X_vec)
            y[np.all(dec_matrix < 0, axis=1)] = outlier_label

        if self.detect_ood:
            # -1 is for outliers as per docs
            y_ood = self.ood_model.predict(X_vec)
            y[y_ood==-1] = outlier_label

        return y

    def decision_function(self, X):
        X_vec = self.vectorize(X, mode='transform')
        return self.model.decision_function(X_vec)


    def get_params(self, deep=True):
        return self.params_dict

    def set_params(self, **params):
        self.params_dict = params
        return self

    def save_to_file(self, model_dir):
        mkdir_if_not_exists(model_dir)
        joblib.dump(self.model, os.path.join(model_dir,self.model_filename))



class MyLightGBM(lgb.LGBMClassifier):
    def fit(self, X_train, y_train, X_val, y_val):
        """This version accepts a val set, in a way required by model_selection.select_model"""
        super().fit(X_train, y_train, eval_set=[(X_val, y_val)])

class MyCountVectorizer(CountVectorizer):
    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y).astype(dtype=np.float32).toarray()
    def transform(self, raw_documents):
        return super().transform(raw_documents).astype(dtype=np.float32).toarray()


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


if __name__ == "__main__":
    pass
    db_file_path = r'../scratch/test_db'
    p = PersistentStringTransformer(db_file_path)
    X = np.array(['hi', 'hello'])
    p.fit(X)