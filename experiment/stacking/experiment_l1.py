# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
from utils.config_utils import Config


class ExperimentL1:
    """
    Level 1 experiment wrapper for model stacking.
    """

    def __init__(self, train_fname=None, test_fname=None):
        #self.random_state = 325243  # do not change it for different l1 models!
        self.random_state = 98754  # do not change it for different l1 models!
        if not train_fname:
            train_fname = 'filtered_train.csv'
        if not test_fname:
            test_fname = 'filtered_test.csv'
        train_fname = os.path.join(Config.get_string('data.path'), 'input', train_fname)
        test_fname = os.path.join(Config.get_string('data.path'), 'input', test_fname)
        # load train data
        train = pd.read_csv(train_fname)
        train.sort(columns='ID', inplace=1)
        self.train_id = train.ID.values
        self.train_y = train.TARGET.values
        self.train_x = train.drop(['ID', 'TARGET'], axis=1)
        # load test data
        test = pd.read_csv(test_fname)
        self.test_id = test.ID.values
        self.test_x = test.drop(['ID'], axis=1)
        pass

    def cross_validation(self, model, n_folds=5):
        # kfold = cross_validation.KFold(self.train_x.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=n_folds, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = np.zeros(len(self.train_y))
        i = 0
        for train_idx, test_idx in kfold:
            print ' - fold {0}  '.format(i),
            start = time.time()
            train_x = self.train_x.iloc[train_idx]
            train_y = self.train_y[train_idx]
            test_x = self.train_x.iloc[test_idx]
            test_y = self.train_y[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred)
            preds[test_idx] = pred
            scores.append(score)
            end = time.time()
            print (' score:{}  time:{}s.'.format(score, end - start))
            i += 1
        scores = np.asarray(scores)
        print scores.mean(), scores.std(), metrics.roc_auc_score(self.train_y, preds)
        return scores, preds

    def fit_fullset_and_predict(self, model):
        model.fit(self.train_x, self.train_y)
        preds = model.predict(self.test_x)
        return preds

    def param_selection(self, params):
        pass