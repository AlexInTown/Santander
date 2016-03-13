# -*- coding: utf-8 -*-
import os
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
        self.random_state = 325243  # do not change it for different l1 models!
        if not train_fname:
            train_fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_train.csv')
        if not test_fname:
            test_fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_test.csv')
        # load train data
        train = pd.read_csv(train_fname)
        train.sort(columns='ID', inplace=1)
        self.XID = train.values
        self.Y = train.TARGET.values
        self.X = train.drop(['ID', 'TARGET'], axis=1)
        # load test data
        test = pd.read_csv(test_fname)
        self.YID = test.ID.values
        self.test = test.drop(['ID'], axis=1)
        pass

    def cross_validation(self, model):
        kfold = cross_validation.KFold(self.X.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = np.zeros(len(self.Y))
        i = 0
        for train_idx, test_idx in kfold:
            print (' --------- fold {0} ---------- '.format(i))
            train_x = self.X.iloc[train_idx]
            train_y = self.Y[train_idx]
            test_x = self.X.iloc[test_idx]
            test_y = self.Y[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred)
            preds[test_idx] = pred
            scores.append(score)
            i += 1
        scores = np.asarray(scores)
        print scores.mean(), scores.std()
        return scores, preds

    def fit_fullset_and_predict(self, model):
        model.fit(self.X, self.Y)
        preds = model.predict(self.test)
        return preds


    def param_selection(self, params):
        pass