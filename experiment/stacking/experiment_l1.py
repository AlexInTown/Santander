# -*- coding: utf-8 -*-
import os
import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics


class ExperimentL1:
    """
    Level 1 experiment wrapper for model stacking.
    """

    def __init__(self):
        self.random_state = 325243  # do not change it!

        # load train data

        pass

    def cross_validation(self, model):
        kfold = cross_validation.KFold(self.X.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = list()
        i = 0
        for train_idx, test_idx in kfold:
            print 'fold ', i
            train_x = self.X.iloc[train_idx]
            train_y = self.Y.iloc[train_idx]
            test_x = self.X.iloc[test_idx]
            test_y = self.Y.iloc[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred)
            preds.append(pred)
            scores.append(score)
            i += 1
        scores = np.asarray(scores)
        print scores.mean(), scores.std()
        return scores, preds

    def fit_fullset(self):
        pass

    def final_predict(self):
        pass

    def param_selection(self, params):
        pass