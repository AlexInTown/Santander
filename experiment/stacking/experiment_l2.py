# -*- coding: utf-8 -*-
import os
import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics


class ExperimentL2:
    def __init__(self):
        # TODO load model + result pkl, use the model to get output attributes
        # traverse the same pkl file pattern
        # parse the meta feature
        # make dataset
        # cross validation to select out parameters
        self.random_stat = 2788863
        pass

    def cross_validation(self, model):
        kfold = cross_validation.KFold(self.X.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = list()
        for train_idx, test_idx in kfold:
            train_x = self.X.iloc[train_idx]
            train_y = self.Y.iloc[train_idx]
            test_x = self.X.iloc[test_idx]
            test_y = self.Y.iloc[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred) # ???
            preds.append(pred)
            scores.append(score)
        scores = np.asarray(scores)
        print scores.mean(), scores.std()
        return scores, preds


