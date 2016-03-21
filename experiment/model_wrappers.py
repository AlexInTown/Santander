# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

import numpy as np
import xgboost as xgb

class XgboostModel:

    def __init__(self, model_params, train_params=None, test_params=None):
        self.model_params = model_params
        if train_params:
            self.train_params = train_params
        else:
            self.train_params = {"num_boost_round": 300 }
        self.test_params = test_params
        fname_parts = ['xgb']
        fname_parts.extend(['{0}#{1}'.format(key, val) for key,val in model_params.iteritems()])
        self.model_out_fname = '-'.join(fname_parts)

    def fit(self, X, y):
        """Fit model."""
        dtrain = xgb.DMatrix(X, label=np.asarray(y))
        #bst, loss, ntree = xgb.train(self.model_params, dtrain, num_boost_round=self.train_params['num_boost_round'])
        self.bst = xgb.train(self.model_params, dtrain, num_boost_round=self.train_params['num_boost_round'])
        #self.loss = loss
        #self.ntree = ntree
        #print loss, ntree

    def predict(self, X):
        """Predict using the xgb model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)

    def to_string(self):
        return self.model_out_fname


class SklearnModel:
    def __init__(self, model_params):
        self.model_params = model_params
        self.model_class = model_params['model_type']
        del model_params['model_type']
        fname_parts = [self.model_class.__name__]
        fname_parts.extend(['{0}#{1}'.format(k,v) for k,v in model_params.iteritems()])
        self.model = self.model_class(**self.model_params)
        self.model_out_fname = '-'.join(fname_parts)

    def fit(self, X, y):
        """Fit model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict using the sklearn model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self.model.predict(X)

    def to_string(self):
        return self.model_out_fname

