# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
from model_wrappers import SklearnModel, XgboostModel


class GridSearch:
    def __init__(self, wrapper_type, experiment, model_param_keys, model_param_vals):
        """
        Constructor of grid search.
        Support search on a set of model parameters, and record the cv result of each param configuration.
        :param wrapper_type: model wrapper type string like 'XgboostModel' or 'SklearnModel'
        :param experiment: experiment object of ExperimentL1 or ExperimentL2
        :param model_param_keys: list of model param keys. eg. ['paramA', 'paramB', 'paramC']
        :param model_param_vals: list of model param values (iterable). eg. [['valAa', 'valAb'], [0.1, 0.2], (1, 2, 3)]
        :return: None
        """
        self.wrapper_type = wrapper_type
        if wrapper_type == 'SklearnModel':
            self.model_name = model_param_vals[0]
        else:
            self.model_name = 'xgb'
        self.best_param = {}
        pass

    def search_by_cv(self, cv_out=None, cv_pred_out=None, refit_pred_out=None):
        """
        Search by cross validation
        :param cv_out: Output pickle file name of cross validation score results.
        :param cv_pred_out: prediction of cross validation each fold.
        :param refit_pred_out: refit on full train set and predict on test set.
        :return: None
        """
        pass

    def to_string(self):
        return self.model_name