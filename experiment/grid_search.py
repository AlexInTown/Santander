# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import cPickle as cp
import itertools

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
        self.experiment = experiment
        self.model_param_keys = model_param_keys
        self.model_param_vals = model_param_vals

        if wrapper_type == 'SklearnModel':
            self.model_name = model_param_vals[0]
        else:
            self.model_name = 'xgb'
        pass

    def search_by_cv(self, cv_out, cv_pred_out=None, refit_pred_out=None):
        """
        Search by cross validation
        :param cv_out: Output pickle file name of cross validation score results.
        :param cv_pred_out: prediction of cross validation each fold.
        :param refit_pred_out: refit on full train set and predict on test set.
        :return: None
        """
        # create dataframe of results
        scores_list = []
        preds_list = []
        param_vals_list = []
        for v in itertools.product(*self.model_param_vals):
            param_dic = {}
            for i in xrange(len(self.model_param_keys)):
                param_dic[self.model_param_keys[i]] = v[i]
            print param_dic
            model = eval('{0}({1})'.format(self.wrapper_type, param_dic))
            scores, preds = self.experiment.cross_validation(model)
            scores_list.append(scores)
            preds_list.append(preds)
            param_vals_list.append(v)

        if cv_pred_out:
            cp.dump(preds_list, open(cv_pred_out, 'wb'), protocol=2)

        cp.dump((self.model_param_keys, param_vals_list, scores_list), open(cv_out, 'wb'), protocol=2)

        if refit_pred_out:
            preds_list = []
            for v in itertools.product(*self.model_param_vals):
                param_dic = {}
                for i in xrange(len(self.model_param_keys)):
                    param_dic[self.model_param_keys[i]] = v[i]
                model = eval('{0}({1})'.format(self.wrapper_type, param_dic))
                preds = self.experiment.fit_fullset_and_predict(model)
                preds_list.append(preds)
            cp.dump(preds_list, open(refit_pred_out, 'wb'), protocol=2)
        pass

    def to_string(self):
        return self.model_name+ '_cv_'