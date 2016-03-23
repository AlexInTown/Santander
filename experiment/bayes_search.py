# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import time
import cPickle as cp
import itertools
from model_wrappers import SklearnModel, XgboostModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class BayesSearch:
    def __init__(self, wrapper_class, experiment, model_param_keys, model_param_space,
                 cv_out=None, cv_pred_out=None, refit_pred_out=None, dump_round = 10):
        """
        Constructor of grid search.
        Support search on a set of model parameters, and record the cv result of each param configuration.

        :param wrapper_class: model wrapper type string like 'XgboostModel' or 'SklearnModel'
        :param experiment: experiment object of ExperimentL1 or ExperimentL2
        :param model_param_keys: list of model param keys. eg. ['paramA', 'paramB', 'paramC']
        :param model_param_space: list of model param space
        :return: None
        """

        self.wrapper_class = wrapper_class
        self.experiment = experiment
        self.model_param_keys = model_param_keys
        self.model_param_space = model_param_space
        self.model_name = self.wrapper_class.__name__

        self.cv_out = cv_out
        self.cv_pred_out = cv_pred_out
        self.refit_pred_out = refit_pred_out

        self.eval_round = 0
        self.dump_round = dump_round
        self.trials = Trials()
        pass

    def objective(self, param_dic):
        if self.eval_round % self.dump_round == 0:
            self.dump_result()
        self.eval_round += 1
        print param_dic
        model = self.wrapper_class(param_dic)
        scores, preds = self.experiment.cross_validation(model)
        return {
            'loss': scores.mean(),
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            # -- attachments are handled differently
            'attachments':
                {'scores': scores, 'preds': preds}
        }

    def dump_result(self):
        if self.cv_pred_out:
            preds_list = list()
            for dic in self.trials.trials:
                preds = self.trials.trial_attacments(dic)['preds']
                preds_list.append(preds)
            cp.dump(preds_list, open(self.cv_pred_out, 'wb'), protocol=2)
        if self.cv_out:
            scores_list = list()
            for dic in self.trials.trials:
                scores = self.trials.trial_attacments(dic)['scores']
                scores_list.append(scores)
            param_vals_list = [ [dic[k] for k in self.model_param_keys] for dic in self.trials.trials]
            cp.dump((self.model_param_keys, param_vals_list, scores_list), open(self.cv_out, 'wb'), protocol=2)

        if self.refit_pred_out:
            self.fit_full_set_and_predict(self.refit_pred_out)
        pass

    def search_by_cv(self, max_evals=100):
        best = fmin(self.objective, space=self.model_param_space, algo=tpe.suggest, max_evals=max_evals, trials=self.trials)
        print 'Best Param:'
        print best
        pass

    def fit_full_set_and_predict(self, refit_pred_out):
        pass