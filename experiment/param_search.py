# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import time
import cPickle as cp
import itertools
from model_wrappers import SklearnModel, XgboostModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def write_cv_res_csv(cls, cv_out, cv_csv_out):
    param_keys, param_vals, scores = cp.load(open(cv_out, 'rb'))
    assert len(param_vals) == len(scores), 'Error: param value list length do not match score list length!'
    assert len(param_keys) == len(param_vals[0]), 'Error: param key count and value count do not match!'
    f = open(cv_csv_out, 'w')
    for key in param_keys:
        f.write('{0},'.format(key))
    for i in xrange(len(scores[0])):
        f.write('score_{0},'.format(i))
    f.write('score_mean,score_std\n')
    for i, params in enumerate(param_vals):
        for p in params:
            f.write('{0},'.format(p))
        for s in scores[i]:
            f.write('{0},'.format(s))
        f.write('{0},{1}\n'.format(scores[i].mean(), scores[i].std()))
    f.close()
    pass


class GridSearch:
    def __init__(self, wrapper_class, experiment, model_param_keys, model_param_vals):
        """
        Constructor of grid search.
        Support search on a set of model parameters, and record the cv result of each param configuration.

        :param wrapper_class: model wrapper type string like 'XgboostModel' or 'SklearnModel'
        :param experiment: experiment object of ExperimentL1 or ExperimentL2
        :param model_param_keys: list of model param keys. eg. ['paramA', 'paramB', 'paramC']
        :param model_param_vals: list of model param values (iterable). eg. [['valAa', 'valAb'], [0.1, 0.2], (1, 2, 3)]
        :return: None
        """

        self.wrapper_class = wrapper_class
        self.experiment = experiment
        self.model_param_keys = model_param_keys
        self.model_param_vals = model_param_vals

        if wrapper_class == SklearnModel:
            self.model_name = model_param_vals[0]
        else:
            self.model_name = 'xgb'
        pass

    def search_by_cv(self, cv_out, cv_pred_out=None, refit_pred_out=None):
        """
        Search by cross validation.
        :param cv_out: Output pickle file name of cross validation score results.
        :param cv_pred_out: prediction of cross validation each fold.
        :param refit_pred_out: refit on full train set and predict on test set.
        :return: None
        """
        # create dataframe of results
        scores_list = []
        preds_list = []
        param_vals_list = []
        best_param = None
        best_score = None
        for v in itertools.product(*(self.model_param_vals[::-1])):
            param_dic = {}
            for i in xrange(len(self.model_param_keys)):
                param_dic[self.model_param_keys[-(i+1)]] = v[i]
            print param_dic
            model = self.wrapper_class(param_dic)
            scores, preds = self.experiment.cross_validation(model)
            scores_list.append(scores)
            preds_list.append(preds)
            param_vals_list.append(v)
            if cv_pred_out:
                cp.dump(preds_list, open(cv_pred_out, 'wb'), protocol=2)
            cp.dump((self.model_param_keys, param_vals_list, scores_list), open(cv_out, 'wb'), protocol=2)
            if not best_param or best_score > scores.mean():
                best_param = param_dic
        if refit_pred_out:
            self.fit_full_set_and_predict(refit_pred_out)
        return best_param

    def fit_full_set_and_predict(self, refit_pred_out):
        preds_list = []
        for v in itertools.product(*self.model_param_vals[::-1]):
            param_dic = {}
            for i in xrange(len(self.model_param_keys)):
                param_dic[self.model_param_keys[-(i+1)]] = v[i]
            model = self.wrapper_class(param_dic)
            preds = self.experiment.fit_fullset_and_predict(model)
            preds_list.append(preds)
        cp.dump(preds_list, open(refit_pred_out, 'wb'), protocol=2)
        pass

    def to_string(self):
        return self.model_name+ '_cv_'


class BayesSearch:
    def __init__(self, wrapper_class, experiment, model_param_keys, model_param_space,
                 cv_out=None, cv_pred_out=None, refit_pred_out=None, dump_round = 10):
        """
        Constructor of bayes search.
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

        self.eval_round = 1
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
                preds = self.trials.trial_attachments(dic)['preds']
                preds_list.append(preds)
            cp.dump(preds_list, open(self.cv_pred_out, 'wb'), protocol=2)
        if self.cv_out:
            scores_list = list()
            for dic in self.trials.trials:
                scores = self.trials.trial_attachments(dic)['scores']
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
        return best

    def fit_full_set_and_predict(self, refit_pred_out):
        pass