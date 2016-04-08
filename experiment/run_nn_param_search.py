# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
from experiment.stacking.experiment_l1 import ExperimentL1
import param_search
from model_wrappers import LasagneModel
from hyperopt import hp
from lasagne.nonlinearities import sigmoid, tanh, rectify, leaky_rectify


def nn_bayes_search(train_fname, test_fname, out_fname_prefix='nn-bayes'):
    exp = ExperimentL1(train_fname=train_fname, test_fname=test_fname)
    param_keys = ['in_size', 'hid_size', 'batch_size', 'in_dropout',
                  'hid_dropout', 'nonlinearity', 'learning_rate', 'num_epochs']
    param_space = {'in_size': exp.train_x.shape[1],
                   'hid_size': hp.quniform('hid', 50, 500, 25),
                   'batch_size': hp.quniform('bsize', 50, 1000, 50),
                   'in_dropout': hp.uniform('in_drop',  0.5, 1.0),
                   'hid_dropout': hp.uniform('hid_drop',  0.5, 1.0),
                   'nonlinearity': hp.choice('nonlinear',  [sigmoid, tanh, rectify, leaky_rectify]),
                   'learning_rate': hp.uniform('lr', 0.0001, 0.01),
                   'num_epochs': hp.quniform('epochs', )
                   }

    bs = param_search.BayesSearch(LasagneModel, exp, model_param_keys=param_keys, model_param_space=param_space,
                     cv_out=out_fname_prefix+'-scores.pkl',
                     cv_pred_out=out_fname_prefix+'-preds.pkl',
                     refit_pred_out=out_fname_prefix+'refit-preds.pkl',
                     dump_round=10)
    bs.search_by_cv(max_evals=201)
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))


def main():
    nn_bayes_search('standard_train.csv', 'standard_test.csv', 'nn-standard-bayes')
    pass