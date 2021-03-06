# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
import os
from sklearn.linear_model import LogisticRegression
from experiment.stacking.experiment_l1 import ExperimentL1
import param_search
from model_wrappers import SklearnModel
from hyperopt import hp


def lr_bayes_search(train_fname, test_fname, out_fname_prefix='sk-svc-bayes'):
    exp = ExperimentL1(train_fname=train_fname, test_fname=test_fname)

    param_keys = ['model_type', 'C',
                  #'loss',
                  'penalty', 'tol', 'solver', 'class_weight',
                  'random_state']

    param_space = {'model_type': LogisticRegression, 'C': hp.uniform('c', 0.1, 3),
                   #'loss': hp.choice('loss', ['hinge', 'squared_hinge']),
                   #'penalty': hp.choice('pen', ['l1', 'l2']),
                   'penalty': 'l2',
                   'tol': hp.uniform('tol', 1e-6, 3e-4),
                   'solver': hp.choice('solver', ['liblinear', 'lbfgs','newton-cg']),
                   'class_weight': hp.choice('cls_w', [None, 'auto']),
                   'random_state': hp.choice('seed', [1234, 53454, 6676, 12893]),
                   #'n_jobs': 2
                   }

    bs = param_search.BayesSearch(SklearnModel, exp, param_keys, param_space,
                                  cv_out=out_fname_prefix+'-scores.pkl',
                                  cv_pred_out=out_fname_prefix+'-preds.pkl',
                                  refit_pred_out=out_fname_prefix+'-refit-preds.pkl',
                                  dump_round=1)
    best = bs.search_by_cv(max_evals=60)
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))
    return best

if __name__ == '__main__':
    #lr_bayes_search('scaled_extend_train.csv', 'scaled_extend_test.csv', 'sk-lr-bayes-scaled-extend')
    #lr_bayes_search('standard_extend_train.csv', 'standard_extend_test.csv', 'sk-lr-bayes-standard-extend')
    #lr_bayes_search('raw_extend_train.csv', 'raw_extend_test.csv', 'sk-lr-bayes-raw-extend')
    #lr_bayes_search('pca100_train.csv', 'pca100_test.csv', 'sk-lr-bayes-pca100')
    #lr_bayes_search('pca200_train.csv', 'pca200_test.csv', 'sk-lr-bayes-pca200')
    #lr_bayes_search('pca10_and_standard_train.csv', 'pca10_and_standard_test.csv', 'sk-lr-bayes-pca10-standard')
    lr_bayes_search('pca20_and_standard_train.csv', 'pca20_and_standard_test.csv', 'sk-lr-bayes-pca20-standard')


