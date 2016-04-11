# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
import os
from sklearn.ensemble import RandomForestClassifier
from experiment.stacking.experiment_l1 import ExperimentL1
import param_search
from model_wrappers import SklearnModel
from hyperopt import hp


def rf_grid_search():
    #exp = ExperimentL1()
    exp = ExperimentL1(train_fname='scaled_extend_train.csv', test_fname='scaled_extend_test.csv')
    param_keys = ['model_type', 'n_estimators', 'criterion',
                  'n_jobs']
    param_vals = [[RandomForestClassifier], [500], ['gini', 'entropy'],  [6]]
    gs = param_search.GridSearch(SklearnModel, exp, param_keys, param_vals)
    gs.search_by_cv('sk-rf-grid-scores.pkl', cv_pred_out='sk-rf-grid-preds.pkl', refit_pred_out='sk-rf-refit-preds.pkl')
    pass


def rf_bayes_search(train_fname, test_fname, out_fname_prefix='sk-rf-bayes'):
    exp = ExperimentL1(train_fname=train_fname, test_fname=test_fname)

    param_keys = ['model_type', 'n_estimators', 'max_features', 'min_samples_split', 'criterion', 'n_jobs',
                  'random_state']

    param_space = {'model_type': RandomForestClassifier, 'n_estimators': hp.quniform('n_estimators', 100, 1200, 50),
                   'max_features': hp.uniform('max_feats', 0.1, 0.95),
                   'min_samples_split': hp.quniform('min_split', 1, 10, 1),
                   'criterion': hp.choice('crit', ['gini', 'entropy']),
                   'n_jobs': 2, 'random_state': hp.choice('seed', [1234,53454,6676,12893])}

    bs = param_search.BayesSearch(SklearnModel, exp, param_keys, param_space,
                                  cv_out=out_fname_prefix+'-scores.pkl',
                                  cv_pred_out=out_fname_prefix+'-preds.pkl',
                                  refit_pred_out=out_fname_prefix+'refit-preds.pkl',
                                  dump_round=10)
    best = bs.search_by_cv()
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))
    return best

if __name__ == '__main__':
    rf_bayes_search('scaled_extend_train.csv', 'scaled_extend_test.csv', 'sk-rf-bayes-scaled-extend')
    rf_bayes_search('pca100_train.csv', 'pca100_test.csv', 'sk-rf-bayes-pca100')
    rf_bayes_search('pca200_train.csv', 'pca200_test.csv', 'sk-rf-bayes-pca200')
    rf_bayes_search('pca10_and_standard_train.csv', 'pca10_and_standard_test.csv', 'sk-rf-bayes-pca10-standard')
    rf_bayes_search('pca20_and_standard_train.csv', 'pca20_and_standard_test.csv', 'sk-rf-bayes-pca20-standard')
    rf_bayes_search('standard_extend_train.csv', 'standard_extend_test.csv', 'sk-rf-bayes-standard-extend')
    rf_bayes_search('raw_train.csv', 'raw_test.csv', 'sk-rf-bayes-raw')






