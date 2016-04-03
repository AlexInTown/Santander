# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from experiment.stacking.experiment_l1 import ExperimentL1
from param_search import GridSearch, BayesSearch
from model_wrappers import SklearnModel
from utils.config_utils import Config
from hyperopt import hp


def knn_bayes_search():
    exp = ExperimentL1(train_fname='scaled_train.csv', test_fname='scaled_test.csv')
    param_keys = ['model_type', 'n_neighbors', 'weights',
                  'algorithm', 'leaf_size', 'metric', 'p']#, 'n_jobs']
    param_space = {'model_type': KNeighborsClassifier,
                   'n_neighbors': hp.quniform('neighbors', 1, 9, 1),
                   'weights': hp.choice('weights',  ['uniform', 'distance']),
                   'algorithm': hp.choice('algorithm',  ['ball_tree', 'kd_tree']),
                   'algorithm':'ball_tree',
                   'leaf_size': 30, 'metric': 'minkowski',
                   'p': hp.quniform('n_neighbors', 1, 2, 1),
                   #'n_jobs': 4
                   }

    bs = BayesSearch(SklearnModel, exp, model_param_keys=param_keys, model_param_space=param_space,
                     cv_out='sk-knn-bayes-scores.pkl', cv_pred_out='sk-knn-bayes-preds.pkl',
                     refit_pred_out='sk-knn-bayes-refit-preds.pkl')
    bs.search_by_cv()


def knn_grid_search():
    exp = ExperimentL1(train_fname='scaled_train.csv', test_fname='scaled_test.csv')
    param_keys = ['model_type', 'n_neighbors', 'weights',
                  'algorithm', 'leaf_size', 'metric', 'p', 'n_jobs']
    param_vals = [[KNeighborsClassifier], [1, 2, 4, 8, 16, 24, 32, 64], ['uniform', 'distance'],
                  ['ball_tree'], [30], ['minkowski'], [2], [4]]
    gs = GridSearch(SklearnModel, exp, param_keys, param_vals)
    gs.search_by_cv('sk-knn-grid-scores.pkl', cv_pred_out='sk-knn-grid-preds.pkl', refit_pred_out='sk-knn-refit-preds.pkl')
    pass

if __name__=='__main__':
    #knn_grid_search()
    knn_bayes_search()