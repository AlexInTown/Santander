# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
from sklearn.neighbors import KNeighborsClassifier
from experiment.stacking.experiment_l1 import ExperimentL1
from param_search import GridSearch
from model_wrappers import SklearnModel
from utils.config_utils import Config
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
    knn_grid_search()