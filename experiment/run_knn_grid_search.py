# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'

from sklearn.neighbors import KNeighborsClassifier
from experiment.stacking.experiment_l1 import ExperimentL1
from grid_search import GridSearch
from model_wrappers import SklearnModel

def knn_grid_search():
    exp = ExperimentL1()
    param_keys = ['model_type', 'n_neighbors', 'weights',
                  'algorithm', 'leaf_size', 'metric', 'p', 'n_jobs']
    param_vals = [[KNeighborsClassifier], [1, 2, 4, 8, 16, 24, 32, 64], ['uniform', 'distance'],
                  ['ball_tree', 'kd_tree'], [20, 30, 40], ['minkowski'], [2], [4]]
    gs = GridSearch(SklearnModel, exp, param_keys, param_vals)
    gs.search_by_cv('sk-knn-grid-scores.pkl', cv_pred_out='sk-knn-grid-preds.pkl', refit_pred_out='sk-knn-refit-preds.pkl')
    pass

if __name__=='__main__':
    knn_grid_search()