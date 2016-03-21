# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from experiment.stacking.experiment_l1 import ExperimentL1
from grid_search import GridSearch
from model_wrappers import SklearnModel

def rf_grid_search():
    exp = ExperimentL1()
    param_keys = ['model_type', 'n_estimators', 'criterion',
                  'n_jobs']
    param_vals = [[RandomForestClassifier], [300], ['gini', 'entropy'],  [6]]
    gs = GridSearch(SklearnModel, exp, param_keys, param_vals)
    gs.search_by_cv('sk-rf-grid-scores.pkl', cv_pred_out='sk-rf-grid-preds.pkl', refit_pred_out='sk-rf-refit-preds.pkl')
    pass

if __name__=='__main__':
    rf_grid_search()


