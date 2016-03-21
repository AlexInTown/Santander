# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from experiment.stacking.experiment_l1 import ExperimentL1
from grid_search import GridSearch

def xgb_grid_search():
    exp = ExperimentL1()
    param_keys = ['model_type', 'n_estimators', 'criterion', 'max_depth',
                  'min_sample_split', 'silent', 'objective', 'nthread', 'eval_metric', 'seed']
    param_vals = [['sklearn.ensemble.RandomForestClassifier'], [3, 4, 5], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8, 0.9] ,
                  [0.05], [1], ['binary:logistic'], [4], [ 'logloss'], [9438]]
    gs = GridSearch('SklearnModel', exp, param_keys, param_vals)
    gs.search_by_cv('xgb-grid-scores.pkl', cv_pred_out='xgb-grid-preds.pkl', refit_pred_out='xgb-refit-preds.pkl')
    pass

if __name__=='__main__':
    xgb_grid_search()


