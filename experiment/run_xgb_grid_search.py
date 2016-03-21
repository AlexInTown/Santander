# -*- coding:utf-8 -*-
import copy
import sys
import os
import cPickle as cp
from stacking.experiment_l1 import ExperimentL1
from utils.config_utils import Config
from utils.submit_utils import save_submissions
from model_wrappers import *
from xgboost import XGBClassifier
from grid_search import GridSearch


def xgb_grid_search():
    exp = ExperimentL1()
    param_keys = ['model_type', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
                  'learning_rate', 'silent', 'objective', 'nthread', 'n_estimators', 'seed']
    param_vals = [[XGBClassifier], [4, 5, 6, 7, 8], [3, 4, 5, 6], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.5, 0.6, 0.7, 0.8, 0.85, 0.9] ,
                  [0.01, 0.02, 0.03, 0.04], [1], ['binary:logistic'], [4], [350, 450], [9438]]
    gs = GridSearch(SklearnModel, exp, param_keys, param_vals)
    gs.search_by_cv('xgb-grid-scores2.pkl', cv_pred_out='xgb-grid-preds2.pkl', refit_pred_out='xgb-refit-preds2.pkl')
    pass

def print_cv_res():
    GridSearch.write_cv_res_csv('xgb-grid-scores2.pkl', 'xgb-grid-scores2.csv')

if __name__=='__main__':
    xgb_grid_search()
    print_cv_res()