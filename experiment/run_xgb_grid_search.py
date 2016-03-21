# -*- coding:utf-8 -*-
import copy
import sys
import os
import cPickle as cp
from stacking.experiment_l1 import ExperimentL1
from utils.config_utils import Config
from utils.submit_utils import save_submissions
from model_wrappers import *

from grid_search import GridSearch


def xgb_grid_search():
    exp = ExperimentL1()
    param_keys = ['bst:max_depth', 'bst:min_child_weight', 'bst:subsample', 'bst:colsample_bytree',
                  'bst:eta', 'silent', 'objective', 'nthread', 'eval_metric', 'seed']
    param_vals = [[8, 9, 10, 11, 12], [3, 4, 5], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8, 0.9] ,
                  [0.05], [1], ['binary:logistic'], [4], [ 'logloss'], [9438]]
    gs = GridSearch(XgboostModel, exp, param_keys, param_vals)
    gs.search_by_cv('xgb-grid-scores.pkl', cv_pred_out='xgb-grid-preds.pkl', refit_pred_out='xgb-refit-preds.pkl')
    pass

if __name__=='__main__':
    xgb_grid_search()