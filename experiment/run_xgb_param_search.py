# -*- coding: utf-8 -*-
import copy
import sys
import os
import cPickle as cp
from stacking.experiment_l1 import ExperimentL1
from utils.config_utils import Config
from model_wrappers import *



def xgb_param_selection():
    exp_l1 = ExperimentL1()
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    other = {'silent':0, 'objective':'binary:logistic', 'nthread': 4, 'eval_metric': 'logloss', 'seed':0}
    model_param = copy.deepcopy(param)
    model_param.update(other)
    xgb_model = XgboostModel(model_param)
    scores, preds = exp_l1.cross_validation(xgb_model)
    cp.dump((scores,preds), os.path.join(Config.get_string('data.path'), 'output', 'adhoc-xgb.pkl' ))
    pass

if __name__ == '__main__':
    xgb_param_selection()