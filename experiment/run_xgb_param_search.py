# -*- coding: utf-8 -*-
import copy
import sys
import os
import cPickle as cp
from stacking.experiment_l1 import ExperimentL1
from utils.config_utils import Config
from utils.submit_utils import save_submissions
from model_wrappers import *



def xgb_param_selection(exp_l1):
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    other = {'silent':0, 'objective':'binary:logistic', 'nthread': 4, 'eval_metric': 'logloss', 'seed':0}
    model_param = copy.deepcopy(param)
    model_param.update(other)
    xgb_model = XgboostModel(model_param)
    scores, preds = exp_l1.cross_validation(xgb_model)
    fname = os.path.join(Config.get_string('data.path'), 'output', 'adhoc-xgb.pkl' )
    cp.dump((scores,preds), open(fname, 'wb'), protocol=2)
    pass


def xgb_submmision(param):
    if not param:
        param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
        other = {'silent':0, 'objective':'binary:logistic', 'nthread': 4, 'eval_metric': 'logloss', 'seed':0}
        param.update(other)
    xgb_model = XgboostModel(param)
    final_preds = exp_l1.fit_fullset_and_predict(xgb_model)
    submission_path = os.path.join(Config.get_string('data.path'), 'submission')
    # fname = os.path.join(submission_path, xgb_model.to_string() + '_res.csv')
    fname = os.path.join(submission_path, 'xgb_adhoc_param_res.csv')
    print final_preds
    print exp_l1.test_id
    save_submissions(fname, exp_l1.test_id, final_preds)


if __name__ == '__main__':
    exp_l1 = ExperimentL1()
    # param = xgb_param_selection(exp_l1)
    xgb_submmision(None)

