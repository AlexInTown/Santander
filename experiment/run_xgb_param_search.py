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
    param = {'bst:max_depth':8, 'bst:min_child_weight': 5, 'bst:subsample': 0.6, 'bst:colsample_bytree':0.7,  'bst:eta':0.03}
    other = {'silent':1, 'objective':'binary:logistic', 'nthread': 4, 'eval_metric': 'logloss', 'seed':0}
    model_param = copy.deepcopy(param)
    model_param.update(other)
    xgb_model = XgboostModel(model_param)
    scores, preds = exp_l1.cross_validation(xgb_model)
    fname = os.path.join(Config.get_string('data.path'), 'output', 'adhoc-xgb.pkl' )
    cp.dump((scores,preds), open(fname, 'wb'), protocol=2)
    pass


def xgb_submmision(param):
    if not param:
        # {'colsample_bytree': 0.6, 'silent': 1, 'model_type': <class 'xgboost.sklearn.XGBClassifier'>,
        # 'learning_rate': 0.01, 'nthread': 4, 'min_child_weight': 5, 'n_estimators': 350, 'subsample': 0.9,
        # 'seed': 9438, 'objective': 'binary:logistic', 'max_depth': 8}
        param = {'bst:max_depth':8, 'bst:min_child_weight': 5, 'bst:subsample': 0.7, 'bst:colsample_bytree':0.6,  'bst:eta':0.01}
        other = {'silent':0, 'objective':'binary:logistic', 'nthread': 4, 'eval_metric': 'logloss', 'seed':9438}
        param.update(other)
    xgb_model = XgboostModel(param, train_params= {"num_boost_round": 500 })
    final_preds = exp_l1.fit_fullset_and_predict(xgb_model)
    submission_path = os.path.join(Config.get_string('data.path'), 'submission')
    # fname = os.path.join(submission_path, xgb_model.to_string() + '_res.csv')
    fname = os.path.join(submission_path, 'xgb_adhoc_param_res.csv')
    print final_preds
    print exp_l1.test_id
    save_submissions(fname, exp_l1.test_id, final_preds)



if __name__ == '__main__':
    #exp_l1 = ExperimentL1()
    exp_l1 = ExperimentL1(train_fname='raw_extend_train.csv', test_fname='raw_extend_test.csv')
    xgb_param_selection(exp_l1)
    #xgb_submmision(None)

