# -*- coding:utf-8 -*-
import os
from stacking.experiment_l1 import ExperimentL1
from utils.config_utils import Config
from utils.submit_utils import save_submissions, get_top_model_avg_preds
from model_wrappers import *
from xgboost.sklearn import XGBClassifier
import param_search
import cPickle as cp
from hyperopt import hp


def xgb_bayes_search(train_fname, test_fname, out_fname_prefix='xgb-bayes'):
    exp = ExperimentL1(train_fname=train_fname, test_fname=test_fname)
    param_keys = ['model_type', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
                  'learning_rate', 'silent', 'objective', 'nthread', 'n_estimators', 'seed']
    param_space = {'model_type': XGBClassifier, 'max_depth': hp.quniform('max_depth', 4, 9, 1),
                   'min_child_weight': hp.quniform('min_child_weight', 3, 7, 1),
                   'subsample': hp.uniform('subsample', 0.5, 1.0),
                   'colsample_bytree': hp.uniform('colsample', 0.5, 1.0),
                   'learning_rate': hp.uniform('eta', 0.01, 0.02),
                   'silent': 1, 'objective': 'binary:logistic',
                   'nthread': 8, 'n_estimators': 580, 'seed': hp.choice('seed', [1234,53454,6676,12893])}
    bs = param_search.BayesSearch(SklearnModel, exp, param_keys, param_space,
                                  cv_out=out_fname_prefix+'-scores.pkl',
                                  cv_pred_out=out_fname_prefix+'-preds.pkl',
                                  refit_pred_out=out_fname_prefix+'-refit-preds.pkl',
                                  dump_round=10, n_folds=10)
    best = bs.search_by_cv()
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))
    return best


def xgb_grid_search(exp):
    param_keys = ['model_type', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
                  'learning_rate', 'silent', 'objective', 'nthread', 'n_estimators', 'seed']
    param_vals = [[XGBClassifier], [4, 5, 6, 7, 8], [3, 4, 5, 6], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.5, 0.6, 0.7, 0.8, 0.85, 0.9] ,
                  [0.01, 0.02, 0.03, 0.04], [1], ['binary:logistic'], [4], [350, 450], [9438]]
    gs = param_search.GridSearch(SklearnModel, exp, param_keys, param_vals)
    best = gs.search_by_cv('xgb-grid-scores2.pkl', cv_pred_out='xgb-grid-preds2.pkl', refit_pred_out='xgb-refit-preds2.pkl')
    param_search.write_cv_res_csv('xgb-grid-scores2.pkl', 'xgb-grid-scores2.csv')
    return best


def xgb_submmision(exp, param=None):
    if not param:
        #param = {'colsample_bytree': 0.6475941408157723, 'silent': 1, 'model_type': XGBClassifier, 'learning_rate': 0.018480417410705455, 'nthread': 4, 'min_child_weight': 5.0, 'n_estimators': 400, 'subsample': 0.5998597024787456, 'seed': 9438, 'objective': 'binary:logistic', 'max_depth': 6.0}
        param = {'colsample_bytree': 0.701, 'silent': 1, 'model_type': XGBClassifier, 'learning_rate': 0.202, 'nthread': 4, 'n_estimators': 572, 'subsample': 0.6815, 'seed': 1234, 'objective': 'binary:logistic', 'max_depth': 5}

    xgb_model = SklearnModel(param)
    final_preds = exp.fit_fullset_and_predict(xgb_model)
    submission_path = os.path.join(Config.get_string('data.path'), 'submission')
    # fname = os.path.join(submission_path, xgb_model.to_string() + '_res.csv')
    fname = os.path.join(submission_path, 'xgb_bayes_param_res.csv')
    print final_preds
    print exp.test_id
    save_submissions(fname, exp.test_id, final_preds)


def xgb_param_avg_submission(exp):
    score_fname = os.path.join(Config.get_string('data.path'), 'output', 'xgb-bayes-scores.pkl')
    refit_pred_fname =os.path.join(Config.get_string('data.path'), 'output', 'xgb-bayes-refit-preds.pkl')
    preds = get_top_model_avg_preds(score_fname, refit_pred_fname, topK=30)
    submission_fname = os.path.join(Config.get_string('data.path'), 'submission', 'avg-xgb-bayes-refit-preds30.csv')
    save_submissions(submission_fname, exp.test_id, preds)


if __name__=='__main__':
    #exp = ExperimentL1(train_fname='scaled_extend_train.csv', test_fname='scaled_extend_test.csv')
    #param_search.write_cv_res_csv('xgb-bayes-scores.pkl', 'xgb-bayes-scores.csv')
    #param = None
    #param = xgb_grid_search(exp)
    #param = xgb_bayes_search(exp)
    #xgb_submmision(exp, param)
    #xgb_param_avg_submission(exp)
    #xgb_bayes_search('pca10_and_standard_train.csv', 'pca10_and_standard_test.csv', 'xgb-bayes-pca10-and-standard')
    xgb_bayes_search('filtered_train.csv', 'filtered_test.csv', 'xgb-bayes-cv10')
    pass


