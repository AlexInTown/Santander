# -*- coding:utf-8 -*-
__author__ = 'AlexInTown'
from experiment.stacking.experiment_l2 import ExperimentL2, get_top_cv_and_test_preds
from experiment.stacking.experiment_l1 import ExperimentL1
from model_wrappers import SklearnModel
from utils.config_utils import Config
from utils.submit_utils import get_top_model_avg_preds, save_submissions
from sklearn.linear_model import LogisticRegression
import param_search
import cPickle as cp
from hyperopt import hp


def get_l2_experiment(use_raw_feats=0):
    exp_l1 = ExperimentL1()

    l1_model_results = [
        # neural network results
        #{'prefix': 'nn-standard-bayes', 'top_k': 5, 'is_avg': 0},
        {'prefix': 'nn-adam-small-standard-bayes', 'ids': [64, 23, 26]},

        # logistic regression results
        {'prefix': 'sk-lr-bayes-pca20-standard', 'top_k': 5, 'is_avg': 1},
        # {'prefix': 'sk-lr-bayes-pca10-standard', 'top_k': 5, 'is_avg': 1},
        # {'prefix': 'sk-lr-bayes-pca200', 'top_k': 5, 'is_avg': 0},
        # {'prefix': 'sk-lr-bayes-pca100', 'top_k': 5, 'is_avg': 0},
        # {'prefix': 'sk-lr-bayes-raw-extend', 'top_k': 5, 'is_avg': 0},
        #{'prefix': 'sk-lr-bayes-standard-extend', 'top_k': 5, 'is_avg': 1},
        #{'prefix': 'sk-lr-bayes-scaled-extend', 'top_k': 5, 'is_avg': 1},

        # xgboost results
        #{'prefix': 'xgb-bayes-pca10-and-standard', 'top_k': 15, 'is_avg': 0},
        {'prefix': 'xgb-bayes', 'top_k': 30, 'is_avg': 0},

        # knn results
        # {'prefix': 'sk-knn-bayes-pca100', 'top_k': 2, 'is_avg': 0},
        # {'prefix': 'sk-knn-bayes-pca200', 'top_k': 2, 'is_avg': 0},
        {'prefix': 'sk-knn-bayes-pca100', 'ids': [65, 14, 52, 46]},
        {'prefix': 'sk-knn2-bayes-pca100', 'ids': [53, 57, 24, 28, 46]},
        # {'prefix': 'sk-knn-bayes-pca10-standard', 'top_k': 5, 'is_avg': 1},
        # {'prefix': 'sk-knn-bayes-pca20-standard', 'top_k': 5, 'is_avg': 1},

        # random forest results
        {'prefix': 'sk-rf-bayes-pca20-standard', 'top_k': 5, 'is_avg': 0},
        #{'prefix': 'sk-rf-bayes-pca10-standard', 'top_k': 5, 'is_avg': 1},
        # {'prefix': 'sk-rf-bayes-pca200', 'top_k': 5, 'is_avg': 0},
        # {'prefix': 'sk-rf-bayes-pca100', 'top_k': 5, 'is_avg': 0},
        # {'prefix': 'sk-rf-bayes-raw-extend', 'top_k': 5, 'is_avg': 0},
        # {'prefix': 'sk-rf-bayes-standard-extend', 'top_k': 5, 'is_avg': 0},
        # {'prefix': 'sk-rf-bayes-scaled-extend', 'top_k': 5, 'is_avg': 0},

    ]
    exp_l2 = ExperimentL2(exp_l1, l1_model_results, use_raw_feats=use_raw_feats)
    return exp_l2


def xgb_model_stacking(exp_l2, out_fname_prefix):
    from xgboost.sklearn import XGBClassifier
    param_keys = ['model_type', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
                  'learning_rate', 'silent', 'objective', 'nthread', 'n_estimators', 'seed']
    param_space = {'model_type': XGBClassifier, 'max_depth': hp.quniform('max_depth', 2, 9, 1),
                   'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1),
                   'subsample': hp.uniform('subsample', 0.1, 1.0),
                   'colsample_bytree': hp.uniform('colsample', 0.3, 1.0),
                   'learning_rate': hp.uniform('eta', 0.01, 0.02),
                   'silent': 1, 'objective': 'binary:logistic',
                   'nthread': 3, 'n_estimators': hp.quniform('n', 100, 1000, 50),
                   'seed': hp.choice('seed', [1234,53454,6676,12893])}
    # param_space = {'model_type': XGBClassifier, 'max_depth': hp.quniform('max_depth', 3, 9, 1),
    #                'min_child_weight': hp.quniform('min_child_weight', 3, 7, 1),
    #                'subsample': hp.uniform('subsample', 0.1, 1.0),
    #                'colsample_bytree': hp.uniform('colsample', 0.1, 0.6),
    #                'learning_rate': hp.uniform('eta', 0.01, 0.02),
    #                'silent': 1, 'objective': 'binary:logistic',
    #                'nthread': 4, 'n_estimators': 600, 'seed': hp.choice('seed', [1234,53454,6676,12893])}
    # l2 model output
    bs = param_search.BayesSearch(SklearnModel, exp_l2, param_keys, param_space,
                                  cv_out=out_fname_prefix+'-scores.pkl',
                                  cv_pred_out=out_fname_prefix+'-preds.pkl',
                                  refit_pred_out=out_fname_prefix+'refit-preds.pkl',
                                  dump_round=10)
    best = bs.search_by_cv()
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))
    return best


def lr_model_stacking(exp_l2, out_fname_prefix):
    param_keys = ['model_type', 'C',
                  'penalty', 'tol', 'solver', 'class_weight',
                  'random_state']

    param_space = {'model_type': LogisticRegression, 'C': hp.uniform('c', 0.1, 3),
                   'penalty': 'l2',
                   'tol': hp.uniform('tol', 1e-6, 3e-4),
                   'solver': hp.choice('solver', ['liblinear', 'lbfgs','newton-cg']),
                   'class_weight': hp.choice('cls_w', [None, 'balanced']),
                   'random_state': hp.choice('seed', [1234, 53454, 6676, 12893]),
                   }
    bs = param_search.BayesSearch(SklearnModel, exp_l2, param_keys, param_space,
                                  cv_out=out_fname_prefix+'-scores.pkl',
                                  cv_pred_out=out_fname_prefix+'-preds.pkl',
                                  refit_pred_out=out_fname_prefix+'refit-preds.pkl',
                                  dump_round=10)
    best = bs.search_by_cv(max_evals=101)
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))
    return best


def nn_model_stacking(exp_l2, out_fname_prefix):
    from model_wrappers import LasagneModel
    from lasagne.nonlinearities import sigmoid, tanh, rectify, leaky_rectify
    param_keys = ['in_size', 'hid_size', 'batch_size', 'in_dropout',
                  'hid_dropout', 'nonlinearity', 'learning_rate', 'l1_reg', 'l2_reg', 'num_epochs']
    param_space = {'in_size': exp_l2.train_x.shape[1],
                   'hid_size': hp.quniform('hid', 10, 200, 25),
                   'batch_size': hp.quniform('bsize', 50, 300, 50),
                   'in_dropout': hp.uniform('in_drop',  0.0, 0.3),
                   'hid_dropout': hp.uniform('hid_drop',  0.0, 0.4),
                   'nonlinearity': hp.choice('nonlinear',  [sigmoid, tanh, rectify, leaky_rectify]),
                   'learning_rate': hp.uniform('lr', 0.00001, 0.01),
                   'l1_reg': hp.uniform('l1_reg', 0.00001, 0.0005),
                   'l2_reg': hp.uniform('l2_reg', 0.01, 0.2),
                   'num_epochs': hp.quniform('epochs', 100, 1000, 100),
                   }
    bs = param_search.BayesSearch(LasagneModel, exp_l2, model_param_keys=param_keys, model_param_space=param_space,
                     cv_out=out_fname_prefix+'-scores.pkl',
                     cv_pred_out=out_fname_prefix+'-preds.pkl',
                     refit_pred_out=out_fname_prefix+'refit-preds.pkl',
                     dump_round=1)
    best = bs.search_by_cv(max_evals=201)
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))
    return best


def save_l2_submission(prefix='stacking-xgb'):
    import os
    exp = ExperimentL1()
    score_fname = os.path.join(Config.get_string('data.path'), 'output', prefix+'-scores.pkl')
    refit_pred_fname =os.path.join(Config.get_string('data.path'), 'output', prefix+'-refit-preds.pkl')
    topK = 1
    preds = get_top_model_avg_preds(score_fname, refit_pred_fname, topK=topK)
    submission_fname = os.path.join(Config.get_string('data.path'), 'submission',
                                    prefix+'-refit-preds{}.csv'.format(topK))
    save_submissions(submission_fname, exp.test_id, preds)


def main():
    exp_l2 = get_l2_experiment(use_raw_feats=0)
    from experiment.feat_tuning.xgb_feat_analysis import fit_and_print_important_feats
    fit_and_print_important_feats(exp_l2.train_x, exp_l2.train_y, 300)
    #xgb_model_stacking(exp_l2, 'stacking-xgb-knn-neural-ids')
    # lr_model_stacking(exp_l2, 'stacking-sk-lr')
    # nn_model_stacking(exp_l2, 'stacking-nn')

if __name__ == '__main__':
    main()
    #save_l2_submission('stacking-xgb-knn-neural-ids')
    #save_l2_submission('stacking-nn')
    pass