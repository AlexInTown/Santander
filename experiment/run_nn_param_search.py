# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
import os
from experiment.stacking.experiment_l1 import ExperimentL1
import param_search
from model_wrappers import LasagneModel
from utils.config_utils import Config
from utils.submit_utils import get_top_model_avg_preds, save_submissions
from hyperopt import hp
from lasagne.nonlinearities import sigmoid, tanh, rectify, leaky_rectify
from lasagne.updates import nesterov_momentum, adam, adadelta


def nn_bayes_search(train_fname, test_fname, out_fname_prefix='nn-bayes'):
    exp = ExperimentL1(train_fname=train_fname, test_fname=test_fname)
    param_keys = ['in_size', 'hid_size', 'batch_size', 'in_dropout',
                  'hid_dropout', 'nonlinearity',
                  'updates',
                  'learning_rate',
                  #'l1_reg',
                  #'l2_reg',
                  'num_epochs']
    param_space = {'in_size': exp.train_x.shape[1],
                   'hid_size': hp.quniform('hid', 10, 300, 5),
                   'batch_size': hp.quniform('bsize', 200, 5000, 50),
                   'in_dropout': hp.uniform('in_drop',  0.0, 0.5),
                   'hid_dropout': hp.uniform('hid_drop',  0.0, 0.6),
                   'updates': hp.choice('updates', [nesterov_momentum, adam]),
                   'nonlinearity': hp.choice('nonlinear',  [sigmoid, tanh, rectify]),
                   'learning_rate': hp.uniform('lr', 0.0001, 0.1),

                   #'learning_rate': 0.01,
                   #'l1_reg': hp.uniform('l1_reg', 0.0, 0.000001),
                   #'l2_reg': hp.uniform('l2_reg', 0.0, 0.000001),
                   'num_epochs': hp.quniform('epochs', 200, 1000, 50),
                   }

    bs = param_search.BayesSearch(LasagneModel, exp, model_param_keys=param_keys, model_param_space=param_space,
                     cv_out=out_fname_prefix+'-scores.pkl',
                     cv_pred_out=out_fname_prefix+'-preds.pkl',
                     refit_pred_out=out_fname_prefix+'-refit-preds.pkl',
                     dump_round=1, use_lower=0, n_folds=5)
    bs.search_by_cv(max_evals=301)
    param_search.write_cv_res_csv(bs.cv_out, bs.cv_out.replace('.pkl', '.csv'))

def nn_param_avg_submission(prefix, top_k=1):
    exp = ExperimentL1()
    score_fname = os.path.join(Config.get_string('data.path'), 'output',prefix+ '-scores.pkl')
    refit_pred_fname =os.path.join(Config.get_string('data.path'), 'output',prefix+ '-refit-preds.pkl')
    preds = get_top_model_avg_preds(score_fname, refit_pred_fname, topK=top_k)
    submission_fname = os.path.join(Config.get_string('data.path'), 'submission', 'avg-{}-refit-preds{}.csv'.format(prefix, top_k))
    save_submissions(submission_fname, exp.test_id, preds)


def main():
    nn_bayes_search('standard_train.csv', 'standard_test.csv', 'nn-updates-high-lr-bayes-cv5')
    pass


if __name__=='__main__':
    main()
    #nn_param_avg_submission('nn-standard-bayes')
    pass