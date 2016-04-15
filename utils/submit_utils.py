# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
import os
import sys
import cPickle as cp
import numpy as np

def save_submissions(fname, ids, preds):
    assert len(ids) == len(preds), "Error: the id and pred length not match!"
    f = open(fname, 'w')
    f.write("ID,TARGET\n")
    for i in xrange(len(ids)):
        f.write("{0},{1}\n".format(ids[i], preds[i]))
    f.close()
    pass


def get_top_model_avg_preds(score_fname, refit_pred_fname, topK=10, use_lower=0):
    param_keys, param_vals, scores = cp.load(open(score_fname, 'rb'))
    refit_preds = cp.load(open(refit_pred_fname, 'rb'))
    scores = np.asarray(scores)
    idxs = np.arange(len(scores))
    mscores = scores.mean(axis=1)
    if use_lower:
        mscores -= scores.std()
    idxs = sorted(idxs, key=lambda x:mscores[x], reverse=1)[:topK]
    for i in idxs:
        print param_vals[i]
        print scores[i], scores[i].mean(), scores[i].std()
    to_avg = np.asarray(refit_preds)[idxs]
    return to_avg.mean(axis=0)


def get_selected_model_avg_preds(score_fname, refit_pred_fname, idxs):
    param_keys, param_vals, scores = cp.load(open(score_fname, 'rb'))
    refit_preds = cp.load(open(refit_pred_fname, 'rb'))
    scores = np.asarray(scores)
    for i in idxs:
        print param_vals[i]
        print scores[i], scores[i].mean(), scores[i].std()
    to_avg = np.asarray(refit_preds)[idxs]
    return to_avg.mean(axis=0)


def main():
    if len(sys.argv) != 3:
        print 'Usage: python submit_utils.py <model-prefix> <model-idxs>'
        exit()
    from utils.config_utils import Config
    model_prefix = sys.argv[1]
    score_fname = os.path.join(Config.get_string('data.path'), 'output', model_prefix + '-scores.pkl')
    refit_pred_fname = os.path.join(Config.get_string('data.path'), 'output', model_prefix + '-refit-preds.pkl')
    model_idxs =  sys.argv[2].strip()
    idxs = [int(s) for s in model_idxs.split(',')]
    preds = get_selected_model_avg_preds(score_fname, refit_pred_fname, idxs)
    from experiment.stacking.experiment_l1 import ExperimentL1
    exp = ExperimentL1()
    submission_fname = os.path.join(Config.get_string('data.path'), 'submission',
                                    '{}-{}-submission.csv'.format(model_prefix, model_idxs))
    save_submissions(submission_fname, exp.test_id, preds)
    pass


if __name__ == '__main__':
    main()
    pass
