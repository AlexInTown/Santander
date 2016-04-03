# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
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


def get_top_model_avg_preds(score_fname, refit_pred_fname, topK=10):
    param_keys, param_vals, scores = cp.load(open(score_fname, 'rb'))
    refit_preds = cp.load(open(refit_pred_fname, 'rb'))
    scores = np.asarray(scores)
    idxs = np.arange(len(scores))
    scores = scores.mean(axis=1)
    idxs = sorted(idxs, key=lambda x:scores[x], reverse=1)[:topK]
    to_avg = refit_preds[idxs]
    return np.asarray(to_avg).mean(axis=1)

