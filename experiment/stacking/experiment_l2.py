# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import cPickle as cp
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics


def get_top_cv_and_test_preds(out_fname_prefix, top_k=10, use_lower=0):
    """
    Get the top k cross-validation predictions of trainset and refit predictions of testset from experiment_l1 results.
    You can use numpy.hstack to join different model results
    :param out_fname_prefix: prefix to identify a given experiment (L1)
    :param k: top k
    :return: top k cv preds and refit preds (numpy array)
    """
    from utils.config_utils import Config
    # file names
    score_fname = os.path.join(Config.get_string('data.path'), 'output', out_fname_prefix+'-scores.pkl')
    pred_fname = os.path.join(Config.get_string('data.path'), 'output', out_fname_prefix +'-preds.pkl')
    refit_pred_fname = os.path.join(Config.get_string('data.path'), 'output', out_fname_prefix +'-refit-preds.pkl')
    # load pickle files
    param_keys, param_vals, scores = cp.load(open(score_fname, 'rb'))
    refit_preds = cp.load(open(refit_pred_fname, 'rb'))
    preds = cp.load(open(pred_fname, 'rb'))
    # calculate top results
    scores = np.asarray(scores)
    idxs = np.arange(len(scores))
    mscores = scores.mean(axis=1)
    if use_lower:
        mscores -= scores.std()
    idxs = sorted(idxs, key=lambda x:mscores[x], reverse=1)[:top_k]
    preds = np.transpose(np.asarray(preds)[idxs])
    refit_preds = np.transpose(np.asarray(refit_preds)[idxs])
    return preds, refit_preds

def get_cv_and_test_preds(out_fname_prefix, idxs):
    from utils.config_utils import Config
    pred_fname = os.path.join(Config.get_string('data.path'), 'output', out_fname_prefix +'-preds.pkl')
    refit_pred_fname = os.path.join(Config.get_string('data.path'), 'output', out_fname_prefix +'-refit-preds.pkl')
    refit_preds = cp.load(open(refit_pred_fname, 'rb'))
    preds = cp.load(open(pred_fname, 'rb'))
    preds = np.transpose(np.asarray(preds)[idxs])
    refit_preds = np.transpose(np.asarray(refit_preds)[idxs])
    return preds, refit_preds

class ExperimentL2:
    def __init__(self, exp_l1, exp_l1_output, use_raw_feats=0):
        # load result pkl, use the model to get output attributes
        self.train_id = exp_l1.train_id
        self.test_id = exp_l1.test_id
        self.train_y = exp_l1.train_y

        self.train_x = None
        self.test_x = None

        train_dict = {}
        test_dict = {}
        if use_raw_feats:
            for col in exp_l1.train_x.columns:
                train_dict[col] = exp_l1.train_x[col].values
            for col in exp_l1.test_x.columns:
                test_dict[col] = exp_l1.test_x[col].values
        # parse the meta feature
        for data in exp_l1_output:
            prefix = data['prefix']
            is_avg = 0
            try:
                if 'ids' in data:
                    print '- Loading {} results of {} '.format(str(data['ids']), prefix)
                    preds, refit_preds = get_cv_and_test_preds(prefix, data['ids'])
                else:
                    top_k = data['top_k']
                    is_avg = data['is_avg']
                    print '- Loading {} results of {} (is_avg={})'.format(top_k, prefix,is_avg),
                    preds, refit_preds = get_top_cv_and_test_preds(prefix, top_k=top_k)
                print 'SUCCESS'
            except Exception, e:
                print 'FAIL', e
                continue

            if is_avg:
                preds = preds.mean(axis=1)
                train_dict[prefix+'-avg'] = preds
                refit_preds = refit_preds.mean(axis=1)
                test_dict[prefix+'-avg'] = refit_preds
            else:
                for i in xrange(preds.shape[1]):
                    train_dict[prefix+'-'+str(i)] = preds[:, i]
                    test_dict[prefix+'-'+str(i)] = refit_preds[:, i]

        import pandas as pd
        self.train_x = pd.DataFrame(data=train_dict)
        self.test_x = pd.DataFrame(data=test_dict)
        self.random_state = exp_l1.random_state
        #self.random_state = 8862
        pass

    def cross_validation(self, model, n_folds=5):
        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=n_folds, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = np.zeros(len(self.train_y))
        i = 0
        for train_idx, test_idx in kfold:
            print ' - fold {0}  '.format(i),
            start = time.time()
            train_x = self.train_x.iloc[train_idx]
            train_y = self.train_y[train_idx]
            test_x = self.train_x.iloc[test_idx]
            test_y = self.train_y[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred)
            preds[test_idx] = pred
            scores.append(score)
            end = time.time()
            print (' score:{}  time:{}s.'.format(score, end - start))
            i += 1
        scores = np.asarray(scores)
        print scores.mean(), scores.std(), metrics.roc_auc_score(self.train_y, preds)
        return scores, preds

    def fit_fullset_and_predict(self, model):
        model.fit(self.train_x, self.train_y)
        preds = model.predict(self.test_x)
        return preds

    def get_avg_result(self):
        return self.test_x.mean(axis=1)
