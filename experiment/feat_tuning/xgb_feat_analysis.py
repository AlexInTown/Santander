# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
from experiment.stacking.experiment_l1 import ExperimentL1
from xgboost.sklearn import XGBClassifier
from experiment.model_wrappers import SklearnModel


def fit_and_print_important_feats(train_x, train_y, top_feats):
    param_space = {'model_type': XGBClassifier,
                   'max_depth': 6,
                   'min_child_weight': 6,
                   'subsample': 0.582198299,
                   'colsample_bytree': 0.677688653,
                   'learning_rate': 0.017632947,
                   'silent': 1, 'objective': 'binary:logistic',
                   'nthread': 8, 'n_estimators': 400, 'seed': 9438}

    model = SklearnModel(param_space)
    model.fit(train_x, train_y)
    fmap = model.model.booster().get_fscore()
    total = 0.0
    for k,v in fmap.iteritems():
        total += v
    feat_importances = sorted(fmap.items(), key = lambda x:x[1], reverse=1)[:top_feats]
    for k,v in feat_importances:
        print k, v/total
    return feat_importances


def l1_model_important_feats(train_fname, test_fname, top_feats=10):
    exp = ExperimentL1(train_fname=train_fname, test_fname=test_fname)
    feat_importances = fit_and_print_important_feats(exp.train_x, exp.train_y, top_feats=top_feats)


def main():
    l1_model_important_feats('filtered_train.csv', 'filtered_test.csv', top_feats=300)
    pass

if __name__ == '__main__':
    main()
    pass