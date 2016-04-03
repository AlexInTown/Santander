# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
import cPickle as cp
from utils.config_utils import Config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def fix_abnormal_value(df):

    return df

def scale(train, test):
    train_ids = train.ID.values
    targets = train.TARGET.values
    train1 = train.drop(['ID', 'TARGET'], axis = 1)
    test_ids = test.ID.values
    test1 = test.drop(['ID'], axis=1)
    mm = MinMaxScaler()
    train1 = mm.fit_transform(train1)
    test1 = mm.transform(test1)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic[train.columns[i+1]] = train1[:, i]
    train1 = pd.DataFrame(dic)
    dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)
    return train1, test1

def standard(train, test):
    train_ids = train.ID.values
    targets = train.TARGET.values
    train1 = train.drop(['ID', 'TARGET'], axis = 1)
    test_ids = test.ID.values
    test1 = test.drop(['ID'], axis=1)
    ss = StandardScaler()
    train1 = ss.fit_transform(train1)
    test1 = ss.transform(test1)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic[train.columns[i+1]] = train1[:, i]
    train1 = pd.DataFrame(dic)
    dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)
    return train1, test1


def extend_df(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    sum_col_val = df.sum(axis=1)
    max_col_val = df.max(axis=1)
    min_col_val = df.min(axis=1)
    df['mean'] = mean
    df['std'] = std
    df['sum_col_val'] = sum_col_val
    df['max_col_val'] = max_col_val
    df['min_col_val'] = min_col_val
    return df

def pca(train, test, components=100):
    train_ids = train.ID.values
    targets = train.TARGET.values
    train1 = train.drop(['ID', 'TARGET'], axis = 1)
    test_ids = test.ID.values
    test1 = test.drop(['ID'], axis=1)

    pca = PCA(n_components=components)
    train1 = pca.fit_transform(train1)
    test1 = pca.transform(test1)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic[train.columns[i+1]] = train1[:, i]
    train1 = pd.DataFrame(dic)
    dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)
    return train1, test1

def main():
    # for train set
    fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_train.csv')
    train = pd.read_csv(fname)

    # for test dataset
    fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_test.csv')
    test = pd.read_csv(fname)

    # for pca
    train1, test1 = pca(train, test)
    train1.to_csv(os.path.join('data.path'), 'input', 'pca_train_100.csv')
    test1.to_csv(os.path.join('data.path'), 'input', 'pca_test_100.csv')

    train1, test1 = pca(train, test, components=200)
    train1.to_csv(os.path.join('data.path'), 'input', 'pca_train_200.csv')
    test1.to_csv(os.path.join('data.path'), 'input', 'pca_test_200.csv')

    # for extended
    train1 = extend_df(train.copy())
    test1 = extend_df(test.copy())
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'raw_extend_train.csv'))
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'raw_extend_test.csv'))

    # for scaled
    train1, test1  = scale(train, test)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_train.csv'))
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_test.csv'))

    # for extended scaled
    train1 = extend_df(train1)
    test1 = extend_df(test1)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_extend_train.csv'))
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_extend_test.csv'))

    # for normalized data
    train1, test1 = standard(train, test)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_train.csv'))
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_test.csv'))

    # for extended normalized data
    train1 = extend_df(train1)
    test1 = extend_df(test1)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_extend_train.csv'))
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_extend_test.csv'))
    pass

if __name__ == '__main__':
    main()
    pass