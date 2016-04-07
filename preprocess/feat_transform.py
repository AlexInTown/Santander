# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
import cPickle as cp
from utils.config_utils import Config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


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
    df_ids = None
    df_targets = None
    df_ids = df.ID.values
    if 'ID' in df.columns and 'TARGET' in df.columns:
        df_targets = df.TARGET.values
        df = df.drop(['ID', 'TARGET'], axis=1)
    elif 'ID' in df.columns:
        df = df.drop(['ID'], axis=1)

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
    df['ID'] = df_ids
    if df_targets is not None:
        df['TARGET'] = df_targets
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
        dic['pca_'+str(i)] = train1[:, i]
    train1 = pd.DataFrame(dic)
    dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic['pca_'+str(i)] = test1[:, i]
    test1 = pd.DataFrame(dic)
    return train1, test1


def pca_extend(train, test, components=10):
    train_ids = train.ID.values
    targets = train.TARGET.values
    train0 = train.drop(['ID', 'TARGET'], axis = 1)
    test_ids = test.ID.values
    test0 = test.drop(['ID'], axis=1)

    pca = PCA(n_components=components)
    train1 = pca.fit_transform(train0)
    test1 = pca.transform(test0)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic['pca_'+str(i)] = train1[:, i]
    for i in xrange(train0.shape[1]):
        dic[train.columns[i+1]] = train0[:, i]
    train1 = pd.DataFrame(dic)
    dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic['pca_'+str(i)] = test1[:, i]
    for i in xrange(test0.shape[1]):
        dic[test.columns[i+1]] = test0[:, i]
    test1 = pd.DataFrame(dic)
    return train1, test1


def main():
    # for train set
    fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_train.csv')
    train = pd.read_csv(fname)

    # for test dataset
    fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_test.csv')
    test = pd.read_csv(fname)



    # for extended
    print '--- extending raw dataset ---'
    train1 = extend_df(train.copy())
    test1 = extend_df(test.copy())
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'raw_extend_train.csv'), index=0)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'raw_extend_test.csv'), index=0)

    # for scaled
    print '--- scaling raw dataset to [0, 1] ---'
    train1, test1 = scale(train, test)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_train.csv'), index=0)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_test.csv'), index=0)

    # for extended scaled
    print '--- extending scaled dataset ---'
    train1 = extend_df(train1)
    test1 = extend_df(test1)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_extend_train.csv'), index=0)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'scaled_extend_test.csv'), index=0)

    # for normalized data
    print '--- standardizing dataset ---'
    train1, test1 = standard(train, test)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_train.csv'), index=0)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_test.csv'), index=0)

    # for pca
    print "--- transforming pca dataset  ----"
    train2, test2 = pca(train1, test1, components=100)
    train2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca100_train.csv'), index=0)
    test2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca100_test.csv'), index=0)

    train2, test2 = pca(train1, test1, components=200)
    train2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca200_train.csv'), index=0)
    test2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca200_test.csv'), index=0)

    # for pca extend
    print "--- standard -> standard + pca  ----"
    train2, test2 = pca_extend(train1, test1, components=10)
    train2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca10_and_standard_train.csv'), index=0)
    test2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca10_and_standard_test.csv'), index=0)

    train2, test2 = pca_extend(train1, test1, components=20)
    train2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca20_and_standard_train.csv'), index=0)
    test2.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'pca20_and_standard_test.csv'), index=0)

    del train2
    del test2

    # for extended normalized data
    print '--- extending standard dataset ---'
    train1 = extend_df(train1)
    test1 = extend_df(test1)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_extend_train.csv'), index=0)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 'input', 'standard_extend_test.csv'), index=0)
    pass

if __name__ == '__main__':
    main()
    pass