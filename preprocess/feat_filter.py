# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

import os
import itertools
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from utils.config_utils import Config


def remove_feat_constants(data_frame):
    """
    Remove feature vectors containing one unique value, because such features do not have predictive value.
    :param data_frame:
    :return:
    """
    print("Deleting zero variance features...")
    # Let's get the zero variance features by fitting VarianceThreshold
    # selector to the data, but let's not transform the data with
    # the selector because it will also transform our Pandas data frame into
    # NumPy array and we would like to keep the Pandas data frame. Therefore,
    # let's delete the zero variance features manually.
    n_features_originally = data_frame.shape[1]
    selector = VarianceThreshold()
    selector.fit(data_frame)
    # Get the indices of zero variance feats
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(data_frame.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
    # Delete zero variance feats from the original pandas data frame
    feat_deleted = [name for name in data_frame.columns[feat_ix_delete]]
    data_frame = data_frame.drop(labels=feat_deleted,
                                 axis=1)
    # Print info
    n_features_deleted = feat_ix_delete.size
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame, feat_deleted


def remove_feat_identicals(data_frame):
    """
    Find feature vectors having the same values in the same order and remove all but one of those redundant features.
    :param data_frame:
    :return:
    """
    print("Deleting identical features...")
    n_features_originally = data_frame.shape[1]
    # Find the names of identical features by going through all the
    # combinations of features (each pair is compared only once).
    feat_names_delete = []
    for feat_1, feat_2 in itertools.combinations(
            iterable=data_frame.columns, r=2):
        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):
            feat_names_delete.append(feat_2)
    feat_names_delete = np.unique(feat_names_delete)
    # Delete the identical features
    data_frame = data_frame.drop(labels=feat_names_delete, axis=1)
    n_features_deleted = len(feat_names_delete)
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame, feat_names_delete


if __name__=='__main__':
    data_path = os.path.join(Config.get_string('data.path'), 'input')
    raw_train_path = os.path.join(data_path, 'train.csv')
    raw_test_path = os.path.join(data_path, 'test.csv')
    print("= Reading raw datasets ...")
    raw_train = pd.read_csv(raw_train_path, index_col=0, sep=',')
    raw_test = pd.read_csv(raw_test_path, index_col=0, sep=',')

    print("= Droping useless feat columns in training set ")
    raw_train, feat_to_delete = remove_feat_constants(raw_train)
    raw_train, temp = remove_feat_identicals(raw_train)
    feat_to_delete.extend(temp)

    print("= Droping useless feat columns in test set:")
    print feat_to_delete
    raw_test = raw_test.drop(feat_to_delete, axis=1)

    print("= Saving filtered dataset ...")
    filtered_train_path = os.path.join(data_path, 'filtered_train.csv')
    filtered_test_path = os.path.join(data_path, 'filtered_test.csv')
    raw_train.to_csv(filtered_train_path)
    raw_test.to_csv(filtered_test_path)
    print("= DONE! ")