# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
import cPickle as cp
from utils.config_utils import Config
import pandas as pd

def fix_abnormal_value(df):
    return df

def uniform_value(df):
    pass

def extend_feature(df):
    pass

def main():
    fname = os.path.join(Config.get_string('data.path'), 'input', 'filtered_train.csv')
    df = pd.read_csv(fname)

    pass

if __name__ == '__main__':
    pass