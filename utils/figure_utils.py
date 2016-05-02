# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp


def draw_roc_figure(pred, label, name='res', out_fname=None):
    fpr, tpr, thresh = roc_curve(label, pred)
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='{0} with auc {1:.2f}'.format(name, area))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    if out_fname:
        plt.savefig(out_fname)
        plt.clf()
    pass
