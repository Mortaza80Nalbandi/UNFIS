import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import itertools
from sklearn.datasets import load_iris,load_wine
plt.rcParams['axes.xmargin'] = 0  # remove margins from all plots
##############################################################################

def get_membsFuncs(dataid):
    MEMBFUNCS,n_input = None,None
    if(dataid == 'iris'):
        n_input = 4
        MEMBFUNCS = [
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}}
        ]
    elif dataid=='wine':
        n_input = 13
        MEMBFUNCS = [
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [2, 4.0, 6],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}}
        ]
    return MEMBFUNCS,n_input

def gen_data(dataid):
    if(dataid=="iris"):
        test_size = 0.2
        data = load_iris()
    elif (dataid=="wine"):
        test_size = 0.2
        data = load_wine()
    X = data['data']
    y = data['target']
    X, y = torch.Tensor(X), torch.Tensor(y)
    y = y.to(torch.int64)
    # split data into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    return X, X_train, X_test, y, y_train, y_test

