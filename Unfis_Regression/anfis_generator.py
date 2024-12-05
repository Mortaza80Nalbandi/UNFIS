import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import itertools
from sklearn.datasets import load_iris
plt.rcParams['axes.xmargin'] = 0  # remove margins from all plots
##############################################################################

def get_membsFuncs(data_id):
    MEMBFUNCS,n_input = None,None
    if data_id == 'mackey':
        n_input = 4
        MEMBFUNCS = [
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}}
        ]
    elif data_id == 'sinc':
        n_input = 2
        MEMBFUNCS = [
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}},
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}}
        ]
    elif data_id == 'sin':
        n_input = 1
        MEMBFUNCS = [
            {'function': 'gaussian',
             'n_memb': 3,
             'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                               'trainable': True},
                        'sigma': {'value': [1.0, 1.0, 1.0],
                                  'trainable': True}}}
        ]
    return MEMBFUNCS,n_input

def gen_data(data_id: str = 'mackey', n_obs: int = 1000, n_input: int = 4, lag: int = 1):
    test_size=0.3
    # Mackey
    if data_id == 'mackey':
        y = mackey(124 + n_obs + n_input)[124:]
        X, y = gen_X_from_y(y, n_input, lag)
    elif data_id == 'sin':
        X, y = sin_data(n_obs)
        assert n_input == 1, 'Nonlin sinc equation data set requires n_input==1. Please chhange to 1.'
        test_size = 0.05
        X=[X]
    # Nonlin sinc equation
    elif data_id == 'sinc':
        X, y = sinc_data(n_obs)
        test_size = 0.05
        assert n_input == 2, 'Nonlin sinc equation data set requires n_input==2. Please chhange to 2.'



    # to torch
    if data_id == 'sin':
        X, y = torch.Tensor(X[0]), torch.Tensor(y)
        X=X.unsqueeze(1)
    else:
        # standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        scaler = StandardScaler()
        y = scaler.fit_transform(y)
        X, y = torch.Tensor(X), torch.Tensor(y)
    # split data into test and train set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    return X, X_train, X_test, y, y_train, y_test

##############################################################################


# Mackey-Glass series computation
def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x


# Modelling a two-Input Nonlinear Function (Sinc Equation)
def sinc_equation(x1, x2):
    return ((np.sin(x1) / x1) * (np.sin(x2) / x2))


def sinc_data(n_obs, noise=False):
    pts = np.linspace(-1, 1, n_obs)+0.0001
    X = np.array(list(itertools.product(pts, pts)))
    y = sinc_equation(X[:, 0], X[:, 1]).reshape(-1, 1)
    if noise == True:
        y = y + np.random.randn(n_obs) * 0.1
    return X.astype('float32'), y.astype('float32')
def sin_data(n_obs, multiplier=2, noise=False):
    X = np.linspace(-3, 3, n_obs)
    y = np.sin(X).reshape(-1, 1)
    if noise == True:
        y = y + np.random.randn(n_obs) * 0.1
    return X.astype('float32'), y.astype('float32')

# Generate a input matrix X from time series y
def gen_X_from_y(x, n_input=1, lag=1):
    n_obs = len(x) - n_input * lag

    data = np.zeros((n_obs, n_input + 1))
    for t in range(n_input * lag, n_obs + n_input * lag):
        data[t - n_input * lag, :] = [x[t - i * lag]
                                      for i in range(n_input + 1)]
    X = data[:, 1:].reshape(n_obs, -1)
    y = data[:, 0].reshape(n_obs, 1)

    return X.astype('float32'), y.astype('float32')


