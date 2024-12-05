import matplotlib.pyplot as plt
from typing import Optional
from pandas.core.frame import DataFrame

plt.style.use('seaborn')


def plt_prediction(y, y_pred, save_path):
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'r', label='trained')
    plt.plot(range(len(y)), y, 'b', label='original')
    plt.legend(loc='upper left')
    plt.xlabel("Datas")
    plt.ylabel("Ys")
    plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')

    plt.show()


def plt_learningcurves(losses, save_path):
    plt.figure()
    plt.plot(range(len(losses[0])), losses[0], 'b', label='Train')
    plt.plot(range(len(losses[1])), losses[1], 'r', label='Valid')
    plt.legend(loc='upper left')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')

    plt.show()

