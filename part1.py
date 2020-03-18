import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
from scipy.special import factorial

import utils
from utils import *


def gen_distr_histogram(name: str, pdf, grid, title: str, figName: str):
    x = np.linspace(grid[0], grid[1], 1000)
    distr_10 = utils.get_distribution(name, 10)
    distr_50 = utils.get_distribution(name, 50)
    distr_1000 = utils.get_distribution(name, 1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.plot(x, pdf(x))
    sns.distplot(distr_10, color="r", kde=False, norm_hist=True)
    plt.title('10 значений')

    plt.subplot(1, 3, 2)
    plt.plot(x, pdf(x))
    sns.distplot(distr_50, color="y", kde=False, norm_hist=True)
    plt.title(title + ' распределение\n' + '50 значений')

    plt.subplot(1, 3, 3)
    plt.plot(x, pdf(x))
    sns.distplot(distr_1000, color="g", kde=False, norm_hist=True)
    plt.title('1000 значений')

    plt.show()
    f.savefig(figName, dpi=200)


def lab1():
    gen_distr_histogram('Normal', stats.norm.pdf, (-10, 10), 'Нормальное', 'norm.png')
    gen_distr_histogram('Cauchy', stats.cauchy.pdf, (-10, 10), 'Коши', 'cauchy.png')
    gen_distr_histogram('Laplace', stats.laplace.pdf, (-10, 10), 'Лапласа', 'laplace.png')
    gen_distr_histogram('Poisson', lambda x: np.exp(-10) * np.power(10, x) / factorial(x), (0, 30), 'Пуассона', 'poisson.png')
    gen_distr_histogram('Uniform', stats.uniform.pdf, (-10, 10), 'Равномерное', 'uniform.png')
