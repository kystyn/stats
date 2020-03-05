import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
from scipy.special import factorial

import utils
from utils import *


def gen_distr_histogram(name: str, pdf_func, grid, title: str, pdf_val, figName: str):
    x = np.linspace(grid[0], grid[1], 1000)
    distr_10 = utils.get_distribution(name, 10)
    distr_50 = utils.get_distribution(name, 50)
    distr_1000 = utils.get_distribution(name, 1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(1, 3, 1)
    sns.distplot(distr_10, color="g")
    plt.plot(x, pdf_func(x, pdf_val[0], pdf_val[1]), color="orange")
    plt.title('10 значений', y=-0.3)

    plt.subplot(1, 3, 2)
    sns.distplot(distr_50, color="g")
    plt.plot(x, pdf_func(x, pdf_val[0], pdf_val[1]), color="orange")
    plt.title('50 значений \n' + title + ' распределение', y=-0.4)

    plt.subplot(1, 3, 3)
    sns.distplot(distr_1000, color="g", label='Гистограмма сгенерированных значений')
    plt.plot(x, pdf_func(x, pdf_val[0], pdf_val[1]), color="orange", label='График плотности распределения')
    plt.title('1000 значений', y=-0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
    plt.show()
    f.savefig(figName, dpi=200)


def lab1():
    gen_distr_histogram('Normal', stats.norm.pdf, (-10, 10), 'Нормальное', (0, 1), 'norm.png')
    gen_distr_histogram('Cauchy', stats.cauchy.pdf, (-10, 10), 'Коши', (0, 1), 'cauchy.png')
    gen_distr_histogram('Laplace', stats.laplace.pdf, (-4, 4), 'Коши', (0, math.sqrt(2)/2), 'cauchy.png')
    gen_distr_histogram('Poisson', lambda x, g1, g2: np.exp(-10) * np.power(10, x)/factorial(x), (-4, 4), 'Пуассона',
                        (0, math.sqrt(2) / 2), 'poisson.png')
    gen_distr_histogram('Uniform', stats.uniform.pdf, (-10, 10), 'Пуассона',
                        (-math.sqrt(3), 2 * math.sqrt(3)), 'uniform.png')
