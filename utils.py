import numpy as np
import math
import scipy.stats as stats
from scipy.special import factorial

distr_samples = \
{
    'Normal': lambda num: np.random.normal(0, 1, num),
    'Cauchy': lambda num: np.random.standard_cauchy(num),
    'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
    'Poisson': lambda num: np.random.poisson(10, num),
    'Uniform': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
}

distr_cdf = \
{
    'Normal': lambda x: stats.norm.cdf(x, 0, 1),
    'Cauchy': lambda x: stats.cauchy.cdf(x, 0, 1),
    'Laplace': lambda x: stats.laplace.cdf(x, 0, math.sqrt(2) / 2),
    'Poisson': lambda x: stats.poisson.cdf(x, 10),
    'Uniform': lambda x: stats.uniform.cdf(x, -math.sqrt(3), 2 * math.sqrt(3))
}

distr_pdf = \
{
    'Normal': lambda x: stats.norm.pdf(x, 0, 1),
    'Cauchy': lambda x: stats.cauchy.pdf(x, 0, 1),
    'Laplace': lambda x: stats.laplace.pdf(x, 0, math.sqrt(2) / 2),
    'Poisson': lambda x: np.exp(-10) * np.power(10, x) / factorial(x),
    'Uniform': lambda x: stats.uniform.pdf(x, -math.sqrt(3), 2 * math.sqrt(3))
}


def get_distribution_sample(distr_name, num):
    return distr_samples.get(distr_name)(num)


def get_distribution_cdf(x, distr_name):
    return distr_cdf.get(distr_name)(x)


def get_distribution_pdf(x, distr_name):
    return distr_pdf.get(distr_name)(x)


def get_quartil(sample, p):
    return sample[math.ceil(sample.shape[0] * p + 0.5)]


def table_to_tex(f, table: list, header=''):
    f.write('\\begin{tabular}{' + '|c' * len(table[0]) + '|}' + '\n')
    f.write(header)
    f.write('\hline\n')
    for row in table:
        for cell in row[:-1]:
            f.write(str(cell) + ' & ')
        f.write(str(row[-1]) + '\\\\' + '\n')
        f.write('\hline\n')
    f.write('\\end{tabular}' + '\n\n')
