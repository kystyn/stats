import numpy as np
import math

distributions = \
{'Normal': lambda num: np.random.normal(0, 1, num),
 'Cauchy': lambda num: np.random.standard_cauchy(num),
 'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
 'Poisson': lambda num: np.random.poisson(10, num),
 'Uniform': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
}


def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)


def get_quartil(sample, p):
    return sample[math.ceil(sample.shape[0] * p + 0.5)]


def table_to_tex(f, table: list, header):
    f.write('\\begin{tabular}{' + '|c' * len(table[0]) + '|}' + '\n')
    f.write(header)
    f.write('\hline\n')
    for row in table:
        for cell in row[:-1]:
            f.write(str(cell) + ' & ')
        f.write(str(row[-1]) + '\\\\' + '\n')
        f.write('\hline\n')
    f.write('\\end{tabular}' + '\n\n')