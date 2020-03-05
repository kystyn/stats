import numpy as np
import math

distributions = \
{'Normal': lambda num: np.random.normal(0, 1, num),
 'Cauchy': lambda num: np.random.standard_cauchy(num),
 'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
 'Poisson': lambda num: np.random.poisson(10, num),
 'Uniform': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
};


def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)