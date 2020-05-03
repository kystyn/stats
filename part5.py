import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import utils

correlation_list = [0, 0.5, 0.9]
len_list = [20, 60, 100]


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor,
                      edgecolor='midnightblue', **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def correlation_params():
    for size in len_list:
        table = []
        for cor in correlation_list:
            table.append(['$\\rho = ' + str(cor) + '~\eqref{eq:correlation}$', '$r ~\eqref{eq:pearson}$',
             '$r_Q ~\eqref{eq:rq}$', '$r_S ~\eqref{eq:spearman}$'])
            pearson1000 = []
            quadrant1000 = []
            spearman1000 = []
            for i in range(1000):
                sample_d = np.random.multivariate_normal([0, 0], [[1, cor], [cor, 1]], size=size)

                cov = np.cov(sample_d[:, 0], sample_d[:, 1])
                pearson1000.append(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

                n1 = 0
                n2 = 0
                n3 = 0
                n4 = 0
                for i in sample_d:
                    if i[0] >= 0:
                        if i[1] >= 0:
                            n1 += 1
                        else:
                            n4 += 0
                    else:
                        if i[1] >= 0:
                            n2 += 1
                        else:
                            n3 += 0

                quadrant1000.append(((n1 + n3) - (n2 + n4)) / size)
                spearman1000.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1])[0])

            table.append(['$E(z)$', np.mean(pearson1000), np.mean(spearman1000), np.mean(quadrant1000)])
            table.append(['$E(z^2)$', np.mean(np.asarray([el * el for el in pearson1000])),
                                      np.mean(np.asarray([el * el for el in spearman1000])),
                                      np.mean(np.asarray([el * el for el in quadrant1000]))])
            table.append(['$D(z)$', np.std(pearson1000), np.std(spearman1000), np.std(quadrant1000)])

            if size != len_list[-1]:
                table.append(['', '', '', ''])

        utils.table_to_tex(open('report/table/correlation/' + str(size) + '.tex', 'w'), table)

    # normal mix
    table = []
    for size in len_list:
        table.append(['$n = ' + str(size) + '$', '$r ~\eqref{eq:pearson}$',
                      '$r_Q ~\eqref{eq:rq}$', '$r_S ~\eqref{eq:spearman}$'])
        pearson1000 = []
        quadrant1000 = []
        spearman1000 = []
        for i in range(1000):
            sample_d = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) +\
                       0.1 * np.random.multivariate_normal([0, 0], [[10, 0.9], [0.9, 10]], size=size)

            cov = np.cov(sample_d[:, 0], sample_d[:, 1])
            pearson1000.append(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

            n1 = 0
            n2 = 0
            n3 = 0
            n4 = 0
            for i in sample_d:
                if i[0] >= 0:
                    if i[1] >= 0:
                        n1 += 1
                    else:
                        n4 += 0
                else:
                    if i[1] >= 0:
                        n2 += 1
                    else:
                        n3 += 0

            quadrant1000.append(((n1 + n3) - (n2 + n4)) / size)
            spearman1000.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1])[0])

        table.append(['$E(z)$', np.mean(pearson1000), np.mean(spearman1000), np.mean(quadrant1000)])
        table.append(['$E(z^2)$', np.mean(np.asarray([el * el for el in pearson1000])),
                      np.mean(np.asarray([el * el for el in spearman1000])),
                      np.mean(np.asarray([el * el for el in quadrant1000]))])
        table.append(['$D(z)$', np.std(pearson1000), np.std(spearman1000), np.std(quadrant1000)])

        if cor != correlation_list[-1]:
            table.append(['', '', '', ''])
    utils.table_to_tex(open('report/table/correlation/mix.tex', 'w'), table)


for size in len_list:
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    for ax, rho in zip(axs, correlation_list):
        sample_d = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)
        ax.scatter(sample_d[:, 0], sample_d[:, 1], color='cornflowerblue', s=3)
        ax.set_xlim(min(sample_d[:, 0]) - 2, max(sample_d[:, 0]) + 2)
        ax.set_ylim(min(sample_d[:, 1]) - 2, max(sample_d[:, 1]) + 2)
        print(min(sample_d[:, 0]), max(sample_d[:, 0]))
        print(min(sample_d[:, 1]), max(sample_d[:, 1]))
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        title = r'n = ' + str(size) + r', $\rho$  = ' + str(rho)
        ax.set_title(title)
        confidence_ellipse(sample_d[:, 0], sample_d[:, 1], ax)
    fig.savefig('N_2d_' + str(size) + '.png', dpi=200)
    plt.show()


# fig, axs = plt.subplots(1, 3, figsize=(9, 3))
# for ax, size in zip(axs, len_list):
#     size = 2
#     sample_d = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) \
#                + 0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], size=size)
#     ax.scatter(sample_d[:, 0], sample_d[:, 1], color='cornflowerblue', s=3)
#     ax.set_xlim(min(sample_d[:, 0]) - 2, max(sample_d[:, 0]) + 2)
#     ax.set_ylim(min(sample_d[:, 1]) - 2, max(sample_d[:, 1]) + 2)
#     print(min(sample_d[:, 0]), max(sample_d[:, 0]))
#     print(min(sample_d[:, 1]), max(sample_d[:, 1]))
#     ax.axvline(c='grey', lw=1)
#     ax.axhline(c='grey', lw=1)
#     title = r'mixed: n = ' + str(size)
#     ax.set_title(title)
#     confidence_ellipse(sample_d[:, 0], sample_d[:, 1], ax)
# fig.savefig('mixed_2d.png', dpi=200)
# plt.show()


def lab5():
    correlation_params()
