import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from utils import *

len_list = [20, 60, 100]


def cdf_plots():
    samples = [[], [], []]
    for dist_name in distr_samples.keys():
        fig, ax = plt.subplots(1, 3)
        for i in range(len(len_list)):
            if dist_name == 'Poisson':
                r = (6, 14)
            else:
                r = (-4, 4)
            x = np.linspace(r[0], r[1], 1000)
            samples[i] = get_distribution_sample(dist_name, len_list[i])
            ecdf = sm.distributions.ECDF(samples[i])
            y = ecdf(x)
            ax[i].plot(x, y, color='m', label='Empirical distribution function')
            y = get_distribution_cdf(x, dist_name)
            ax[i].plot(x, y, color='orange', label='Distribution function')
            ax[i].set_title(dist_name + '\n n = ' + str(len_list[i]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
        plt.show()
        fig.savefig(dist_name + '_cdf.png', dpi=400)


def kde_plots():
    for dist_name in distr_samples.keys():
        for i in range(len(len_list)):
            fig, ax = plt.subplots(1, 3)
            if dist_name == 'Poisson':
                r = (6, 14)
            else:
                r = (-4, 4)
            x = np.linspace(r[0], r[1], 1000)
            y = get_distribution_pdf(x, dist_name)
            samples = get_distribution_sample(dist_name, len_list[i])
            samples = samples[samples <= r[1]]
            samples = samples[samples >= r[0]]
            kde = stats.gaussian_kde(samples)
            kde.set_bandwidth(bw_method='silverman')
            h_n = kde.factor
            sns.kdeplot(samples, ax=ax[0], bw=h_n/2, color='m')
            ax[0].set_title(r'$h = \frac{h_n}{2}$')
            ax[0].plot(x, y, color='orange')
            ax[1].plot(x, y, color='orange')
            ax[2].plot(x, y, color='orange', label='Real density function')
            sns.kdeplot(samples, ax=ax[1], bw=h_n, color='m')
            ax[1].set_title(r'$h = h_n$')
            sns.kdeplot(samples, ax=ax[2], bw=2*h_n, color='m', label='Kernel density estimation')
            ax[2].set_title(r'$h = 2 * h_n$')
            fig.suptitle(dist_name + ' n = ' + str(len_list[i]))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
            plt.show()
            fig.savefig(dist_name + str(len_list[i]) + '_kde.png', dpi=200)


def lab4():
    cdf_plots()
    kde_plots()


lab4()