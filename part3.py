import matplotlib.pyplot as plt
from utils import *

def lab3():
    len_list = [20, 100]

    texfile = open('report/table/trashdata.tex', 'w')
    table = []

    for dist_name in distr_samples.keys():
        fig, ax = plt.subplots(1, 1)
        bp_data = [get_distribution_sample(dist_name, length) for length in len_list]
        trashdata_part = [0] * len(bp_data)
        for idx in range(1000):
            data = [get_distribution_sample(dist_name, length) for length in len_list]
            for datum in data:
                datum_sorted = np.sort(datum)
                q1 = get_quartil(datum_sorted, 0.25)
                q3 = get_quartil(datum_sorted, 0.75)
                moustache_left = q1 - 1.5 * (q3 - q1)
                moustache_right = q3 + 1.5 * (q3 - q1)
                trashdata_part[data.index(datum)] +=\
                    (len(datum_sorted[datum_sorted > moustache_right]) +
                     len(datum_sorted[datum_sorted < moustache_left])) / len(datum)
        for idx in range(len(bp_data)):
            table.append([dist_name + ', ' + str(len(bp_data[idx])), trashdata_part[idx] / 1000])

        ax.set_title(dist_name + ' distribution')
        ax.boxplot(bp_data, vert=False, positions=len_list, widths=[35, 35])
        plt.savefig('report/figure/boxplot/' + dist_name + '_boxplot.png', dpi=600)

    table_to_tex(texfile, table, '\hline\n Выборка & Доля выбросов\\\\' + '\n')