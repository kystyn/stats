import numpy as np
import math
import scipy.stats as stats
from utils import *


def gen_table():
    tables = []
    len_list = [10, 100, 1000]
    for dist_name in distributions.keys():
        table = []
        for d_num in len_list:
            mean = []
            med = []
            z_r = []
            z_q = []
            z_tr = []
            for it in range(1000):
                sample_d = get_distribution(dist_name, d_num)
                sample_d_sorted = np.sort(sample_d)
                mean.append(np.mean(sample_d))
                med.append(np.median(sample_d))
                z_r.append((sample_d_sorted[0] + sample_d_sorted[-1]) / 2)
                z_q.append((get_quartil(sample_d_sorted, 0.25) + get_quartil(sample_d_sorted, 0.75)) / 2)
                z_tr.append(stats.trim_mean(sample_d, 0.25))
            table.append([dist_name + ' ' + str(d_num), '', '', '', '', ''])
            table.append(["$E(z)$ (\\ref{ez})",
                          np.around(np.mean(mean), decimals=6),
                          np.around(np.mean(med), decimals=6),
                          np.around(np.mean(z_r), decimals=6),
                          np.around(np.mean(z_q), decimals=6),
                          np.around(np.mean(z_tr), decimals=6)])
            table.append(["$D(z)$ (\\ref{dz})",
                          np.around(np.std(mean) * np.std(mean), decimals=6),
                          np.around(np.std(med) * np.std(med), decimals=6),
                          np.around(np.std(z_r) * np.std(z_r), decimals=6),
                          np.around(np.std(z_q) * np.std(z_q), decimals=6),
                          np.around(np.std(z_tr) * np.std(z_tr), decimals=6)])
        tables.append((dist_name, table))

    return tables


def lab2():
    tables = gen_table()
    for table in tables:
        table_to_tex(open('report/table/' + table[0].lower() + '.tex', 'w'), table[1],
                     '\hline\n & $\overline{x}$ (\\ref{mean}) & $med x$ (\\ref{med})'
                     ' & $z_R$ (\\ref{zr}) & $z_Q$ (\\ref{zq}) & $z_{tr}$ (\\ref{tr_mean})\\\\' + '\n'
                     )