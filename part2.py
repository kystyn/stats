import numpy as np
import scipy.stats as stats


len_list = [10, 100, 1000]
distr_type = ['Norm', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']



for dist_name in distr_type:
    print(f"Распределение {dist_name}")
    resTable = PrettyTable()
    resTable.float_format = "2.2"
    resTable.field_names = ["Characteristic", "Mean", "Median", "Zr", "Zq", "Ztr"]
    for d_num in len_list:
        mean = []
        med = []
        z_r = []
        z_q = []
        z_tr = []
        for it in range(1000):
            sample_d = get_distr_samples(dist_name, d_num)
            sample_d_sorted = np.sort(sample_d)
            mean.append(np.mean(sample_d))
            med.append(np.median(sample_d))
            z_r.append((sample_d_sorted[0] + sample_d_sorted[-1]) / 2)
            z_q.append((get_quartil(sample_d, 0.25) + get_quartil(sample_d, 0.75)) / 2)
            z_tr.append(stats.trim_mean(sample_d, 0.25))
        resTable.add_row([dist_name + " E(z) " + str(d_num),
                          np.around(np.mean(mean), decimals=6),
                          np.around(np.mean(med), decimals=6),
                          np.around(np.mean(z_r), decimals=6),
                          np.around(np.mean(z_q), decimals=6),
                         np.around(np.mean(z_tr), decimals=6)])
        resTable.add_row([dist_name + " D(z) " + str(d_num),
                          np.around(np.std(mean) * np.std(mean), decimals=6),
                          np.around(np.std(med) * np.std(med), decimals=6),
                          np.around(np.std(z_r) * np.std(z_r), decimals=6),
                          np.around(np.std(z_q) * np.std(z_q), decimals=6),
                          np.around(np.std(z_tr) * np.std(z_tr), decimals=6)])

    print(resTable)