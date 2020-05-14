import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_lsm(x, y):
    xy_med = np.mean(np.multiply(x, y))
    x_med = np.mean(x)
    x_2_med = np.mean(np.multiply(x, x))
    y_med = np.mean(y)
    b1_mnk = (xy_med - x_med * y_med) / (x_2_med - x_med * x_med)
    b0_mnk = y_med - x_med * b1_mnk
    return b0_mnk, b1_mnk


def abs_dev_val(b_arr, x, y):
    return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))


def get_lam(x, y):
    init_b = np.array([0, 1])
    res = minimize(abs_dev_val, init_b, args=(x, y), method='COBYLA')
    return res.x


def draw_res(lsm_0, lsm_1, lam_0, lam_1, x, y, title, fname):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', s=6, label='Выборка')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='blue', label='МНК')
    ax.plot(x, y_lam, color='red', label='МНМ')
    ax.plot(x, y_real, color='green', label='Модель')
    ax.set(xlabel='X', ylabel='Y',
       title=title)
    ax.legend()
    ax.grid()
    #fig.savefig(fname + '.png', dpi=200)
    plt.show()


def lab6():
    x = np.arange(-1.8, 2.1, 0.2)
    eps = np.random.normal(0, 1, size=20)
    y = np.add(np.add(np.full(20, 2), x * 2), eps)
    y2 = np.add(np.add(np.full(20, 2), x * 2), eps)
    y2[0] += 10
    y2[-1] -= 10

    lsm_0, lsm_1 = get_lsm(x, y)
    print(f" МНК, без возмущений: {lsm_0}, {lsm_1}")
    lam_0, lam_1 = get_lam(x, y)
    print(f" МНM, без возмущений: {lam_0}, {lam_1}")

    lsm_02, lsm_12 = get_lsm(x, y2)
    print(f" МНК, с возмущениями: {lsm_02}, {lsm_12}")
    lam_02, lam_12 = get_lam(x, y2)
    print(f" МНM, с возмущениями: {lam_02}, {lam_12}")

    draw_res(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Выборка без возмущений', 'report/figure/no_dev')
    draw_res(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Выборка с возмущениями', 'report/figure/dev')