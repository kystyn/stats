import os
import sys
import json
import numpy as np
from numpy import char, array
from matplotlib import pyplot
from frequency_processing import calculate_frequency
from sawtooth_detection import get_sawtooth_indexes
from math import sqrt

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(WORK_DIR)
SHT_DIR = PARENT_DIR + '/SHT/'
OUT_DIR = WORK_DIR + '/out/'

import pyglobus


MINIMUM_SAWTOOTH_LENGTH = 10000  # in steps
SIGNAL_NAMES = {
    18: b'SXR 15 \xec\xea\xec\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    19: b'SXR 27 \xec\xea\xec\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    20: b'SXR 50 mkm\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    26: b'SXR 80 mkm\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
}

def get_borders(freqs):
    l_border_t = 0
    r_border_t = float('inf')
    for freq in freqs:
        if l_border_t < min(freq[0]):
            l_border_t = min(freq[0])
        if r_border_t > max(freq[0]):
            r_border_t = max(freq[0])
    assert (l_border_t < r_border_t)

    b_border_f = float('inf')
    u_border_f = 0
    for freq in freqs:
        a_ind = 0
        b_ind = len(freq[0])
        for i in range(0, len(freq[0])):
            if freq[0][i] >= l_border_t:
                a_ind = i
                break
        for i in range(len(freq[0]) - 1, 0, -1):
            if freq[0][i] <= r_border_t:
                b_ind = i
                break
        temp = array(freq[1])[a_ind:b_ind]
        if b_border_f > min(temp):
            b_border_f = min(temp)
        if u_border_f < max(temp):
            u_border_f = max(temp)
    assert (b_border_f < u_border_f)

    return l_border_t, r_border_t, b_border_f, u_border_f

def process_freqs():
    data_file = open('data.json')
    data = json.load(data_file)

    for sht_data in data['sht_data']:
        sht_number = sht_data[0]

        sht_file_name = 'sht' + str(sht_number) + '.SHT'
        sht_abs_path = SHT_DIR + sht_file_name

        if not os.path.isfile(sht_abs_path):
            print('WARNING: unable process file\n' +
                  sht_abs_path)
            continue

        freqs = []
        sawtooth_signals_numbers = []

        fig = pyplot.figure()
        signal_numbers = sht_data[1]
        for signal_number in signal_numbers:
            sht_reader = pyglobus.util.ShtReader(sht_abs_path)
            signal_name = SIGNAL_NAMES[signal_number]
            #signal_name = sht_reader.get_signal_name(signal_number)
            signal = sht_reader.get_signals(signal_name)[0]

            empirical_indexes = sht_data[2]
            sawtooth_indexes = get_sawtooth_indexes(signal)
            if sawtooth_indexes[0] < empirical_indexes[0] or sawtooth_indexes[1] > empirical_indexes[1]:
                sawtooth_indexes = empirical_indexes

            if sawtooth_indexes[1] - sawtooth_indexes[0] > MINIMUM_SAWTOOTH_LENGTH:
                sawtooth_signals_numbers.append(signal_number)
                freq = calculate_frequency(signal, sawtooth_indexes)
                freqs.append(freq)
                pyplot.plot(freq[0], freq[1], '-o', markersize=3)
            else:
                print('WARNING: unable to detect sawtooth sequence of minimum length for data\n' +
                      sht_file_name + '\n'
                      'processing signal number ' +
                      str(signal_number))

        l_border_t, r_border_t, b_border_f, u_border_f = get_borders(freqs)
        pyplot.xlim(l_border_t, r_border_t)
        pyplot.ylim(b_border_f - 50, u_border_f + 50)
        pyplot.legend(char.mod('%d', sawtooth_signals_numbers))
        pyplot.xlabel('Time, s')
        pyplot.ylabel('Frequency, Hz')
        pyplot.title('Frequency of ' + sht_file_name + ' data')
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        fig.savefig(OUT_DIR + 'freq_sht' + str(sht_number) + '.png')

        return freqs


def process_periods():
    data_file = open('data.json')
    data = json.load(data_file)

    for sht_data in data['sht_data']:
        sht_number = sht_data[0]

        sht_file_name = 'sht' + str(sht_number) + '.SHT'
        sht_abs_path = SHT_DIR + sht_file_name

        if not os.path.isfile(sht_abs_path):
            print('WARNING: unable process file\n' +
                  sht_abs_path)
            continue

        freqs = []
        sawtooth_signals_numbers = []

        fig = pyplot.figure()
        signal_numbers = sht_data[1]
        for signal_number in signal_numbers:
            sht_reader = pyglobus.util.ShtReader(sht_abs_path)
            signal_name = SIGNAL_NAMES[signal_number]
            # signal_name = sht_reader.get_signal_name(signal_number)
            signal = sht_reader.get_signals(signal_name)[0]

            empirical_indexes = sht_data[2]
            sawtooth_indexes = get_sawtooth_indexes(signal)
            if sawtooth_indexes[0] < empirical_indexes[0] or sawtooth_indexes[1] > empirical_indexes[1]:
                sawtooth_indexes = empirical_indexes

            if sawtooth_indexes[1] - sawtooth_indexes[0] > MINIMUM_SAWTOOTH_LENGTH:
                sawtooth_signals_numbers.append(signal_number)
                freq = calculate_frequency(signal, sawtooth_indexes)
                freqs.append(freq)

                prevextremum = freq[0][0]
                prevextremumi = 0
                #prevmax = 0
                #prevmaxi =0
                periods = []
                finish_dist = []
                for i in range(1, len(freq[1]) - 1):
                    #  and freq[0][i] - prevextremum >= 0.002
                    if (freq[1][i - 1] > freq[1][i] and freq[1][i] < freq[1][i + 1]) or\
                       (freq[1][i - 1] < freq[1][i] and freq[1][i] > freq[1][i + 1]): #  and freq[0][i] - prevextremum >= 0.002
                        finish_dist.append(freq[0][-1] - freq[0][i])
                        periods.append(sqrt(((freq[0][i] - prevextremum) * 10000) ** 2 + (freq[1][i] - freq[1][prevextremumi]) ** 2))
                        prevextremum = freq[0][i]
                        prevextremumi = i

                #periods.append(freq[0][-1] - prevextremum)
                #periods.remove(periods[0])
                #periods.remove(periods[-1])
                pyplot.plot(finish_dist, periods, '-o', markersize=3)
                #pyplot.show()

            else:
                print('WARNING: unable to detect sawtooth sequence of minimum length for data\n' +
                      sht_file_name + '\n'
                                      'processing signal number ' +
                      str(signal_number))

        l_border_t, r_border_t, b_border_f, u_border_f = get_borders(freqs)
        #pyplot.xlim(l_border_t, r_border_t)
        #pyplot.ylim(b_border_f - 50, u_border_f + 50)
        pyplot.legend(char.mod('%d', sawtooth_signals_numbers))
        pyplot.xlabel('Time to fault, s')
        pyplot.ylabel('Extremum distance, arb. un.')
        pyplot.title('Periods of ' + sht_file_name + ' data')
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        fig.savefig(OUT_DIR + 'period_sht' + str(sht_number) + '.png')


def process_global_min():
    data_file = open('data.json')
    data = json.load(data_file)

    for sht_data in data['sht_data']:
        sht_number = sht_data[0]

        sht_file_name = 'sht' + str(sht_number) + '.SHT'
        sht_abs_path = SHT_DIR + sht_file_name

        if not os.path.isfile(sht_abs_path):
            print('WARNING: unable process file\n' +
                  sht_abs_path)
            continue

        freqs = []
        sawtooth_signals_numbers = []

        fig = pyplot.figure()
        signal_numbers = sht_data[1]
        for signal_number in signal_numbers:
            sht_reader = pyglobus.util.ShtReader(sht_abs_path)
            signal_name = SIGNAL_NAMES[signal_number]
            # signal_name = sht_reader.get_signal_name(signal_number)
            signal = sht_reader.get_signals(signal_name)[0]

            empirical_indexes = sht_data[2]
            sawtooth_indexes = get_sawtooth_indexes(signal)
            if sawtooth_indexes[0] < empirical_indexes[0] or sawtooth_indexes[1] > empirical_indexes[1]:
                sawtooth_indexes = empirical_indexes

            if sawtooth_indexes[1] - sawtooth_indexes[0] > MINIMUM_SAWTOOTH_LENGTH:
                sawtooth_signals_numbers.append(signal_number)
                freq = calculate_frequency(signal, sawtooth_indexes)
                freqs.append(freq)

                local_min = []
                previ = 0
                for i in range(1, len(freq[1]) - 1):
                    if freq[1][i - 1] > freq[1][i] and freq[1][i] < freq[1][i + 1]:
                        local_min.append(i)
                        previ = i

                pyplot.plot([freq[0][i] for i in local_min],
                [freq[1][i] for i in local_min], '-o', markersize=3)
                #pyplot.show()

            else:
                print('WARNING: unable to detect sawtooth sequence of minimum length for data\n' +
                      sht_file_name + '\n'
                                      'processing signal number ' +
                      str(signal_number))

        l_border_t, r_border_t, b_border_f, u_border_f = get_borders(freqs)
        pyplot.xlim(l_border_t, r_border_t)
        pyplot.ylim(b_border_f - 50, u_border_f + 50)
        pyplot.legend(char.mod('%d', sawtooth_signals_numbers))
        pyplot.xlabel('Time, s')
        pyplot.ylabel('Frequency, Hz')
        pyplot.title('Minimum of ' + sht_file_name + ' data')
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        fig.savefig(OUT_DIR + 'min_sht' + str(sht_number) + '.png')


def process_time():
    data_file = open('data.json')
    data = json.load(data_file)

    dataplot = []
    fig = pyplot.figure()
    for sht_data in [data['sht_data'][i] for i in [0, 1, -1, -2, -4, -5]]:
        sht_number = sht_data[0]

        sht_file_name = 'sht' + str(sht_number) + '.SHT'
        sht_abs_path = SHT_DIR + sht_file_name

        if not os.path.isfile(sht_abs_path):
            print('WARNING: unable process file\n' +
                  sht_abs_path)
            continue

        freqs = []
        sawtooth_signals_numbers = []

        signal_numbers = sht_data[1]

        #for signal_number in signal_numbers:
        signal_number = signal_numbers[0]
        sht_reader = pyglobus.util.ShtReader(sht_abs_path)
        signal_name = SIGNAL_NAMES[signal_number]
        # signal_name = sht_reader.get_signal_name(signal_number)
        signal = sht_reader.get_signals(signal_name)[0]

        empirical_indexes = sht_data[2]
        sawtooth_indexes = get_sawtooth_indexes(signal)
        if sawtooth_indexes[0] < empirical_indexes[0] or sawtooth_indexes[1] > empirical_indexes[1]:
            sawtooth_indexes = empirical_indexes

        if sawtooth_indexes[1] - sawtooth_indexes[0] > MINIMUM_SAWTOOTH_LENGTH:
            sawtooth_signals_numbers.append(signal_number)
            freq = calculate_frequency(signal, sawtooth_indexes)
            freqs.append(freq)

            pmin = 1e7
            argmin = 0
            for i in range(0, len(freq[1])):
                if freq[1][i] < pmin:
                    pmin = freq[1][i]
                    argmin = i

            dataplot.append(freq[0][-1] - freq[0][argmin])
            #pyplot.show()

        else:
            print('WARNING: unable to detect sawtooth sequence of minimum length for data\n' +
                  sht_file_name + '\n'
                                  'processing signal number ' +
                  str(signal_number))

    l_border_t, r_border_t, b_border_f, u_border_f = get_borders(freqs)
    #pyplot.xlim(l_border_t, r_border_t)
    #pyplot.ylim(b_border_f - 50, u_border_f + 50)

    pyplot.plot([i for i in range(len(dataplot))], dataplot, '-o', markersize=3)
    pyplot.xlabel('Number')
    pyplot.ylabel('Time to fault, s')
    pyplot.title('Times')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    fig.savefig(OUT_DIR + 'times_sht2.png')

    dataplot = np.asarray(dataplot)
    print(f'mean: {np.mean(dataplot)} \ndev: {np.std(dataplot)}')


if __name__ == '__main__':
    #process_periods()
    #process_global_min()
    process_time()