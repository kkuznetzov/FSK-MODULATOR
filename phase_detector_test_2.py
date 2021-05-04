#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Ьфн 2 18:36:53 2021

@author: kkn
"""

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import pdb
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Частота дискретизации файла
signal_samplerate = 40000

# Частота сигнала
signal_frequency = 4000

# Длительность сигнала секунд
signal_length_second = 0.005

# Число отсчётов дискретизации на период сигнала
signal_period_sample_count = signal_samplerate / signal_frequency

# Число отсчётов на период опорного сигнала, больше сигнала
reference_more_samples = 10
reference_signal_period_sample_count = reference_more_samples * signal_period_sample_count

# Число отсчётов дискретизации для длительности сигнала
signal_length_sample_count = signal_samplerate * signal_length_second

# Фаза сигнала и фаза опорного сигнала
signal_phase = 10

# Фаза сигналов в отсчётах дискретизации и разность фаз в отсчётах дискретизации
signal_phase_sample    = signal_phase * signal_period_sample_count / 360

# Вывод параметров
print("Частота дискретизации =", signal_samplerate)
print("Частота сигнала =", signal_frequency)
print("Число отсчётов дискретизации на период сигнала =", signal_period_sample_count)
print("Длительность сигнала секунд = ", signal_length_second)
print("Число отсчётов дискретизации на длину сигнала =", signal_length_sample_count)
print("Фаза сигнала, градусы = ", signal_phase, " отсчёты = ", signal_phase_sample)

# Формируем отсчёты синусов сигнала на всю длину
signal_sin_arange           = np.arange(signal_length_sample_count)
signal_sin_value            = np.sin(2 * np.pi * signal_sin_arange / signal_period_sample_count + signal_phase * 2 * np.pi / 360)

# Формируем отсчёты опорного сигнала на один период
reference_signal_sin_arange = np.arange(reference_signal_period_sample_count)
reference_signal_sin_value_period  = np.sin(2 * np.pi * reference_signal_sin_arange / reference_signal_period_sample_count)
reference_signal_sin_value = np.linspace(0, 0, int(signal_length_sample_count))

# Фильтр для результата перемножения
filter_result = np.linspace(0, 0, int(signal_length_sample_count))
filter_delay_values = np.linspace(0, 0, int(11))
filter_counter = 0
filter_fir_coefficients = np.linspace(0, 0, int(11))
filter_fir_coefficients[0] = 0.0125
filter_fir_coefficients[1] = 0.0866
filter_fir_coefficients[2] = 0.0989
filter_fir_coefficients[3] = 0.139
filter_fir_coefficients[4] = 0.1638
filter_fir_coefficients[5] = 0.1732
filter_fir_coefficients[6] = 0.1638
filter_fir_coefficients[7] = 0.139
filter_fir_coefficients[8] = 0.0989
filter_fir_coefficients[9] = 0.0866
filter_fir_coefficients[10] = 0.0125

# Перемножение сигнала и опорного сигнала
reference_signal_phase_counter = 0
reference_signal_phase_period_counter = 0
multiplication_result = np.linspace(0, 0, int(signal_length_sample_count))
for i in range(int(signal_length_sample_count)):
    multiplication_result[i] = signal_sin_value[i] * reference_signal_sin_value_period[reference_signal_phase_counter]
    reference_signal_sin_value[i] = reference_signal_sin_value_period[reference_signal_phase_counter]

    # Значения линии задержки фильтра
    for j in reversed(range(1, len(filter_delay_values))):
        filter_delay_values[j] = filter_delay_values[j - 1]
    filter_delay_values[0] = multiplication_result[i]

    # Выход фильтра
    for j in range(len(filter_fir_coefficients)):
        filter_result[i] += filter_delay_values[j] * filter_fir_coefficients[j]

    # Счётчик фазы опорного сигнала
    reference_signal_phase_counter += reference_more_samples
    if reference_signal_phase_counter >= reference_signal_period_sample_count:
        reference_signal_phase_counter = int(reference_signal_phase_counter - reference_signal_period_sample_count)
    
    # Каждый полный период входного сигнала коррекция счётчика фазы
    reference_signal_phase_period_counter += 1
    if reference_signal_phase_period_counter >= signal_period_sample_count:
        reference_signal_phase_period_counter = 0
        if filter_result[i] >= 0:
            reference_signal_phase_counter += int(1 + filter_result[i] * 30)
            if reference_signal_phase_counter >= reference_signal_period_sample_count:
                reference_signal_phase_counter = 0
        if filter_result[i] <= -0:
            reference_signal_phase_counter -= int(1 + filter_result[i] * 30)
            if reference_signal_phase_counter < 0:
                reference_signal_phase_counter = int(reference_signal_period_sample_count - 1)
     
t = np.linspace(0, int(signal_length_sample_count), int(signal_length_sample_count))
plt.subplot(3,1,2)
plt.plot(t, signal_sin_value, 'b', t, reference_signal_sin_value, 'g', t, multiplication_result, 'r', t, filter_result, 'm')
plt.title('Signal/Reference signal')
plt.ylabel('Value')
plt.xlabel('Sample')
plt.show()
