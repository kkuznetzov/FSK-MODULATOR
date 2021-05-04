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
signal_length_second = 0.0025

# Число отсчётов дискретизации на период сигнала
signal_period_sample_count = signal_samplerate / signal_frequency

# Число отсчётов дискретизации для длительности сигнала
signal_length_sample_count = signal_samplerate * signal_length_second

# Фаза сигнала и фаза опорного сигнала
signal_phase = 0
reference_signal_phase = 60

# Разность фаз сигналов
phase_difference = reference_signal_phase - signal_phase

# Фаза сигналов в отсчётах дискретизации и разность фаз в отсчётах дискретизации
signal_phase_sample    = signal_phase * signal_period_sample_count / 360
reference_signal_phase_sample = reference_signal_phase * signal_period_sample_count / 360
phase_difference_sample = reference_signal_phase_sample - signal_phase_sample

# Вывод параметров
print("Частота дискретизации =", signal_samplerate)
print("Частота сигнала =", signal_frequency)
print("Число отсчётов дискретизации на период сигнала =", signal_period_sample_count)
print("Длительность сигнала секунд = ", signal_length_second)
print("Число отсчётов дискретизации на длину сигнала =", signal_length_sample_count)
print("Фаза сигнала, градусы = ", signal_phase, " отсчёты = ", signal_phase_sample)
print("Фаза опорного сигнала, градусы = ", reference_signal_phase, "отсчёты =", reference_signal_phase_sample)
print("Разность фаз сигналов = ", phase_difference, "отсчёты =", phase_difference_sample)

# Формируем отсчёты синусов сигнала и опорного сигнала
signal_sin_arange           = np.arange(signal_length_sample_count)
reference_signal_sin_arange = np.arange(signal_length_sample_count)
signal_sin_value            = np.sin(2 * np.pi * signal_sin_arange / signal_period_sample_count + signal_phase * 2 * np.pi / 360)
reference_signal_sin_value  = np.sin(2 * np.pi * reference_signal_sin_arange / signal_period_sample_count + reference_signal_phase * 2 * np.pi / 360)

# Перемножение сигнала и опрного сигнала
multiplication_result = np.linspace(0, 0, int(signal_length_sample_count))
# multiplication_result = signal_sin_value * reference_signal_sin_value
for i in range(int(signal_length_sample_count)):
    multiplication_result[i] = signal_sin_value[i] * reference_signal_sin_value[i]

# Фильтр для результата перемножения
filter_result = np.linspace(0, 0, int(signal_length_sample_count))
filter_delay_values = np.linspace(0, 0, int(11))
filter_counter = 0
filter_fir_coefficients = np.linspace(0, 0, int(11))
filter_fir_coefficients[0] = 0.05
filter_fir_coefficients[1] = 0.1
filter_fir_coefficients[2] = 0.176
filter_fir_coefficients[3] = 0.25
filter_fir_coefficients[4] = 0.306
filter_fir_coefficients[5] = 0.326
filter_fir_coefficients[6] = 0.306
filter_fir_coefficients[7] = 0.25
filter_fir_coefficients[8] = 0.176
filter_fir_coefficients[9] = 0.1
filter_fir_coefficients[10] = 0.05
for i in range(int(signal_length_sample_count)):
    # Значения линии задержки фильтра
    for j in reversed(range(1, len(filter_delay_values))):
        filter_delay_values[j] = filter_delay_values[j - 1]
    filter_delay_values[0] = multiplication_result[i]

    # Выход фильтра
    for j in range(len(filter_fir_coefficients)):
        filter_result[i] += filter_delay_values[j] * filter_fir_coefficients[j]
 
    
t = np.linspace(0, int(signal_length_sample_count), int(signal_length_sample_count))
plt.subplot(3,1,2)
plt.plot(t, signal_sin_value, 'b', t, reference_signal_sin_value, 'g', t, multiplication_result, 'r', t, filter_result, 'm')
plt.title('Signal/Reference signal')
plt.ylabel('Value')
plt.xlabel('Sample')
plt.show()
