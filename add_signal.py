#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 20 18:36:53 2023

@author: kkuznetzov
"""

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import pdb
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Frequency and amplitude of the added signal
# Частота и амплитуда добавляемого сигнала
signal_frequency = 1100
signal_amplitude = 1

# Output wav file
# Имя входного wav файла
wav_file_in_name = 'wav\\bfsk_out_and_noise.wav'

# Output wav file
# Имя выходного wav файла
wav_file_out_name = 'wav\\bfsk_out_and_noise_and_low_freq.wav'

# Reading input wav file
# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_in_name)
input_signal_length = input_signal_data.shape[0]

# Number of samples per signal period
# Число отсчётов на период сигнала
sample_per_period = input_signal_samplerate / signal_frequency

# Output file parameters
# Вывод параметров файла
print("Частота дискретизации =", input_signal_samplerate)
print("Число отсчётов =", input_signal_length)
print("Длительность секунд =", input_signal_length/input_signal_samplerate)
print("Частота добавляемого сигнала", signal_frequency)
print("Число отсчётов на период сигнала =", sample_per_period)

# Calculate an array with sinusoid samples
# Формируем отсчёты синуса
cos_samples = np.arange(sample_per_period)
cos_signal = np.sin(2 * np.pi * cos_samples / sample_per_period) * signal_amplitude

# Scale the samples of the input signal so that they are in the range from -1 to 1
# Масштабируем отсчёты входного сигнала, что бы были в интервале от -1 до 1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(input_signal_data[i]) > scale_value:
        scale_value = input_signal_data[i]
input_signal_data = input_signal_data/scale_value

# Empty array for signal
# Формруем выходные данные, пока всё 0
output_signal = np.linspace(0, 0, int(input_signal_length))

# Phase sin counter
# Счётчик фазы косинуса
phase_cnt = 0

# Adding a sin signal to an input signal
# Добавляем сигнал к входному сигналу
for i in range(int(input_signal_length)):
    output_signal[i] = input_signal_data[i] + cos_signal[phase_cnt]

    # Increment phase sin counter
    # Счётчик фазы синуса
    # Здесь же счётчик бит
    phase_cnt += 1
    if(phase_cnt >= sample_per_period):
        phase_cnt = 0

# Scale the samples of the output signal so that they are in the range from -1 to 1
# Масштабируем отсчёты выходного сигнала, что бы были в интервале от -1 до 1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(output_signal[i]) > scale_value:
        scale_value = output_signal[i]
output_signal = output_signal/scale_value

# Write the new signal to a file
# Сохраним в файл
output_signal*= 32767
ountput_signal_int = np.int16(output_signal)
wavfile.write(wav_file_out_name, input_signal_samplerate, ountput_signal_int)
