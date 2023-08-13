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
import os

# Input wav file
# Имя входного wav файла
wav_file_in_name = 'wav\\bfsk_out_cosine.wav'
wav_file_in_name = os.path.join(os.path.dirname(__file__), wav_file_in_name)
#wav_file_in_name = 'qfsk_out_and_noise_and_low_freq2.wav'

# Output wav file
# Имя выходного wav файла
wav_file_out_name = 'wav\\bfsk_out_and_noise.wav'
wav_file_out_name = os.path.join(os.path.dirname(__file__), wav_file_out_name)

# Required signal-to-noise ratio
# Требуемое соотношение сигнал/шум
traget_signal_noise_ration_db = 45

# Reading input wav file
# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_in_name)
input_signal_length = input_signal_data.shape[0]

# Output file parameters
# Вывод параметров файла
print("Частота дискретизации =", input_signal_samplerate)
print("Число отсчётов =", input_signal_length)
print("Длительность секунд =", input_signal_length/input_signal_samplerate)

# Signal energy
# Мощность сигнала
input_signal_data = input_signal_data/32767
input_signal_data_watts = input_signal_data ** 2

# Average signal energy
# Средняя мощность сигнала
input_signal_data_avg_watts = np.mean(input_signal_data_watts) * 32767
input_signal_data_avg_watts_db = 10 * np.log10(input_signal_data_avg_watts)

# Output of signal parameters
# Вывод параметров сигналов
print("Средняя мощность сигнала, дБ =", input_signal_data_avg_watts_db)

# Noise energy
# Значение шума
noise_avg_db = input_signal_data_avg_watts_db - traget_signal_noise_ration_db
noise_avg_watts = 10 ** (noise_avg_db / 10)

# Output of noise parameters
# Вывод параметров шума
print("Средняя мощность шума, дБ =", noise_avg_db)

# White noise generation
# Генерация белого шума
mean_noise = 0
noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(input_signal_data_watts))

# Add noise to the signal
# Добавим шум к сигналу
output_signal_noise_data = input_signal_data + noise_volts

# Scale samples if they exceed 1 so that there is no overflow in int16
# Масштабируем отсчёты, если превысят 1, что бы не было переполнения в int16
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(output_signal_noise_data[i]) > scale_value:
       scale_value = output_signal_noise_data[i]
output_signal_noise_data = output_signal_noise_data/scale_value

# Write the signal and noise to a file
# Запишем сигнал и шум
output_signal_noise_data *= 32767
output_signal_noise_data_int = np.int16(output_signal_noise_data)
wavfile.write(wav_file_out_name, input_signal_samplerate, output_signal_noise_data_int)
