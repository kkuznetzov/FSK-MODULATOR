#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:36:53 2021

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

# Имя файла с данными
data_file_name = 'data.txt'

# Имя выходного wav файла
wav_file_name = 'qfsk_out.wav'

# Частота дискретизации файла wav
samplerate = 44100

# Частота 00, параметры. Значение частоты и число периодов на пару бит
frequency_00_value = 3500
frequency_00_period_per_bit = 35

# Частота 01, параметры. Значение частоты и число периодов на пару бит
frequency_01_value = 2500
frequency_01_period_per_bit = 25

# Частота 10, параметры. Значение частоты и число периодов на пару бит
frequency_10_value = 2000
frequency_10_period_per_bit = 20

# Частота 11, параметры. Значение частоты и число периодов на пару бит
frequency_11_value = 1500
frequency_11_period_per_bit = 15

# Число отсчётов дискретизации для частот 00, 01, 10, 11
sample_cnt_freq_00 = (samplerate * frequency_00_period_per_bit) / frequency_00_value
sample_cnt_freq_01 = (samplerate * frequency_01_period_per_bit) / frequency_01_value
sample_cnt_freq_10 = (samplerate * frequency_10_period_per_bit) / frequency_10_value
sample_cnt_freq_11 = (samplerate * frequency_11_period_per_bit) / frequency_11_value

# Вывод параметров
print("Частота дискретизации wav =", samplerate)
print("Частота 00 =", frequency_00_value)
print("Частота 01 =", frequency_01_value)
print("Частота 10 =", frequency_10_value)
print("Частота 11 =", frequency_11_value)
print("Число периодов на бит для частоты 00 =", frequency_00_period_per_bit);
print("Число периодов на бит для частоты 01 =", frequency_01_period_per_bit);
print("Число периодов на бит для частоты 10 =", frequency_10_period_per_bit);
print("Число периодов на бит для частоты 11 =", frequency_11_period_per_bit);
print("Число отсчётов дикретизации для бит 00 =", sample_cnt_freq_00)
print("Число отсчётов дикретизации для бит 01 =", sample_cnt_freq_01)
print("Число отсчётов дикретизации для бит 10 =", sample_cnt_freq_10)
print("Число отсчётов дикретизации для бит 11 =", sample_cnt_freq_11)

# Читаем файл с данными
with open(data_file_name) as fdata:
    data_list = [int(x) for x in next(fdata).split()]

# Ковертируем в int
data_array = np.array(data_list, dtype = np.int32())

# Конвертируем массив в двумерный 2-D
data_array_2_d = np.reshape(data_array, (-1, 2))

# Считаем число пар бит 00, 01, 10, 11
bit_00_cnt = 0
bit_01_cnt = 0
bit_10_cnt = 0
bit_11_cnt = 0
for bit in enumerate(data_array_2_d):
    if bit[1][0] == 0 and bit[1][1] == 0:
        bit_00_cnt += 1
    if bit[1][0] == 0 and bit[1][1] == 1:
        bit_01_cnt += 1
    if bit[1][0] == 1 and bit[1][1] == 0:
        bit_10_cnt += 1
    if bit[1][0] == 1 and bit[1][1] == 1:
        bit_11_cnt += 1

# Длительность посылки
signal_sample_length = (bit_00_cnt * sample_cnt_freq_00) + (bit_01_cnt * sample_cnt_freq_01) + (bit_10_cnt * sample_cnt_freq_10) + (bit_11_cnt * sample_cnt_freq_11)
print("Длина посылки, бит =", len(data_array))
print("Число отсчётов дикретизации посылки =", signal_sample_length)
print("Длина посылки, секунд =", signal_sample_length/samplerate)

# Формируем отсчёты косинуса двух частот
cos_samples_freq_00 = np.arange(sample_cnt_freq_00)
cos_samples_freq_01 = np.arange(sample_cnt_freq_01)
cos_samples_freq_10 = np.arange(sample_cnt_freq_10)
cos_samples_freq_11 = np.arange(sample_cnt_freq_11)
cos_signal_freq_00 = np.sin(2 * np.pi * cos_samples_freq_00 * frequency_00_period_per_bit / sample_cnt_freq_00)
cos_signal_freq_01 = np.sin(2 * np.pi * cos_samples_freq_01 * frequency_01_period_per_bit / sample_cnt_freq_01)
cos_signal_freq_10 = np.sin(2 * np.pi * cos_samples_freq_10 * frequency_10_period_per_bit / sample_cnt_freq_10)
cos_signal_freq_11 = np.sin(2 * np.pi * cos_samples_freq_11 * frequency_11_period_per_bit / sample_cnt_freq_11)

# Формруем выходные данные, пока всё 0
output_signal = np.linspace(0, 0, int(signal_sample_length))

# Формируем выходной сигнал согласно битам, используя FSK
phase_cnt = 0 # Счётчик фазы косинуса
bit_cnt = 0 # Счётчик бит
bits_value = 0 # Значение бит
for i in range(int(signal_sample_length)):
    bits_value = data_array_2_d[bit_cnt] # Значение пар бит
    
    # Используем одну из двух частот в зависимости от значения пар бит
    if bits_value[0] == 0 and bits_value[1] == 0:
        output_signal[i] = cos_signal_freq_00[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_00):
            phase_cnt = 0
            bit_cnt += 1        
    
    if bits_value[0] == 0 and bits_value[1] == 1:
        output_signal[i] = cos_signal_freq_01[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_01):
            phase_cnt = 0
            bit_cnt += 1              

    if bits_value[0] == 1 and bits_value[1] == 0:
        output_signal[i] = cos_signal_freq_10[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_10):
            phase_cnt = 0
            bit_cnt += 1              

    if bits_value[0] == 1 and bits_value[1] == 1:
        output_signal[i] = cos_signal_freq_11[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_11):
            phase_cnt = 0
            bit_cnt += 1              
           
# Сохраним в файл
output_signal*= 32767
ountput_signal_int = np.int16(output_signal)
wavfile.write(wav_file_name, samplerate, ountput_signal_int)











