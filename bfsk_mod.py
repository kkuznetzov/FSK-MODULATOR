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
wav_file_name = 'bfsk_out.wav'

# Частота дискретизации файла wav
samplerate = 44100

# Длина преамбулы бит
preambula_size = 25

# Биты преамбулы
preambula_bits = 0

# Длина стартового символа бит
start_symbol_size = 5

# Биты стартового символа
start_symbol_bits = 1 

# Частота 0, параметры. Значение частоты и число периодов на бит
frequency_0_value = 4000
frequency_0_period_per_bit = 20

# Частота 1, параметры. Значение частоты и число периодов на бит
frequency_1_value = 3000
frequency_1_period_per_bit = 15

# Число отсчётов дискретизации для частоты 0 и частоты 1
sample_cnt_freq_0 = (samplerate * frequency_0_period_per_bit) / frequency_0_value
sample_cnt_freq_1 = (samplerate * frequency_1_period_per_bit) / frequency_1_value

# Вывод параметров
print("Частота дискретизации wav =", samplerate)
print("Частота 0 =", frequency_0_value)
print("Частота 1 =", frequency_1_value)
print("Число периодов на бит для частоты 0 =", frequency_0_period_per_bit);
print("Число периодов на бит для частоты 1 =", frequency_1_period_per_bit);
print("Число отсчётов дикретизации для бита 0 =", sample_cnt_freq_0)
print("Число отсчётов дикретизации для бита 1 =", sample_cnt_freq_1)
print("Длина преамбулы бит =", preambula_size)
print("Длина стартового символа бит =", start_symbol_bits)

# Читаем файл с данными
with open(data_file_name) as fdata:
    data_list = [int(x) for x in next(fdata).split()]

# Ковертируем в int
data_array = np.array(data_list, dtype = np.int32())

# Считаем число бит 0 и 1
bit_0_cnt = 0
bit_1_cnt = 0
for bit in enumerate(data_array):
    if bit[1] > 0:
        bit_1_cnt += 1
    else:
        bit_0_cnt += 1

# Добавим биты преамбулы
if preambula_bits == 0:
    bit_0_cnt += preambula_size
else:
    bit_1_cnt += preambula_size

# Добавим биты стартвого символа
if start_symbol_bits == 0:
    bit_0_cnt += start_symbol_size
else:
    bit_1_cnt += start_symbol_size

# Длительность посылки
signal_sample_length = (bit_0_cnt * sample_cnt_freq_0) + (bit_1_cnt * sample_cnt_freq_1)
print("Длина посылки, бит =", len(data_array))
print("Число отсчётов дикретизации посылки =", signal_sample_length)
print("Длина посылки, секунд =", signal_sample_length/samplerate)

# Формируем отсчёты косинуса двух частот
cos_samples_freq_0 = np.arange(sample_cnt_freq_0)
cos_samples_freq_1 = np.arange(sample_cnt_freq_1)
cos_signal_freq_0 = np.sin(2 * np.pi * cos_samples_freq_0 * frequency_0_period_per_bit / sample_cnt_freq_0)
cos_signal_freq_1 = np.sin(2 * np.pi * cos_samples_freq_1 * frequency_1_period_per_bit / sample_cnt_freq_1)

# Формруем выходные данные, пока всё 0
output_signal = np.linspace(0, 0, int(signal_sample_length))

# Смещение в выходном сигнале на преамбулу и стартовый символ
# Здесь же длина преамбулы и длина стартового символа
output_offset = 0
preambula_length = 0
start_symbol_length = 0
if preambula_bits == 0:
    output_offset += (preambula_size * sample_cnt_freq_0)
    preambula_length = preambula_size * sample_cnt_freq_0
else:
    output_offset += (preambula_size * sample_cnt_freq_1)
    preambula_length = preambula_size * sample_cnt_freq_1
if start_symbol_bits == 0:
    output_offset += (start_symbol_size * sample_cnt_freq_0)
    start_symbol_length = start_symbol_size * sample_cnt_freq_0
else:
    output_offset += (start_symbol_size * sample_cnt_freq_1)
    start_symbol_length = start_symbol_size * sample_cnt_freq_1
output_offset_int = int(output_offset)

# Формируем преамбулу
phase_cnt = 0 # Счётчик фазы косинуса
preambula_bit_cnt = 0 # Счётчик бит
bit_value = preambula_bits # Значение бита
for i in range(0, int(preambula_length)):
    # Используем одну из двух частот в зависимости от значения бита
    if bit_value > 0: # Не ноль
        output_signal[i] = cos_signal_freq_1[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_1):
            phase_cnt = 0
    else:             # Ноль
        output_signal[i] = cos_signal_freq_0[phase_cnt]

        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_0):
            phase_cnt = 0

# Формируем стартовый символ
phase_cnt = 0 # Счётчик фазы косинуса
start_symbol_bit_cnt = 0 # Счётчик бит
bit_value = start_symbol_bits # Значение бита
for i in range(int(preambula_length), int(preambula_length + start_symbol_length)):
    # Используем одну из двух частот в зависимости от значения бита
    if bit_value > 0: # Не ноль
        output_signal[i] = cos_signal_freq_1[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_1):
            phase_cnt = 0
    else:             # Ноль
        output_signal[i] = cos_signal_freq_0[phase_cnt]

        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_0):
            phase_cnt = 0

# Формируем выходной сигнал согласно битам, используя FSK
phase_cnt = 0 # Счётчик фазы косинуса
bit_cnt = 0 # Счётчик бит
bit_value = 0 # Значение бита
for i in range(int(output_offset), int(signal_sample_length)):
    bit_value = data_array[bit_cnt] # Значение бита
    
    # Используем одну из двух частот в зависимости от значения бита
    if bit_value > 0: # Не ноль
        output_signal[i] = cos_signal_freq_1[phase_cnt]
        
        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_1):
            phase_cnt = 0
            bit_cnt += 1        
    else:             # Ноль
        output_signal[i] = cos_signal_freq_0[phase_cnt]

        # Счётчик фазы косинуса
        # Здесь же счётчик бит
        phase_cnt += 1
        if(phase_cnt >= sample_cnt_freq_0):
            phase_cnt = 0
            bit_cnt += 1      
            
# Сохраним в файл
output_signal*= 32767
ountput_signal_int = np.int16(output_signal)
wavfile.write(wav_file_name, samplerate, ountput_signal_int)
