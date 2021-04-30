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

# Начальная фаза приёма
start_phase = 0

# Имя входного wav файла
wav_file_in_name = 'qfsk_out_and_noise.wav'

# Имя выходного файла с данными
data_file_out_name = 'data_rcv.txt'

# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_in_name)
input_signal_length = input_signal_data.shape[0]

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
sample_cnt_freq_00 = (input_signal_samplerate * frequency_00_period_per_bit) / frequency_00_value
sample_cnt_freq_01 = (input_signal_samplerate * frequency_01_period_per_bit) / frequency_01_value
sample_cnt_freq_10 = (input_signal_samplerate * frequency_10_period_per_bit) / frequency_10_value
sample_cnt_freq_11 = (input_signal_samplerate * frequency_11_period_per_bit) / frequency_11_value

# Период семплирования
sample_time_freq_00 = 1 / input_signal_samplerate
sample_time_freq_01 = 1 / input_signal_samplerate
sample_time_freq_10 = 1 / input_signal_samplerate
sample_time_freq_11 = 1 / input_signal_samplerate

# Длительность посылки пар бит
signal_length_freq_00 = input_signal_length * frequency_00_period_per_bit / (sample_cnt_freq_00 * frequency_00_period_per_bit)
signal_length_freq_01 = input_signal_length * frequency_01_period_per_bit / (sample_cnt_freq_01 * frequency_01_period_per_bit)
signal_length_freq_10 = input_signal_length * frequency_10_period_per_bit / (sample_cnt_freq_10 * frequency_10_period_per_bit)
signal_length_freq_11 = input_signal_length * frequency_11_period_per_bit / (sample_cnt_freq_11 * frequency_11_period_per_bit)

# Вывод параметров
print("Частота дискретизации wav =", input_signal_samplerate)
print("Число отсчётов =", input_signal_length)
print("Частота 00 =", frequency_00_value)
print("Частота 01 =", frequency_01_value)
print("Частота 10 =", frequency_10_value)
print("Частота 11 =", frequency_11_value)
print("Число периодов на бит для частоты 00 =", frequency_00_period_per_bit);
print("Число периодов на бит для частоты 01 =", frequency_01_period_per_bit);
print("Число периодов на бит для частоты 10 =", frequency_10_period_per_bit);
print("Число периодов на бит для частоты 11 =", frequency_11_period_per_bit);
print("Число отсчётов дикретизации для бита 00 =", sample_cnt_freq_00)
print("Число отсчётов дикретизации для бита 01 =", sample_cnt_freq_01)
print("Число отсчётов дикретизации для бита 10 =", sample_cnt_freq_10)
print("Число отсчётов дикретизации для бита 11 =", sample_cnt_freq_11)
print("Длительность входных данных секунд =", input_signal_length/input_signal_samplerate)
print("Длина посылки пар бит для частоты 00 =", signal_length_freq_00)
print("Длина посылки пар бит для частоты 01 =", signal_length_freq_01)
print("Длина посылки пар бит для частоты 10 =", signal_length_freq_10)
print("Длина посылки пар бит для частоты 11 =", signal_length_freq_11)

# Формируем отсчёты косинуса четырёх частот
cos_samples_freq_00 = np.arange(sample_cnt_freq_00)
cos_samples_freq_01 = np.arange(sample_cnt_freq_01)
cos_samples_freq_10 = np.arange(sample_cnt_freq_10)
cos_samples_freq_11 = np.arange(sample_cnt_freq_11)
cos_signal_freq_00 = np.sin(2 * np.pi * cos_samples_freq_00 * frequency_00_period_per_bit / sample_cnt_freq_00)
cos_signal_freq_01 = np.sin(2 * np.pi * cos_samples_freq_01 * frequency_01_period_per_bit / sample_cnt_freq_01)
cos_signal_freq_10 = np.sin(2 * np.pi * cos_samples_freq_10 * frequency_10_period_per_bit / sample_cnt_freq_10)
cos_signal_freq_11 = np.sin(2 * np.pi * cos_samples_freq_11 * frequency_11_period_per_bit / sample_cnt_freq_11)

# Формруем данные с результатом перемножения сигнала на опорные частоты, пока всё 0
multiplication_result_freq_00 = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_01 = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_10 = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_11 = np.linspace(0, 0, int(input_signal_length))

# Масштабируем входной сигнал, что бы максимум был 1 или -1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(input_signal_data[i]) > scale_value:
       scale_value = abs(input_signal_data[i])
input_signal_data = input_signal_data/scale_value

# Перемножаем входной сигнал на опорные сигналы
phase_cnt_freq_00 = start_phase # Счётчик фазы косинуса частоты 00
phase_cnt_freq_01 = start_phase # Счётчик фазы косинуса частоты 01
phase_cnt_freq_10 = start_phase # Счётчик фазы косинуса частоты 10
phase_cnt_freq_11 = start_phase # Счётчик фазы косинуса частоты 11
for i in range(int(input_signal_length)):
    multiplication_result_freq_00[i] = cos_signal_freq_00[phase_cnt_freq_00] * input_signal_data[i]
    multiplication_result_freq_01[i] = cos_signal_freq_01[phase_cnt_freq_01] * input_signal_data[i]
    multiplication_result_freq_10[i] = cos_signal_freq_10[phase_cnt_freq_10] * input_signal_data[i]
    multiplication_result_freq_11[i] = cos_signal_freq_11[phase_cnt_freq_11] * input_signal_data[i]

    # Сётчики фазы
    phase_cnt_freq_00 += 1
    if(phase_cnt_freq_00 >= sample_cnt_freq_00):
        phase_cnt_freq_00 = 0
    phase_cnt_freq_01 += 1
    if(phase_cnt_freq_01 >= sample_cnt_freq_01):
        phase_cnt_freq_01 = 0
    phase_cnt_freq_10 += 1
    if(phase_cnt_freq_10 >= sample_cnt_freq_10):
        phase_cnt_freq_10 = 0
    phase_cnt_freq_11 += 1
    if(phase_cnt_freq_11 >= sample_cnt_freq_11):
        phase_cnt_freq_11 = 0

# Значения получаемые в результате интегрирования
output_integrator_freq_00 = np.linspace(0, 0, int(signal_length_freq_00))
output_integrator_freq_01 = np.linspace(0, 0, int(signal_length_freq_01))
output_integrator_freq_10 = np.linspace(0, 0, int(signal_length_freq_10))
output_integrator_freq_11 = np.linspace(0, 0, int(signal_length_freq_11))

# Интегрируем отсчёты
mean_value_freq_00 = 0
mean_value_freq_01 = 0
mean_value_freq_10 = 0
mean_value_freq_11 = 0
integrator_cnt_freq_00 = 0
integrator_cnt_freq_01 = 0
integrator_cnt_freq_10 = 0
integrator_cnt_freq_11 = 0
output_cnt_freq_00 = 0
output_cnt_freq_01 = 0
output_cnt_freq_10 = 0
output_cnt_freq_11 = 0
for i in range(int(input_signal_length)):
    mean_value_freq_00 += multiplication_result_freq_00[i] * sample_time_freq_00
    mean_value_freq_01 += multiplication_result_freq_01[i] * sample_time_freq_01
    mean_value_freq_10 += multiplication_result_freq_10[i] * sample_time_freq_10
    mean_value_freq_11 += multiplication_result_freq_11[i] * sample_time_freq_11

    # Счётчики отсчётов для периода интегрирования
    integrator_cnt_freq_00 += 1
    if(integrator_cnt_freq_00 >= sample_cnt_freq_00):
        integrator_cnt_freq_00 = 0
        output_integrator_freq_00[output_cnt_freq_00] = mean_value_freq_00# / sample_cnt_freq_00
        mean_value_freq_00 = 0
        output_cnt_freq_00 += 1
        
    integrator_cnt_freq_01 += 1
    if(integrator_cnt_freq_01 >= sample_cnt_freq_01):
        integrator_cnt_freq_01 = 0
        output_integrator_freq_01[output_cnt_freq_01] = mean_value_freq_01# / sample_cnt_freq_01
        mean_value_freq_01 = 0
        output_cnt_freq_01 += 1
        
    integrator_cnt_freq_10 += 1
    if(integrator_cnt_freq_10 >= sample_cnt_freq_10):
        integrator_cnt_freq_10 = 0
        output_integrator_freq_10[output_cnt_freq_10] = mean_value_freq_10# / sample_cnt_freq_10
        mean_value_freq_10 = 0
        output_cnt_freq_10 += 1
        
    integrator_cnt_freq_11 += 1
    if(integrator_cnt_freq_11 >= sample_cnt_freq_11):
        integrator_cnt_freq_11 = 0
        output_integrator_freq_11[output_cnt_freq_11] = mean_value_freq_11# / sample_cnt_freq_11
        mean_value_freq_11 = 0
        output_cnt_freq_11 += 1
    
# Значения бит
output_stream_bits = np.linspace(0, 0, int(signal_length_freq_00 * 2))

# Превращаем результат интегрирования в биты        
# Просто сравниваем значение, какое больше та и пара бит
for i in range(int(signal_length_freq_00)):
    output_stream_bits[i* 2] = 0
    output_stream_bits[i* 2 + 1] = 0
    if (output_integrator_freq_00[i] >= output_integrator_freq_01[i]) and (output_integrator_freq_00[i] >= output_integrator_freq_10[i]) and (output_integrator_freq_00[i] >= output_integrator_freq_11[i]):
           output_stream_bits[i* 2] = 0
           output_stream_bits[i* 2 + 1] = 0
    if (output_integrator_freq_01[i] >= output_integrator_freq_00[i]) and (output_integrator_freq_01[i] >= output_integrator_freq_10[i]) and (output_integrator_freq_01[i] >= output_integrator_freq_11[i]):
           output_stream_bits[i* 2] = 0
           output_stream_bits[i* 2 + 1] = 1
    if (output_integrator_freq_10[i] >= output_integrator_freq_00[i]) and (output_integrator_freq_10[i] >= output_integrator_freq_01[i]) and (output_integrator_freq_10[i] >= output_integrator_freq_11[i]):
           output_stream_bits[i* 2] = 1
           output_stream_bits[i* 2 + 1] = 0
    if (output_integrator_freq_11[i] >= output_integrator_freq_00[i]) and (output_integrator_freq_11[i] >= output_integrator_freq_01[i]) and (output_integrator_freq_11[i] >= output_integrator_freq_10[i]):
           output_stream_bits[i* 2] = 1
           output_stream_bits[i* 2 + 1] = 1

# В строку
output_stream_bits_int8 = np.int8(output_stream_bits)
output_stream_bits_str = ''
for i in range(int(signal_length_freq_00 * 2)):
    output_stream_bits_str += ' ' + str(output_stream_bits_int8[i])
    
# Записываем файл с данными
file = open(data_file_out_name, "w")
file.write(output_stream_bits_str)
file.close()     






