#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:54:46 2021

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
start_phase = 179

# Имя входного wav файла
wav_file_in_name = 'bfsk_out_and_noise_and_low_freq.wav'

# Имя выходного файла с данными
data_file_out_name = 'data_rcv.txt'

# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_in_name)
input_signal_length = input_signal_data.shape[0]

# Частота 0, параметры. Значение частоты и число периодов на бит
frequency_0_value = 4000
frequency_0_period_per_bit = 20

# Частота 1, параметры. Значение частоты и число периодов на бит
frequency_1_value = 3000
frequency_1_period_per_bit = 15

# Число отсчётов дискретизации для частоты 0 и частоты 1
sample_cnt_freq_0 = (input_signal_samplerate * frequency_0_period_per_bit) / frequency_0_value
sample_cnt_freq_1 = (input_signal_samplerate * frequency_1_period_per_bit) / frequency_1_value

# Период семплирования
sample_time_freq_0 = 1 / input_signal_samplerate
sample_time_freq_1 = 1 / input_signal_samplerate

# Длительность посылки бит
signal_length_freq_0 = input_signal_length * frequency_0_period_per_bit / (sample_cnt_freq_0 * frequency_0_period_per_bit)
signal_length_freq_1 = input_signal_length * frequency_1_period_per_bit / (sample_cnt_freq_1 * frequency_1_period_per_bit)

# Вывод параметров
print("Частота дискретизации wav =", input_signal_samplerate)
print("Число отсчётов =", input_signal_length)
print("Частота 0 =", frequency_0_value)
print("Частота 1 =", frequency_1_value)
print("Число периодов на бит для частоты 0 =", frequency_0_period_per_bit);
print("Число периодов на бит для частоты 1 =", frequency_1_period_per_bit);
print("Число отсчётов дикретизации для бита 0 =", sample_cnt_freq_0)
print("Число отсчётов дикретизации для бита 1 =", sample_cnt_freq_1)
print("Длительность входных данных секунд =", input_signal_length/input_signal_samplerate)
print("Длина посылки бит для частоты 0 =", signal_length_freq_0)
print("Длина посылки бит для частоты 1 =", signal_length_freq_1)

# Формируем отсчёты косинуса двух частот
cos_samples_freq_0 = np.arange(sample_cnt_freq_0)
cos_samples_freq_1 = np.arange(sample_cnt_freq_1)
cos_signal_freq_0 = np.sin(2 * np.pi * cos_samples_freq_0 * frequency_0_period_per_bit / sample_cnt_freq_0)
cos_signal_freq_1 = np.sin(2 * np.pi * cos_samples_freq_1 * frequency_1_period_per_bit / sample_cnt_freq_1)

# Формруем данные с результатом перемножения сигнала на опорные частоты, пока всё 0
# Результаты синфазный и квадратурный
multiplication_result_freq_0_I = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_0_Q = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_1_I = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_1_Q = np.linspace(0, 0, int(input_signal_length))

# Масштабируем входной сигнал, что бы максимум был 1 или -1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(input_signal_data[i]) > scale_value:
       scale_value = abs(input_signal_data[i])
input_signal_data = input_signal_data/scale_value

# Перемножаем входной сигнал на опорные сигналы
phase_cnt_freq_0_i = start_phase # Счётчик фазы косинуса частоты 0, in-phase
phase_cnt_freq_0_q = int(start_phase + sample_cnt_freq_0/4) # Счётчик фазы косинуса частоты 0, quadrature
if(phase_cnt_freq_0_q >= sample_cnt_freq_0):
    phase_cnt_freq_0_q = int(phase_cnt_freq_0_q - sample_cnt_freq_0)
phase_cnt_freq_1_i = start_phase # Счётчик фазы косинуса частоты 1, in-phase
phase_cnt_freq_1_q = int(start_phase + sample_cnt_freq_1/4) # Счётчик фазы косинуса частоты 1, quadrature
if(phase_cnt_freq_1_q >= sample_cnt_freq_1):
    phase_cnt_freq_1_q = int(phase_cnt_freq_1_q - sample_cnt_freq_1)
for i in range(int(input_signal_length)):
    multiplication_result_freq_0_I[i] = cos_signal_freq_0[phase_cnt_freq_0_i] * input_signal_data[i]
    multiplication_result_freq_0_Q[i] = cos_signal_freq_0[phase_cnt_freq_0_q] * input_signal_data[i]
    multiplication_result_freq_1_I[i] = cos_signal_freq_1[phase_cnt_freq_1_i] * input_signal_data[i]
    multiplication_result_freq_1_Q[i] = cos_signal_freq_1[phase_cnt_freq_1_q] * input_signal_data[i]

    # Счётчики фазы
    phase_cnt_freq_0_i += 1
    if(phase_cnt_freq_0_i >= sample_cnt_freq_0):
        phase_cnt_freq_0_i = 0
    phase_cnt_freq_0_q += 1
    if(phase_cnt_freq_0_q >= sample_cnt_freq_0):
        phase_cnt_freq_0_q = 0
    phase_cnt_freq_1_i += 1
    if(phase_cnt_freq_1_i >= sample_cnt_freq_1):
        phase_cnt_freq_1_i = 0
    phase_cnt_freq_1_q += 1
    if(phase_cnt_freq_1_q >= sample_cnt_freq_1):
        phase_cnt_freq_1_q= 0

# Значения получаемые в результате интегрирования
output_integrator_freq_0_i = np.linspace(0, 0, int(signal_length_freq_0))
output_integrator_freq_0_q = np.linspace(0, 0, int(signal_length_freq_0))
output_integrator_freq_1_i = np.linspace(0, 0, int(signal_length_freq_1))
output_integrator_freq_1_q = np.linspace(0, 0, int(signal_length_freq_1))

# Сумма квадратов синфазной и квадратурной составляющих
output_integrator_freq_0_sum_squares = np.linspace(0, 0, int(signal_length_freq_0))
output_integrator_freq_1_sum_squares = np.linspace(0, 0, int(signal_length_freq_1))

# Интегрируем отсчёты
mean_value_freq_0_i = 0
mean_value_freq_0_q = 0
mean_value_freq_1_i = 0
mean_value_freq_1_q = 0
integrator_cnt_freq_0 = 0
integrator_cnt_freq_1 = 0
output_cnt_freq_0 = 0
output_cnt_freq_1 = 0
for i in range(int(input_signal_length)):
    # Интегрируем
    mean_value_freq_0_i += multiplication_result_freq_0_I[i]# * sample_time_freq_0
    mean_value_freq_0_q += multiplication_result_freq_0_Q[i]# * sample_time_freq_0    
    mean_value_freq_1_i += multiplication_result_freq_1_I[i]# * sample_time_freq_1
    mean_value_freq_1_q += multiplication_result_freq_1_Q[i]# * sample_time_freq_1

    # Счётчики отсчётов для периода интегрирования
    integrator_cnt_freq_0 += 1
    if(integrator_cnt_freq_0 >= sample_cnt_freq_0):
        integrator_cnt_freq_0 = 0
        
        # Результат интегрирования
        output_integrator_freq_0_i[output_cnt_freq_0] = mean_value_freq_0_i #/ sample_cnt_freq_0
        output_integrator_freq_0_q[output_cnt_freq_0] = mean_value_freq_0_q #/ sample_cnt_freq_0
        
        # Сумма квадратов синфазной и квадратурной составляющих
        output_integrator_freq_0_sum_squares[output_cnt_freq_0] = output_integrator_freq_0_i[output_cnt_freq_0] ** 2 + output_integrator_freq_0_q[output_cnt_freq_0] ** 2
        
        mean_value_freq_0_i = 0
        mean_value_freq_0_q = 0
        output_cnt_freq_0 += 1
    integrator_cnt_freq_1 += 1
    if(integrator_cnt_freq_1 >= sample_cnt_freq_1):
        integrator_cnt_freq_1 = 0
        
        # Результат интегрирования
        output_integrator_freq_1_i[output_cnt_freq_1] = mean_value_freq_1_i #/ sample_cnt_freq_1
        output_integrator_freq_1_q[output_cnt_freq_1] = mean_value_freq_1_q #/ sample_cnt_freq_1
        
        # Сумма квадратов синфазной и квадратурной составляющих
        output_integrator_freq_1_sum_squares[output_cnt_freq_1] = output_integrator_freq_1_i[output_cnt_freq_1] ** 2 + output_integrator_freq_1_q[output_cnt_freq_1] ** 2
        
        mean_value_freq_1_i = 0
        mean_value_freq_1_q = 0
        output_cnt_freq_1 += 1

# Значения бит
output_stream_bits = np.linspace(0, 0, int(signal_length_freq_0))

# Превращаем результат интегрирования в биты        
# Просто сравниваем значение, какое больше тот и бит
for i in range(int(signal_length_freq_0)):
    if output_integrator_freq_0_sum_squares[i] >= output_integrator_freq_1_sum_squares[i]:
        output_stream_bits[i] = 0
    else:
        output_stream_bits[i] = 1

# В строку
output_stream_bits_int8 = np.int8(output_stream_bits)
output_stream_bits_str = ''
for i in range(int(signal_length_freq_0)):
    output_stream_bits_str += ' ' + str(output_stream_bits_int8[i])
    
# Записываем файл с данными
file = open(data_file_out_name, "w")
file.write(output_stream_bits_str)
file.close()   

t = np.linspace(0, int(signal_length_freq_0), int(signal_length_freq_0))
plt.subplot(3,1,2)
plt.plot(t, output_integrator_freq_0_sum_squares, 'k', t, output_integrator_freq_1_sum_squares, 'b')
plt.title('Integrator')
plt.ylabel('Value')
plt.xlabel('Sample')
plt.show()