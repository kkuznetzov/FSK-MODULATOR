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
wav_file_in_name = 'bfsk_out.wav'

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

# Длина преамбулы бит
preambula_size = 220

# Длина стартового символа бит
start_symbol_size = 5

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
print("Длина преамбулы бит =", preambula_size)
print("Длина стартового символа бит =", start_symbol_size)

# Формируем отсчёты косинуса двух частот
cos_samples_freq_0 = np.arange(sample_cnt_freq_0)
cos_samples_freq_1 = np.arange(sample_cnt_freq_1)
cos_signal_freq_0 = np.sin(2 * np.pi * cos_samples_freq_0 * frequency_0_period_per_bit / sample_cnt_freq_0)
cos_signal_freq_1 = np.sin(2 * np.pi * cos_samples_freq_1 * frequency_1_period_per_bit / sample_cnt_freq_1)

# Формруем данные с результатом перемножения сигнала на опорные частоты, пока всё 0
multiplication_result_freq_0 = np.linspace(0, 0, int(input_signal_length))
multiplication_result_freq_1 = np.linspace(0, 0, int(input_signal_length))

# Масштабируем входной сигнал, что бы максимум был 1 или -1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(input_signal_data[i]) > scale_value:
       scale_value = abs(input_signal_data[i])
input_signal_data = input_signal_data/scale_value

# Обнаружение сигнала и его фазы
# Перемножаем входной сигнал на опорные сигналы
# Интегрируем и сравниваем с предыдущим значением
# Цель получить максимальное значение интегрирования
integration_result_current_0 = 0  # Текущий результат интегрирования
integration_result_current_1 = 0  # Текущий результат интегрирования
multiplication_result_0 = 0       # Результат перемножение входа на опорную частоту
multiplication_result_1 = 0       # Результат перемножение входа на опорную частоту
preambula_phase_0 = 50             # Фаза опорного  сигнала
preambula_phase_1 = 51             # Фаза опорного  сигнала
preambula_phase_counter = 0     # Счётчик периодов интегрирования, что бы по его истечению сравнивать результат интегрирования с предыдущим
preambula_integrator_0 = np.linspace(0, 0, int(preambula_size))
preambula_integrator_1 = np.linspace(0, 0, int(preambula_size))
integrator_counter = 0
phase_diff = 0
for i in range(int(sample_cnt_freq_0 * preambula_size)):
#for i in range(int(input_signal_length)):
    multiplication_result_0 = cos_signal_freq_0[preambula_phase_0] * input_signal_data[i] # Перемножаем вход на опорный сигнал
    integration_result_current_0 += multiplication_result_0 # Интегрируем - суммируем
    multiplication_result_1 = cos_signal_freq_0[preambula_phase_1] * input_signal_data[i] # Перемножаем вход на опорный сигнал
    integration_result_current_1 += multiplication_result_1 # Интегрируем - суммируем

    # Счётчик фазы
    preambula_phase_0 += 1
    if preambula_phase_0 >= sample_cnt_freq_0:
        preambula_phase_0 = 0
    preambula_phase_1 += 1
    if preambula_phase_1 >= sample_cnt_freq_0:
        preambula_phase_1 = 0

    # Счётчик периодов интегрирования
    preambula_phase_counter += 1
    if preambula_phase_counter >= sample_cnt_freq_0:
        phase_diff = phase_diff/5 + integration_result_current_1 - integration_result_current_0
        preambula_phase_0 += round(phase_diff/5)
        preambula_phase_1 = preambula_phase_0 + 1
        if preambula_phase_0 >= sample_cnt_freq_0:
            preambula_phase_0 = preambula_phase_0 - round(sample_cnt_freq_0)
        if preambula_phase_1 >= sample_cnt_freq_0:
            preambula_phase_1 = preambula_phase_1 - round(sample_cnt_freq_0)
        print("1 ", preambula_phase_counter, preambula_phase_0, integration_result_current_0)
        print("2 ", preambula_phase_counter, preambula_phase_1, integration_result_current_1)
        
        '''
        if integration_result_current_1 >= integration_result_current_0:
            
            preambula_phase_0 += round(phase_diff/5)
            #preambula_phase_1 += 1 + round(phase_diff)
            preambula_phase_1 = preambula_phase_0 + 1
            if preambula_phase_0 >= sample_cnt_freq_0:
                preambula_phase_0 = preambula_phase_0 - round(sample_cnt_freq_0)
            if preambula_phase_1 >= sample_cnt_freq_0:
                preambula_phase_1 = preambula_phase_1 - round(sample_cnt_freq_0)
            print("1+ ", preambula_phase_counter, preambula_phase_0, integration_result_current_0)
            print("2+ ", preambula_phase_counter, preambula_phase_1, integration_result_current_1)
        if integration_result_current_1 < integration_result_current_0:
            phase_diff = phase_diff/10 + integration_result_current_0 - integration_result_current_1
            preambula_phase_0 -= round(phase_diff/5)
            #preambula_phase_1 -= 1 + round(phase_diff)
            preambula_phase_1 = preambula_phase_0 + 1
            if preambula_phase_0 < 0:
                preambula_phase_0 = round(sample_cnt_freq_0) + preambula_phase_0
            if preambula_phase_1 < 0:
                preambula_phase_1 = round(sample_cnt_freq_0) + preambula_phase_1
            print("1- ", preambula_phase_counter, preambula_phase_0, integration_result_current_0)
            print("2- ", preambula_phase_counter, preambula_phase_1, integration_result_current_1)
'''            
        preambula_phase_counter = 0
        preambula_integrator_0[integrator_counter] = integration_result_current_0
        preambula_integrator_1[integrator_counter] = integration_result_current_1
        integrator_counter += 1  
        
        integration_result_current_0 = 0
        integration_result_current_1 = 0
        
print(integration_result_current_0)
print(integration_result_current_1)

t = np.linspace(1, int(preambula_size), int(preambula_size))
plt.subplot(3,1,1)
plt.plot(t, preambula_integrator_0, preambula_integrator_1)
plt.title('PD signal')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.show()

# Перемножаем входной сигнал на опорные сигналы
phase_cnt_freq_0 = start_phase # Счётчик фазы косинуса частоты 0
phase_cnt_freq_1 = start_phase # Счётчик фазы косинуса частоты 1
for i in range(int(input_signal_length)):
    multiplication_result_freq_0[i] = cos_signal_freq_0[phase_cnt_freq_0] * input_signal_data[i]
    multiplication_result_freq_1[i] = cos_signal_freq_1[phase_cnt_freq_1] * input_signal_data[i]

    # Счётчики фазы
    phase_cnt_freq_0 += 1
    if(phase_cnt_freq_0 >= sample_cnt_freq_0):
        phase_cnt_freq_0 = 0
    phase_cnt_freq_1 += 1
    if(phase_cnt_freq_1 >= sample_cnt_freq_1):
        phase_cnt_freq_1 = 0

# Значения получаемые в результате интегрирования
output_integrator_freq_0 = np.linspace(0, 0, int(signal_length_freq_0))
output_integrator_freq_1 = np.linspace(0, 0, int(signal_length_freq_1))

# Интегрируем отсчёты
mean_value_freq_0 = 0
mean_value_freq_1 = 0
integrator_cnt_freq_0 = 0
integrator_cnt_freq_1 = 0
output_cnt_freq_0 = 0
output_cnt_freq_1 = 0
for i in range(int(input_signal_length)):
    mean_value_freq_0 += multiplication_result_freq_0[i] * sample_time_freq_0
    mean_value_freq_1 += multiplication_result_freq_1[i] * sample_time_freq_1

    # Счётчики отсчётов для периода интегрирования
    integrator_cnt_freq_0 += 1
    if(integrator_cnt_freq_0 >= sample_cnt_freq_0):
        integrator_cnt_freq_0 = 0
        output_integrator_freq_0[output_cnt_freq_0] = mean_value_freq_0 / sample_cnt_freq_0
        mean_value_freq_0 = 0
        output_cnt_freq_0 += 1
    integrator_cnt_freq_1 += 1
    if(integrator_cnt_freq_1 >= sample_cnt_freq_1):
        integrator_cnt_freq_1 = 0
        output_integrator_freq_1[output_cnt_freq_1] = mean_value_freq_1 / sample_cnt_freq_1
        mean_value_freq_1 = 0
        output_cnt_freq_1 += 1
    
# Значения бит
output_stream_bits = np.linspace(0, 0, int(signal_length_freq_0))

# Превращаем результат интегрирования в биты        
# Просто сравниваем значение, какое больше тот и бит
for i in range(int(signal_length_freq_0)):
    if output_integrator_freq_0[i] >= output_integrator_freq_1[i]:
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
