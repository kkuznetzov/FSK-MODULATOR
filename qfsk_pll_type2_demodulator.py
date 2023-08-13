#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 20 18:36:53 2023

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
import os

# Input wav file
# Имя входного wav файла
wav_file_name = 'wav\\qfsk_out.wav'
wav_file_name = os.path.join(os.path.dirname(__file__), wav_file_name)

# Name of the output data file
# Имя выходного файла с данными
data_file_out_name = 'received_data.txt'
data_file_out_name = os.path.join(os.path.dirname(__file__), data_file_out_name)

# Sampling rate of wav file, other values don't work
# Частота дискретизации файла wav
wav_samplerate = 44100

# Скорость передачи данных бит
transmit_bit_rate = 588

# Frequency value for symbol transmission[bit]: 0[00], 1[01], 2[10], 3[11]
# Значение частоты для передачи символов[бит]: 0[00], 1[01], 2[10], 3[11]
symbol_0_frequency = 882
symbol_1_frequency = 1470
symbol_2_frequency = 1764
symbol_3_frequency = 2940

# Symbol rate, equal to half of the bit rate
# Скорость передачи символов, половина от битовой, так как один символ передаёт два бита
transmit_symbol_rate = transmit_bit_rate // 2

# Carrier frequency duration in symbol durations, transmitted before the preamble
# Used for PLL operation and signal detection
# Размер несущей в символах
carrier_symbol_size = 24

# Frequency value for preamble transmission
# Значение частоты преамбулы, задано значением бита
carrier_symbol_value = 0

# Preamble (alternating frequency) duration, in symbol durations
# Used for symbol synchronization
# Размер преамбулы, симвоолов
preamble_symbol_size = 24

# Synchronization word size, bits. Used for byte synchronization
# Размер слова синхронизации, символ
synchronization_word_symbol_size = 8

# Sync word symbol value
# Значение символа слова синхронизации
synchronization_word_symbol_value = 1

# Число отсчётов частоты дискретизации на каждый бит
# Число отсчётов частоты дискретизации на каждый символ
samplerate_period_per_symbol = wav_samplerate / transmit_symbol_rate

# Open wav file for reading
# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_name)
input_signal_length = input_signal_data.shape[0]

# Signal duration, symbols and bytes
# Длительность посылки символов и байт
# Один символ два бита, потому один байт 4 символа
input_signal_symbol_length = input_signal_length / samplerate_period_per_symbol
input_signal_byte_length = input_signal_symbol_length / 4

# Output parameters to the console
# Вывод параметров
print("Частота дискретизации wav =", wav_samplerate)
print("Частота для символа 0, биты 00  =", symbol_0_frequency)
print("Частота для символа 1, биты 01  =", symbol_1_frequency)
print("Частота для символа 2, биты 10  =", symbol_2_frequency)
print("Частота для символа 3, биты 11  =", symbol_3_frequency)
print("Число периодов дискрертизации на символ =", samplerate_period_per_symbol)
print("Число отсчётов входного файла =", input_signal_length)
print("Длительность входных данных секунд =", input_signal_length/input_signal_samplerate)
print("Размер несущей, символов =", carrier_symbol_size)
print("Значение несущей несущей, символ =", carrier_symbol_value)
print("Размер преамбулы, символов =", preamble_symbol_size)
print("Размер слова синхронизации, символов =", synchronization_word_symbol_size)
print("Значение символа слова синхронизации =", synchronization_word_symbol_value)
print("Длина посылки в символах =", input_signal_symbol_length)
print("Длина посылки в байтах =", input_signal_byte_length)

# Calculate an array with sinusoid samples
# Формируем отсчёты синусоид и косинусоид четырёх частот для символов
symbol_sin_samples_index = np.arange(samplerate_period_per_symbol)
symbol_0_sin_samples = np.sin(2 * np.pi * (symbol_0_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_0_cos_samples = np.cos(2 * np.pi * (symbol_0_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_1_sin_samples = np.sin(2 * np.pi * (symbol_1_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_1_cos_samples = np.cos(2 * np.pi * (symbol_1_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_2_sin_samples = np.sin(2 * np.pi * (symbol_2_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_2_cos_samples = np.cos(2 * np.pi * (symbol_2_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_3_sin_samples = np.sin(2 * np.pi * (symbol_3_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_3_cos_samples = np.cos(2 * np.pi * (symbol_3_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)

# Scale the samples of the input signal so that they are in the range from -1 to 1
# Масштабируем входной сигнал, что бы максимум был 1 или -1
input_signal_maximum_amplitude = max(abs(input_signal_data))
input_signal_data = input_signal_data / input_signal_maximum_amplitude

# Calculation of PI filter coefficients
# Вычисление коэффициентов ПИ регулятора
pll_bandwidth = transmit_bit_rate
pll_samplerate = wav_samplerate
pll_damping_factor = math.sqrt(2) / 2
pll_phase_error_detector_gain = 0.5 # Kp
pll_oscillator_gain = 1 # Ko

# Type 2 filter coefficients
# Коэффициенты фильтра типа 2
pll_type_2_coefficient_h = pll_bandwidth / pll_samplerate
pll_type_2_coefficient_h = pll_type_2_coefficient_h / (pll_damping_factor + 1 / (4 * pll_damping_factor))
pll_type_2_coefficient_k1 = 4 * pll_damping_factor / (pll_phase_error_detector_gain * pll_oscillator_gain)
pll_type_2_coefficient_k1 = pll_type_2_coefficient_k1 * pll_type_2_coefficient_h
pll_type_2_coefficient_k2 = 4 * (pll_type_2_coefficient_h ** 2) / (pll_phase_error_detector_gain * pll_oscillator_gain)

# Type 2 PLL filter output values for each symbol
# Значения выхода фильтра ФАПЧ типа 2 для каждого символа
pll_type_2_symbol_0_loop_value_k1 = 0
pll_type_2_symbol_1_loop_value_k1 = 0
pll_type_2_symbol_2_loop_value_k1 = 0
pll_type_2_symbol_3_loop_value_k1 = 0
pll_type_2_symbol_0_loop_value_k2 = 0
pll_type_2_symbol_1_loop_value_k2 = 0
pll_type_2_symbol_2_loop_value_k2 = 0
pll_type_2_symbol_3_loop_value_k2 = 0
pll_type_2_symbol_0_filter_output = 0
pll_type_2_symbol_1_filter_output = 0
pll_type_2_symbol_2_filter_output = 0
pll_type_2_symbol_3_filter_output = 0

# PLL error value
# Значение ошибки ФАПЧ
pll_type_2_error_filter_output = 0

# The result of multiplying the signal by the reference signals, quadrature and in-phase components
# Результат перемножения сигнала на опорные сигналы, квадратурная и синфазная составляющие
# Квадратурная для ФАПЧ, синфазная для демодуляции
pll_type_2_symbol_0_i_multiplication_value = 0
pll_type_2_symbol_0_q_multiplication_value = 0
pll_type_2_symbol_1_i_multiplication_value = 0
pll_type_2_symbol_1_q_multiplication_value = 0
pll_type_2_symbol_2_i_multiplication_value = 0
pll_type_2_symbol_2_q_multiplication_value = 0
pll_type_2_symbol_3_i_multiplication_value = 0
pll_type_2_symbol_3_q_multiplication_value = 0

# Phase counter for PLL type 2
# Счётчик фазы для ФАПЧ типа 2
pll_type_2_phase_cnt = 0

# Current sample value of the input signal
# Текущий отсчёт входного сигнала
input_sample = 0

# To implement average filters
# Фильтры плавающего среднего для принимаемых бит
# Сами буферы, счётчик и средние значения
symbol_signal_average_buffer_counter = 0
symbol_signal_average_buffer_size = samplerate_period_per_symbol
symbol_0_signal_average_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_1_signal_average_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_2_signal_average_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_3_signal_average_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_0_signal_average = 0
symbol_1_signal_average = 0
symbol_2_signal_average = 0
symbol_3_signal_average = 0
symbol_n_signal_average = 0

# Second buffer
# Второй буфер
symbol_0_signal_average_second_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_1_signal_average_second_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_2_signal_average_second_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))
symbol_3_signal_average_second_buffer = np.linspace(0, 0, int(symbol_signal_average_buffer_size))

# Carrier capture flag
# Флаг захвата несущей
pll_type_2_carrier_lock_flag = 0

# Carrier capture counter and counter threshold
# Счётчик захвата несущей и порог счётчика
pll_type_2_carrier_lock_counter = 0
pll_type_2_carrier_lock_counter_threshold = carrier_symbol_size // 2

# The value of the carrier to determine its capture
# Значение несущей для определения её захвата
pll_type_2_carrier_symbol_value = 0

# Carrier lock amplitude threshold
# Порог уровня для определения захвата несущей
pll_type_2_carrier_lock_threshold = 0.05

# To search for a preamble
# Preamble capture flag, frequency change counter
# Для поиска преамбулы
preamble_lock_flag = 0
symbol_value_change_counter = 0
symbol_value_change_lock_threshold = preamble_symbol_size // 2

# For symbol sync, symbol amplitude, amplitude threshold, symbol change counter
# Для символьной синхронизации
symbol_0_signal_average_previous = 0
symbol_1_signal_average_previous = 0
symbol_2_signal_average_previous = 0
symbol_3_signal_average_previous = 0
symbol_0_signal_rising_flag = 0
symbol_1_signal_rising_flag = 0
symbol_2_signal_rising_flag = 0
symbol_3_signal_rising_flag = 0
symbol_signal_average_threshold = 0.05
symbol_alternation_counter = 0
symbol_alternation_counter_threshold = preamble_symbol_size // 2

# Filter counter value for the middle of a symbol
# Значение счётчика фильтра для середины символа
symbol_signal_average_buffer_counter_lock_value = 0

# Symbol value discrete
# Значение символа дискретное
symbol_digital_value = 0
symbol_digital_value_previous = 0

# To search for the word sync
# Для поиска слова синхронизации
synchronization_word_symbol_counter = 0
synchronization_word_lock_flag = 0

# Byte counter and bit counter
# Счётчик байт и счётчик бит
output_byte_count = 0
output_bit_count = 0

# Output data, bytes
# Выходные данные, байты
byte_value = 0
output_stream_bytes = []

# Debug
# Для отладки
pll_type_2_symbol_0_filter_output_debug = []
pll_type_2_symbol_1_filter_output_debug = []
pll_type_2_symbol_2_filter_output_debug = []
pll_type_2_symbol_3_filter_output_debug = []
pll_type_2_carrier_lock_sample_debug = []
pll_type_2_carrier_lock_value_debug = []
symbol_digital_value_debug_x = []
symbol_digital_value_debug_y = []
symbol_digital_value_debug = []
preamble_lock_sample_debug = []
preamble_lock_value_debug = []
synchro_lock_sample_debug = []
synchro_lock_value_debug = []
byte_digital_sample_x_debug = []
byte_digital_sample_y_debug = []
byte_digital_value_debug = []

# Loop through input samples
# Проход по входным отсчётам
for i in range(int(input_signal_length)):
    # PLL implementation for carrier lock
    # Реализация ФАПЧ для захвата несущей

    # Current sample of the input signal
    # Текущий отсчёт входного сигнала
    input_sample = input_signal_data[i]

    # Current samples of reference signals
    # Текущие отсчёты опорных сигналов
    symbol_0_sin_sample = symbol_0_sin_samples[int(pll_type_2_phase_cnt)]
    symbol_0_cos_sample = symbol_0_cos_samples[int(pll_type_2_phase_cnt)]
    symbol_1_sin_sample = symbol_1_sin_samples[int(pll_type_2_phase_cnt)]
    symbol_1_cos_sample = symbol_1_cos_samples[int(pll_type_2_phase_cnt)]
    symbol_2_sin_sample = symbol_2_sin_samples[int(pll_type_2_phase_cnt)]
    symbol_2_cos_sample = symbol_2_cos_samples[int(pll_type_2_phase_cnt)]
    symbol_3_sin_sample = symbol_3_sin_samples[int(pll_type_2_phase_cnt)]
    symbol_3_cos_sample = symbol_3_cos_samples[int(pll_type_2_phase_cnt)]

    # Multiply the input signal by the reference signals
    # Перемножаем входной сигнал на опорные сигналы
    pll_type_2_symbol_0_i_multiplication_value = input_sample * symbol_0_sin_sample
    pll_type_2_symbol_0_q_multiplication_value = input_sample * symbol_0_cos_sample
    pll_type_2_symbol_1_i_multiplication_value = input_sample * symbol_1_sin_sample
    pll_type_2_symbol_1_q_multiplication_value = input_sample * symbol_1_cos_sample
    pll_type_2_symbol_2_i_multiplication_value = input_sample * symbol_2_sin_sample
    pll_type_2_symbol_2_q_multiplication_value = input_sample * symbol_2_cos_sample
    pll_type_2_symbol_3_i_multiplication_value = input_sample * symbol_3_sin_sample
    pll_type_2_symbol_3_q_multiplication_value = input_sample * symbol_3_cos_sample

    # Calculate the PLL filters values
    # Считаем значения фильтров ФАПЧ
    pll_type_2_symbol_0_loop_value_k1 = pll_type_2_symbol_0_q_multiplication_value * pll_type_2_coefficient_k1
    pll_type_2_symbol_1_loop_value_k1 = pll_type_2_symbol_1_q_multiplication_value * pll_type_2_coefficient_k1
    pll_type_2_symbol_2_loop_value_k1 = pll_type_2_symbol_2_q_multiplication_value * pll_type_2_coefficient_k1
    pll_type_2_symbol_3_loop_value_k1 = pll_type_2_symbol_3_q_multiplication_value * pll_type_2_coefficient_k1
    pll_type_2_symbol_0_filter_output = pll_type_2_symbol_0_loop_value_k1 + pll_type_2_symbol_0_loop_value_k2
    pll_type_2_symbol_1_filter_output = pll_type_2_symbol_1_loop_value_k1 + pll_type_2_symbol_1_loop_value_k2
    pll_type_2_symbol_2_filter_output = pll_type_2_symbol_2_loop_value_k1 + pll_type_2_symbol_2_loop_value_k2
    pll_type_2_symbol_3_filter_output = pll_type_2_symbol_3_loop_value_k1 + pll_type_2_symbol_3_loop_value_k2
    pll_type_2_symbol_0_loop_value_k2 = pll_type_2_symbol_0_q_multiplication_value * pll_type_2_coefficient_k2
    pll_type_2_symbol_1_loop_value_k2 = pll_type_2_symbol_1_q_multiplication_value * pll_type_2_coefficient_k2
    pll_type_2_symbol_2_loop_value_k2 = pll_type_2_symbol_2_q_multiplication_value * pll_type_2_coefficient_k2
    pll_type_2_symbol_3_loop_value_k2 = pll_type_2_symbol_3_q_multiplication_value * pll_type_2_coefficient_k2

    # Для отладки
    # pll_type_2_symbol_0_filter_output_debug.append(pll_type_2_symbol_0_filter_output)
    # pll_type_2_symbol_1_filter_output_debug.append(pll_type_2_symbol_1_filter_output)
    # pll_type_2_symbol_2_filter_output_debug.append(pll_type_2_symbol_2_filter_output)
    # pll_type_2_symbol_3_filter_output_debug.append(pll_type_2_symbol_3_filter_output)

    # Select an error signal depending on the specified preamble symbol
    # Выбор сигнала ошибки в зависимости от заданного символа преамбулы
    if carrier_symbol_value == 0:
        pll_type_2_error_filter_output = pll_type_2_symbol_0_filter_output
    elif carrier_symbol_value == 1:
        pll_type_2_error_filter_output = pll_type_2_symbol_1_filter_output
    elif carrier_symbol_value == 2:
        pll_type_2_error_filter_output = pll_type_2_symbol_2_filter_output
    else:
        pll_type_2_error_filter_output = pll_type_2_symbol_3_filter_output

    # Calculation of the phase counter value for reference signals
    # costas_loop_iq_filter_output - phase error
    # Инкремент счётчика фазы с добавлением ошибки фазы
    pll_type_2_phase_cnt = pll_type_2_phase_cnt + 1 + pll_type_2_error_filter_output
    if pll_type_2_phase_cnt >= samplerate_period_per_symbol:
        pll_type_2_phase_cnt = pll_type_2_phase_cnt - samplerate_period_per_symbol
    if pll_type_2_phase_cnt < 0:
        pll_type_2_phase_cnt = pll_type_2_phase_cnt + samplerate_period_per_symbol

    # Put the result in the floating average buffer
    # Помещаем результаты для синфазной составляющей в буферы плавающего среднего
    symbol_0_signal_average_buffer[symbol_signal_average_buffer_counter] = pll_type_2_symbol_0_i_multiplication_value
    symbol_1_signal_average_buffer[symbol_signal_average_buffer_counter] = pll_type_2_symbol_1_i_multiplication_value
    symbol_2_signal_average_buffer[symbol_signal_average_buffer_counter] = pll_type_2_symbol_2_i_multiplication_value
    symbol_3_signal_average_buffer[symbol_signal_average_buffer_counter] = pll_type_2_symbol_3_i_multiplication_value

    # Save the previous values of the average, to find the preamble
    # Предыдущие значения среднего, для поиска преамбулы
    if (pll_type_2_carrier_lock_flag == 1) and (preamble_lock_flag == 0):
        symbol_0_signal_average_previous = symbol_0_signal_average
        symbol_1_signal_average_previous = symbol_1_signal_average
        symbol_2_signal_average_previous = symbol_2_signal_average
        symbol_3_signal_average_previous = symbol_3_signal_average

    # Calculate the value of the floating average
    # Считаем значения плавающего среднего
    symbol_0_signal_average = np.mean(symbol_0_signal_average_buffer)
    symbol_1_signal_average = np.mean(symbol_1_signal_average_buffer)
    symbol_2_signal_average = np.mean(symbol_2_signal_average_buffer)
    symbol_3_signal_average = np.mean(symbol_3_signal_average_buffer)

    # Second average buffer
    # Помещаем результат во второй буфер
    symbol_0_signal_average_second_buffer[symbol_signal_average_buffer_counter] = symbol_0_signal_average
    symbol_1_signal_average_second_buffer[symbol_signal_average_buffer_counter] = symbol_1_signal_average
    symbol_2_signal_average_second_buffer[symbol_signal_average_buffer_counter] = symbol_2_signal_average
    symbol_3_signal_average_second_buffer[symbol_signal_average_buffer_counter] = symbol_3_signal_average

    # Calculate the value of the floating average
    # Считаем значения плавающего среднего
    symbol_0_signal_average = abs(np.mean(symbol_0_signal_average_second_buffer))
    symbol_1_signal_average = abs(np.mean(symbol_1_signal_average_second_buffer))
    symbol_2_signal_average = abs(np.mean(symbol_2_signal_average_second_buffer))
    symbol_3_signal_average = abs(np.mean(symbol_3_signal_average_second_buffer))

    # Debug
    # Для отладки
    pll_type_2_symbol_0_filter_output_debug.append(symbol_0_signal_average)
    pll_type_2_symbol_1_filter_output_debug.append(symbol_1_signal_average)
    pll_type_2_symbol_2_filter_output_debug.append(symbol_2_signal_average)
    pll_type_2_symbol_3_filter_output_debug.append(symbol_3_signal_average)

    # Floating average buffer counter increment
    # Инкремент счётчика буфера плавающего среднего
    symbol_signal_average_buffer_counter += 1
    if symbol_signal_average_buffer_counter >= samplerate_period_per_symbol:
        symbol_signal_average_buffer_counter = 0

    # If the carrier capture flag is not set
    # Если не выставлен флаг захвата несущей
    if pll_type_2_carrier_lock_flag == 0:
        # Select symbol value depending on the given preamble symbol
        # Выбор значения символа в зависимости от заданного символа преамбулы
        if carrier_symbol_value == 0:
            pll_type_2_carrier_symbol_value = symbol_0_signal_average
        elif carrier_symbol_value == 1:
            pll_type_2_carrier_symbol_value = symbol_1_signal_average
        elif carrier_symbol_value == 2:
            pll_type_2_carrier_symbol_value = symbol_2_signal_average
        else:
            pll_type_2_carrier_symbol_value = symbol_3_signal_average

        # Carrier lock control
        # Контроль захвата несущей
        if symbol_signal_average_buffer_counter == 0:
            # Increment the counter when the value is greater than the threshold, otherwise reset the counter
            # Инкремент счётчика когда значение больше порога, иначе сброс счётчика
            if pll_type_2_carrier_symbol_value > pll_type_2_carrier_lock_threshold:
                pll_type_2_carrier_lock_counter += 1
            else:
                pll_type_2_carrier_lock_counter = 0

        # Checking the counter value
        # Проверка значения счётчика
        if pll_type_2_carrier_lock_counter > pll_type_2_carrier_lock_counter_threshold:
            # There is a carrier lock
            # Есть захват несущей
            pll_type_2_carrier_lock_flag = 1
            pll_type_2_carrier_lock_sample_debug.append(i)
            pll_type_2_carrier_lock_value_debug.append(pll_type_2_carrier_symbol_value)

    # If the carrier lock flag is set
    # Если выставлен флаг захвата несущей
    else:
        # Preamble search
        # We consider a sequential change of characters
        # Should be alternating 0, 1, 2, 3, 0 ...
        # In addition, we are looking for the maximum of each of the characters, for character synchronization
        # We are looking for an increase and immediately a decrease, this is the maximum
        # Поиск преамбулы
        # Считаем последовательную смену символов
        # Должно быть чередование 0, 1, 2, 3, 0 ...
        # Кроме того ищем максимум каждого из символов, для символьной синхронизации
        # Ищем нарастание и сразу спад, это и есть максимум
        if preamble_lock_flag == 0:
            # Signal level control and slew control
            # Контроль уровня сигнала и контроль нарастания

            # Previous character value
            # Предыдущее значение символа
            symbol_digital_value_previous = symbol_digital_value

            # Symbol 0 received, level greater than other signals and greater than threshold
            # Символ 0, уровень больше других сигналов и больше порога
            if symbol_0_signal_average > symbol_signal_average_threshold:
                if (symbol_0_signal_average > symbol_1_signal_average) and (symbol_0_signal_average > symbol_2_signal_average) and (symbol_0_signal_average > symbol_3_signal_average):
                    if symbol_0_signal_average > symbol_0_signal_average_previous:
                        symbol_0_signal_rising_flag = 1
                    else:
                        # If there was an increase, and now there is no, then this is the maximum
                        # This is the middle of the bit for the receiver and the end for the transmitter. the filter will give the maximum at the end
                        # Если было нарастание, а теперь нет, то это максимум
                        # Это середина бита для приёмника и конец для передатчика т.к. фильтр даст максимум в конце
                        if symbol_0_signal_rising_flag == 1:
                            symbol_0_signal_rising_flag = 0
                            symbol_digital_value = 0 # Значение символа дискретное
                            symbol_n_signal_average = symbol_0_signal_average
                            # Для отладки
                            symbol_digital_value_debug_x.append(i)
                            symbol_digital_value_debug_y.append(symbol_0_signal_average)
                            symbol_digital_value_debug.append('0')

            # Symbol 1 received, level greater than other signals and greater than threshold
            # Символ 1, уровень больше других сигналов и больше порога
            if symbol_1_signal_average > symbol_signal_average_threshold:
                if (symbol_1_signal_average > symbol_0_signal_average) and (symbol_1_signal_average > symbol_2_signal_average) and (symbol_1_signal_average > symbol_3_signal_average):
                    if symbol_1_signal_average > symbol_1_signal_average_previous:
                        symbol_1_signal_rising_flag = 1
                    else:
                        # If there was an increase, and now there is no, then this is the maximum
                        # This is the middle of the bit for the receiver and the end for the transmitter. the filter will give the maximum at the end
                        # Если было нарастание, а теперь нет, то это максимум
                        # Это середина бита для приёмника и конец для передатчика т.к. фильтр даст максимум в конце
                        if symbol_1_signal_rising_flag == 1:
                            symbol_1_signal_rising_flag = 0
                            symbol_digital_value = 1 # Значение символа дискретное
                            symbol_n_signal_average = symbol_1_signal_average
                            # Для отладки
                            symbol_digital_value_debug_x.append(i)
                            symbol_digital_value_debug_y.append(symbol_1_signal_average)
                            symbol_digital_value_debug.append('1')

            # Symbol 2 received, level greater than other signals and greater than threshold
            # Символ 2, уровень больше других сигналов и больше порога
            if symbol_2_signal_average > symbol_signal_average_threshold:
                if (symbol_2_signal_average > symbol_0_signal_average) and (symbol_2_signal_average > symbol_1_signal_average) and (symbol_2_signal_average > symbol_3_signal_average):
                    if symbol_2_signal_average > symbol_2_signal_average_previous:
                        symbol_2_signal_rising_flag = 1
                    else:
                        # If there was an increase, and now there is no, then this is the maximum
                        # This is the middle of the bit for the receiver and the end for the transmitter. the filter will give the maximum at the end
                        # Если было нарастание, а теперь нет, то это максимум
                        # Это середина бита для приёмника и конец для передатчика т.к. фильтр даст максимум в конце
                        if symbol_2_signal_rising_flag == 1:
                            symbol_2_signal_rising_flag = 0
                            symbol_digital_value = 2 # Значение символа дискретное
                            symbol_n_signal_average = symbol_2_signal_average
                            # Для отладки
                            symbol_digital_value_debug_x.append(i)
                            symbol_digital_value_debug_y.append(symbol_2_signal_average)
                            symbol_digital_value_debug.append('2')

            # Symbol 3 received, level greater than other signals and greater than threshold
            # Символ 3, уровень больше других сигналов и больше порога
            if symbol_3_signal_average > symbol_signal_average_threshold:
                if (symbol_3_signal_average > symbol_0_signal_average) and (symbol_3_signal_average > symbol_1_signal_average) and (symbol_3_signal_average > symbol_2_signal_average):
                    if symbol_3_signal_average > symbol_3_signal_average_previous:
                        symbol_3_signal_rising_flag = 1
                    else:
                        # If there was an increase, and now there is no, then this is the maximum
                        # This is the middle of the bit for the receiver and the end for the transmitter. the filter will give the maximum at the end
                        # Если было нарастание, а теперь нет, то это максимум
                        # Это середина бита для приёмника и конец для передатчика т.к. фильтр даст максимум в конце
                        if symbol_3_signal_rising_flag == 1:
                            symbol_3_signal_rising_flag = 0
                            symbol_digital_value = 3 # Значение символа дискретное
                            symbol_n_signal_average = symbol_3_signal_average
                            # Для отладки
                            symbol_digital_value_debug_x.append(i)
                            symbol_digital_value_debug_y.append(symbol_3_signal_average)
                            symbol_digital_value_debug.append('3')

            # Check for alternating symbols 0, 1, 2, 3, 0, ...
            # Проверка на чередование символов 0, 1, 2, 3, 0, ...
            if symbol_digital_value != symbol_digital_value_previous:
                if (symbol_digital_value > symbol_digital_value_previous) or ((symbol_digital_value == 0) and (symbol_digital_value_previous == 3)):
                    symbol_alternation_counter += 1
                    if symbol_alternation_counter > symbol_alternation_counter_threshold:
                        preamble_lock_flag = 1

                        # Remember the value of the counter indicating the middle of the bit
                        # Запомним значение счётичка указывающего на середину бита
                        symbol_signal_average_buffer_counter_lock_value = symbol_signal_average_buffer_counter

                        # Для отладки
                        preamble_lock_sample_debug.append(i)
                        preamble_lock_value_debug.append(symbol_n_signal_average)
                else:
                    symbol_alternation_counter = 0

        # If the preamble capture flag is set
        # Если выставлен флаг захвата преамбулы
        else:
            # Determine the current value of the symbol
            # Определим текущее значение символа
            # Смотрим в середние символа
            if symbol_signal_average_buffer_counter == symbol_signal_average_buffer_counter_lock_value:
                # Previous symbol value
                # Предыдущее значение символа
                symbol_digital_value_previous = symbol_digital_value

                # The symbol with the larger average value is received
                # Принят тот символ у которого больше среднее значение
                if (symbol_0_signal_average > symbol_1_signal_average) and (symbol_0_signal_average > symbol_2_signal_average) and (symbol_0_signal_average > symbol_3_signal_average):
                    symbol_digital_value = 0  # Значение символа дискретное
                    symbol_n_signal_average = symbol_0_signal_average
                    # Для отладки
                    symbol_digital_value_debug_x.append(i)
                    symbol_digital_value_debug_y.append(symbol_0_signal_average)
                    symbol_digital_value_debug.append('0')

                if (symbol_1_signal_average > symbol_0_signal_average) and (symbol_1_signal_average > symbol_2_signal_average) and (symbol_1_signal_average > symbol_3_signal_average):
                    symbol_digital_value = 1  # Значение символа дискретное
                    symbol_n_signal_average = symbol_1_signal_average
                    # Для отладки
                    symbol_digital_value_debug_x.append(i)
                    symbol_digital_value_debug_y.append(symbol_1_signal_average)
                    symbol_digital_value_debug.append('1')

                if (symbol_2_signal_average > symbol_0_signal_average) and (symbol_2_signal_average > symbol_1_signal_average) and (symbol_2_signal_average > symbol_3_signal_average):
                    symbol_digital_value = 2  # Значение символа дискретное
                    symbol_n_signal_average = symbol_2_signal_average
                    # Для отладки
                    symbol_digital_value_debug_x.append(i)
                    symbol_digital_value_debug_y.append(symbol_2_signal_average)
                    symbol_digital_value_debug.append('2')

                if (symbol_3_signal_average > symbol_0_signal_average) and (symbol_3_signal_average > symbol_1_signal_average) and (symbol_3_signal_average > symbol_2_signal_average):
                    symbol_digital_value = 3  # Значение символа дискретное
                    symbol_n_signal_average = symbol_3_signal_average
                    # Для отладки
                    symbol_digital_value_debug_x.append(i)
                    symbol_digital_value_debug_y.append(symbol_3_signal_average)
                    symbol_digital_value_debug.append('3')

                # Waiting for sync word
                # Ищем слово синхронизации
                if synchronization_word_lock_flag == 0:
                    # Sync word bit counter
                    # Счётчик бит слова синхронизации
                    if symbol_digital_value == synchronization_word_symbol_value:
                        synchronization_word_bit_counter += 1
                    else:
                        synchronization_word_bit_counter = 0

                    # Compare counter with threshold
                    # Сравнение счётчика с порогом
                    if synchronization_word_bit_counter == synchronization_word_symbol_size:
                        synchronization_word_lock_flag = 1
                        synchro_lock_sample_debug.append(i)
                        synchro_lock_value_debug.append(symbol_n_signal_average)

                # If the carrier, preamble and sync word are captured
                # Если захвачены несущаяя, преамбула и синхрослово
                else:
                    # Received symbols and put them into bytes
                    # Принимаем символы и помещаем их в байты

                    # Put a symbol (two bits) into a byte
                    # Помещаем символ в байт
                    byte_value = byte_value | (symbol_digital_value << output_bit_count)

                    # Bit and byte counters, save byte, reset byte value
                    # Счётчики бит и байт, сохраняем байт, сброс значения байта
                    output_bit_count += 2
                    if output_bit_count == 8:
                        output_stream_bytes.append(byte_value)
                        output_bit_count = 0
                        output_byte_count += 1
                        byte_digital_sample_x_debug.append(i)
                        byte_digital_sample_y_debug.append(0.55 + (output_byte_count % 2) / 30)
                        byte_digital_value_debug.append(byte_value)
                        byte_value = 0

# Debug
# Для отладки
plt.plot(pll_type_2_symbol_0_filter_output_debug, "-r", pll_type_2_symbol_1_filter_output_debug, "-g", pll_type_2_symbol_2_filter_output_debug, "-b", pll_type_2_symbol_3_filter_output_debug, "-y")
plt.title('Receive QFSK, rate (приём QFSK сигнала со скоростью) {0} бит/сек'.format(transmit_bit_rate))
plt.xlabel('Sample Номер отсчёта', color='gray')
plt.ylabel('Filter output Выход фильтра для значения захвата сигнала', color='gray')
plt.legend(['Symbol (символ) 00, frequency (частота) {0} Hz (Гц)'.format(symbol_0_frequency), 'Symbol (символ) 01, frequency (частота) {0} Hz (Гц)'.format(symbol_1_frequency), 'Symbol (символ) 10, frequency (частота) {0} Hz (Гц)'.format(symbol_2_frequency), 'Symbol (символ)  11, frequency (частота) {0} Hz (Гц)'.format(symbol_3_frequency)], loc = 2)

plt.plot(pll_type_2_carrier_lock_sample_debug, pll_type_2_carrier_lock_value_debug, 'rs')
for i in range(len(pll_type_2_carrier_lock_sample_debug)):
    plt.annotate('pll lock', (pll_type_2_carrier_lock_sample_debug[i], pll_type_2_carrier_lock_value_debug[i]), ha='center')

plt.plot(symbol_digital_value_debug_x, symbol_digital_value_debug_y, 'ro')
for i in range(len(symbol_digital_value_debug_x)):
    plt.annotate(symbol_digital_value_debug[i], (symbol_digital_value_debug_x[i], symbol_digital_value_debug_y[i]), ha='center')

plt.plot(preamble_lock_sample_debug, preamble_lock_value_debug, 'rs')
for i in range(len(preamble_lock_sample_debug)):
    x = preamble_lock_sample_debug[i]
    y = preamble_lock_value_debug[i]
    if y > 0:
        yt = y + 0.1
    else:
        yt = y - 0.1
    plt.annotate('preamble lock', xy = (x, y), xytext = (x, yt), arrowprops = dict(facecolor ='green', shrink = 0.05))

plt.plot(synchro_lock_sample_debug, synchro_lock_value_debug, 'rs')
for i in range(len(synchro_lock_sample_debug)):
    x = synchro_lock_sample_debug[i]
    y = synchro_lock_value_debug[i]
    if y > 0:
        yt = y + 0.1
    else:
        yt = y - 0.1
    plt.annotate('sync lock', xy = (x, y), xytext = (x, yt), arrowprops = dict(facecolor ='green', shrink = 0.05))

plt.plot(byte_digital_sample_x_debug, byte_digital_sample_y_debug, 'rs')
for i in range(len(byte_digital_sample_x_debug)):
    plt.annotate("0x{0:x}".format(byte_digital_value_debug[i]), (byte_digital_sample_x_debug[i], byte_digital_sample_y_debug[i]), ha='center')
    # plt.axvline(byte_digital_sample_x_debug[i], color='y', linestyle='dotted', label='axvline - full height')

plt.grid()
plt.show()

# Convert to uint8
# Преобразуем в uint8
output_stream_bytes_uint8 = np.uint8(output_stream_bytes)

# Write the data file
# Записываем файл с данными
file = open(data_file_out_name, "wb")
file.write(output_stream_bytes_uint8)
file.close()