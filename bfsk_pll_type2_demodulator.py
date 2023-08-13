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
wav_file_name = 'wav\\bfsk_out.wav'
# wav_file_name = 'wav\\bfsk_441bps_metro_5.wav'
# wav_file_name = 'wav\\bfsk_245bps_metro_5.wav'
wav_file_name = os.path.join(os.path.dirname(__file__), wav_file_name)

# Name of the output data file
# Имя выходного файла с данными
data_file_out_name = 'received_data.txt'
data_file_out_name = os.path.join(os.path.dirname(__file__), data_file_out_name)

# Sampling rate of wav file, other values don't work
# Частота дискретизации файла wav
wav_samplerate = 44100

# Carrier frequency value for transmission 0
# Значение частоты для передачи бита 0
bit_0_frequency = 1764

# Carrier frequency value for transmission 1
# Значение частоты для передачи бита 1
bit_1_frequency = 2940

# Data transfer rate, bits per second
# Скорость передачи данных
transmit_bitrate = 147

# Carrier frequency duration in bit durations, transmitted before the preamble
# Used for PLL operation and signal detection
# Размер несущей в битах
carrier_bit_size = 24

# Frequency value for preamble transmission
# Значение частоты преамбулы, задано значением бита
carrier_bit_value = 0

# Preamble (alternating frequency) duration, in bit durations
# Used for bit synchronization
# Размер преамбулы, бит
preamble_bit_size = 32

# Synchronization word size, bits. Used for byte synchronization
# Размер слова синхронизации, бит
synchronization_word_bit_size = 8

# Sync word bits value
# Значение бит слова синхронизации
synchronization_word_bit_value = 1

# Число отсчётов частоты дискретизации на каждый бит
# Число отсчётов частоты дискретщизации на каждый бит
bit_samplerate_period_per_bit = wav_samplerate / transmit_bitrate

# Open wav file for reading
# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_name)
input_signal_length = input_signal_data.shape[0]

# Signal duration, bits and bytes
# Длительность посылки бит и байт
input_signal_bit_length = input_signal_length / bit_samplerate_period_per_bit
input_signal_byte_length = input_signal_bit_length / 8

# Output parameters to the console
# Вывод параметров
print("Частота дискретизации wav =", wav_samplerate)
print("Частота для бита 0 =", bit_0_frequency)
print("Частота для бита 1 =", bit_1_frequency)
print("Число периодов дискрертизации на бит =", bit_samplerate_period_per_bit);
print("Число отсчётов входного файла =", input_signal_length)
print("Длительность входных данных секунд =", input_signal_length/input_signal_samplerate)
print("Размер несущей, бит =", carrier_bit_size)
print("Значение несущей несущей, бит =", carrier_bit_value)
print("Размер преамбулы, бит =", preamble_bit_size)
print("Размер слова синхронизации, бит =", synchronization_word_bit_size)
print("Значение бит слова синхронизации =", synchronization_word_bit_value)
print("Длина посылки в битах =", input_signal_bit_length)
print("Длина посылки в байтах =", input_signal_byte_length)

# Calculate an array with sinusoid samples
# Формируем отсчёты синусоид и косинусоид двух частот для одного бита
bit_sin_samples_index = np.arange(bit_samplerate_period_per_bit)
bit_0_sin_samples = np.sin(2 * np.pi * (bit_0_frequency / transmit_bitrate) * bit_sin_samples_index / bit_samplerate_period_per_bit)
bit_0_cos_samples = np.cos(2 * np.pi * (bit_0_frequency / transmit_bitrate) * bit_sin_samples_index / bit_samplerate_period_per_bit)
bit_1_sin_samples = np.sin(2 * np.pi * (bit_1_frequency / transmit_bitrate) * bit_sin_samples_index / bit_samplerate_period_per_bit)
bit_1_cos_samples = np.cos(2 * np.pi * (bit_1_frequency / transmit_bitrate) * bit_sin_samples_index / bit_samplerate_period_per_bit)

# plt.plot(bit_sin_samples_index, bit_0_sin_samples, "-g", bit_sin_samples_index, bit_1_sin_samples, "-b")
# plt.show()

# Scale the samples of the input signal so that they are in the range from -1 to 1
# Масштабируем входной сигнал, что бы максимум был 1 или -1
input_signal_maximum_amplitude = max(abs(input_signal_data))
input_signal_data = input_signal_data / input_signal_maximum_amplitude

# The result of multiplying the input signal by the reference signal
# Результат перемножения сигнала на опорный сигнал
pll_type_2_signal_reference_multiplication_value = 0

# Calculation of PI filter coefficients
# Вычисление коэффициентов ПИ регулятора
pll_bandwidth = transmit_bitrate
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

# Type 2 PLL filter output value
# Значение выхода фильтра ФАПЧ типа 2
pll_type_2_loop_value_k1 = 0
pll_type_2_loop_value_k2 = 0
pll_type_2_filter_output = 0

# Current reference signal value for type 2 PLL
# Текущее значение опорного сигнала для ФАПЧ типа 2
pll_type_2_reference_cos_sample_value = 0
pll_type_2_reference_sin_sample_value = 0

# Phase counter for PLL type 2
# Счётчик фазы для ФАПЧ типа 2
pll_type_2_phase_cnt = 0

# The result of multiplying the signal by the reference signal
# Результат перемножения сигнала на опорный сигнал
pll_type_2_signal_reference_multiplication_value = 0

# Carrier capture flag
# Флаг захвата несущей
pll_type_2_carrier_lock_flag = 0

# To implement average filters
# Буфер плавающего среднего, счётчик буфера и размер буфера
# Для определения захвата несущей
pll_type_2_average_buffer_size = bit_samplerate_period_per_bit
pll_type_2_average_buffer_counter = 0
pll_type_2_average_buffer = np.linspace(0, 0, int(pll_type_2_average_buffer_size))
pll_type_2_average_value = 0

# Carrier lock amplitude threshold
# Порог определения захвата несущей
pll_type_2_carrier_lock_threshold = 0.05

# Signal threshold for bit selection/detection
# Порог для выделения бит
bit_signal_level_threshold = 0.05

# Carrier capture counter and counter threshold
# Счётчик для определения захвата несущей и порог счётчика
pll_type_2_carrier_lock_counter = 0
pll_type_2_carrier_lock_counter_threshold = carrier_bit_size // 2

# Reference signal values for bit detection
# Значения опорных сигналов для выделения бит
pll_bit_0_reference_sin_sample_value = 0
pll_bit_1_reference_sin_sample_value = 0

# Result of signal multiplication by reference signals
# Результат перемножения сигнала на опорные сигналы
bit_0_signal_reference_multiplication_value = 0
bit_1_signal_reference_multiplication_value = 0

# To implement average filters
# Фильтр плавающего среднего для принимаемых бит
bits_signal_average_buffer_size = bit_samplerate_period_per_bit
bits_signal_average_buffer_counter = 0
bit_0_signal_average_buffer = np.linspace(0, 0, int(bits_signal_average_buffer_size))
bit_1_signal_average_buffer = np.linspace(0, 0, int(bits_signal_average_buffer_size))
bit_0_signal_average = 0
bit_1_signal_average = 0

# Second buffer
# Второй буфер
bit_0_signal_average_second_buffer = np.linspace(0, 0, int(bits_signal_average_buffer_size))
bit_1_signal_average_second_buffer = np.linspace(0, 0, int(bits_signal_average_buffer_size))

# To search for a preamble
# Для поиска преамбулы
preamble_lock_flag = 0
bit_0_signal_difference = 0
bit_1_signal_difference = 0
bit_0_signal_difference_previous = 0
bit_1_signal_difference_previous = 0
bit_0_signal_difference_max = 0
bit_1_signal_difference_max = 0
bit_0_signal_difference_rising_flag = 0
bit_1_signal_difference_rising_flag = 0
bit_value_change_counter = 0
bit_value_change_lock_threshold = preamble_bit_size // 2

# To search for the word sync
# Для поиска слова синхронизации
synchronization_word_bit_counter = 0
synchronization_word_lock_flag = 0

# The value of the received bit and the previous bit value, byte value, value filter counter in which bit received
# Данные
bit_digital_value = 0
bit_digital_previous_value = 0
bit_digital_buffer_counter_value = 0
byte_value = 0

# Byte counter and bit counter
# Счётчик байт и счётчик бит
output_byte_count = 0
output_bit_count = 0

# Output data, bytes
# Выходные данные, байты
output_stream_bytes = []

# Debug
# Для отладки
pll_type_2_filter_output_debug = []
pll_type_2_average_value_debug = []
pll_type_2_carrier_lock_sample_debug = []
pll_type_2_carrier_lock_value_debug = []
bit_0_signal_average_debug = []
bit_1_signal_average_debug = []
preamble_lock_sample_debug = []
preamble_lock_value_debug = []
bit_digital_value_debug_x = []
bit_digital_value_debug_y = []
bit_digital_value_debug = []
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

    # If the carrier capture flag is not set
    # Если не выставлен флаг захвата несущей
    if pll_type_2_carrier_lock_flag == 0:
        # Current samples of reference signals
        # Текущее значение опорного сигнала
        if carrier_bit_value == 0:
            pll_type_2_reference_cos_sample_value = bit_0_cos_samples[int(pll_type_2_phase_cnt)]
            pll_type_2_reference_sin_sample_value = bit_0_sin_samples[int(pll_type_2_phase_cnt)]
        else:
            pll_type_2_reference_cos_sample_value = bit_1_cos_samples[int(pll_type_2_phase_cnt)]
            pll_type_2_reference_sin_sample_value = bit_1_sin_samples[int(pll_type_2_phase_cnt)]

        # Multiply the input signal by the reference signals
        # Перемножаем входной сигнал на опорный сигнал, опорный сигнал на квадратурный опорный
        pll_type_2_signal_reference_multiplication_value = pll_type_2_reference_cos_sample_value * input_signal_data[i]

        # Calculate the PLL filter values
        # Считаем значения фильтра ФАПЧ
        pll_type_2_loop_value_k1 = pll_type_2_signal_reference_multiplication_value * pll_type_2_coefficient_k1
        pll_type_2_filter_output = pll_type_2_loop_value_k1 + pll_type_2_loop_value_k2
        pll_type_2_loop_value_k2 = pll_type_2_signal_reference_multiplication_value * pll_type_2_coefficient_k2

        # Debug
        pll_type_2_filter_output_debug.append(pll_type_2_filter_output)

        # Calculation of the phase counter value for reference signals
        # costas_loop_iq_filter_output - phase error
        # Инкремент счётчика фазы с добавлением ошибки фазы
        pll_type_2_phase_cnt = pll_type_2_phase_cnt + 1 + pll_type_2_filter_output
        if pll_type_2_phase_cnt >= bit_samplerate_period_per_bit:
            pll_type_2_phase_cnt = pll_type_2_phase_cnt - bit_samplerate_period_per_bit
        if pll_type_2_phase_cnt < 0:
            pll_type_2_phase_cnt = pll_type_2_phase_cnt + bit_samplerate_period_per_bit

        # Multiply the input signal by the reference signal, the reference signal by the inphase reference
        # Перемножаем входной сигнал на опорный сигнал, опорный сигнал на синфазный опорный
        pll_type_2_signal_reference_multiplication_value = pll_type_2_reference_sin_sample_value * input_signal_data[i]

        # Put the result in the floating average buffer
        # Помещаем результат в буфер плавающего среднего
        pll_type_2_average_buffer[pll_type_2_average_buffer_counter] = pll_type_2_signal_reference_multiplication_value

        # Calculate the value of the floating average
        # Считаем значение плавающего среднего
        pll_type_2_average_value = np.mean(pll_type_2_average_buffer)

        # Debug
        pll_type_2_average_value_debug.append(pll_type_2_average_value)
        bit_0_signal_average_debug.append(0)
        bit_1_signal_average_debug.append(0)

        # Floating average buffer counter increment
        # Инкремент счётчика буфера плавающего среднего
        pll_type_2_average_buffer_counter += 1
        if pll_type_2_average_buffer_counter >= pll_type_2_average_buffer_size:
            pll_type_2_average_buffer_counter = 0

        # Carrier lock control
        # Контроль захвата несущей
        if pll_type_2_average_buffer_counter == 0:
            # Increment the counter when the value is greater than the threshold, otherwise reset the counter
            # Инкремент счётчика когда значение больше порога, иначе сброс счётчика
            if pll_type_2_average_value > pll_type_2_carrier_lock_threshold:
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
                pll_type_2_carrier_lock_value_debug.append(pll_type_2_average_value)
    else:
        # Добавил ФАПЧ в процессе приёма, улучшило работу
        # Было потеряно 2 бита из 256 без этого

        # The current value of the reference signal
        # Текущее значение опорного сигнала
        if carrier_bit_value == 0:
            pll_type_2_reference_cos_sample_value = bit_0_cos_samples[int(pll_type_2_phase_cnt)]
            pll_type_2_reference_sin_sample_value = bit_0_sin_samples[int(pll_type_2_phase_cnt)]
        else:
            pll_type_2_reference_cos_sample_value = bit_1_cos_samples[int(pll_type_2_phase_cnt)]
            pll_type_2_reference_sin_sample_value = bit_0_sin_samples[int(pll_type_2_phase_cnt)]

        # Multiply the input signal by the reference signal, the reference signal by the quadrature reference
        # Перемножаем входной сигнал на опорный сигнал, опорный сигнал на квадратурный опорный
        pll_type_2_signal_reference_multiplication_value = pll_type_2_reference_cos_sample_value * input_signal_data[i]

        # Calculate the PLL filters values
        # Считаем значения фильтра ФАПЧ
        pll_type_2_loop_value_k1 = pll_type_2_signal_reference_multiplication_value * pll_type_2_coefficient_k1
        pll_type_2_filter_output = pll_type_2_loop_value_k1 + pll_type_2_loop_value_k2
        pll_type_2_loop_value_k2 = pll_type_2_signal_reference_multiplication_value * pll_type_2_coefficient_k2

        # Calculation of the phase counter value for reference signals
        # costas_loop_iq_filter_output - phase error
        # Инкремент счётчика фазы с добавлением ошибки фазы
        pll_type_2_phase_cnt = pll_type_2_phase_cnt + 1 + pll_type_2_filter_output
        if pll_type_2_phase_cnt >= bit_samplerate_period_per_bit:
            pll_type_2_phase_cnt = pll_type_2_phase_cnt - bit_samplerate_period_per_bit
        if pll_type_2_phase_cnt < 0:
            pll_type_2_phase_cnt = pll_type_2_phase_cnt + bit_samplerate_period_per_bit

    # If the carrier lock flag is set
    # Если выставлен флаг захвата несущей
    if pll_type_2_carrier_lock_flag == 1:
        # Implementation of receiving individual bits
        # Реализация приёма отдельных бит

        # Reference frequency samples for bit 0 and bit 1
        # Отсчёты опорных частот для бита 0 и бита 1
        pll_bit_0_reference_sin_sample_value = bit_0_sin_samples[int(pll_type_2_phase_cnt)]
        pll_bit_1_reference_sin_sample_value = bit_1_sin_samples[int(pll_type_2_phase_cnt)]

        # Инкремент счётчика фазы, частоты бита 0 - закоментировал так как ввёл ФАПЧ при приёме выше
        '''pll_type_2_phase_cnt = pll_type_2_phase_cnt + 1
        if pll_type_2_phase_cnt >= bit_samplerate_period_per_bit:
            pll_type_2_phase_cnt = 0'''

        # Multiply the input signal by the reference signals
        # Перемножаем входной сигнал на опорные сигналы
        pll_bit_0_signal_reference_multiplication_value = input_signal_data[i] * pll_bit_0_reference_sin_sample_value
        pll_bit_1_signal_reference_multiplication_value = input_signal_data[i] * pll_bit_1_reference_sin_sample_value

        # Place the result of the multiplication in the floating average buffer
        # Помещаем результат умножения в буфер плавающего среднего
        bit_0_signal_average_buffer[bits_signal_average_buffer_counter] = pll_bit_0_signal_reference_multiplication_value
        bit_1_signal_average_buffer[bits_signal_average_buffer_counter] = pll_bit_1_signal_reference_multiplication_value

        # Floating average buffer counter increment
        # Инкремент счётчика буфера плавающего среднего
        bits_signal_average_buffer_counter += 1
        if bits_signal_average_buffer_counter >= bits_signal_average_buffer_size:
            bits_signal_average_buffer_counter = 0

        # Calculate the floating average
        # Считаем плавающее среднее
        bit_0_signal_average = np.mean(bit_0_signal_average_buffer)
        bit_1_signal_average = np.mean(bit_1_signal_average_buffer)

        # Put the result in the second buffer
        # Помещаем результат во второй буфер
        bit_0_signal_average_second_buffer[bits_signal_average_buffer_counter] = bit_0_signal_average
        bit_1_signal_average_second_buffer[bits_signal_average_buffer_counter] = bit_1_signal_average

        # Calculate the floating average for the second buffer
        # Absolute value
        # Считаем плавающее среднее для второго буфера
        # Абсолютное значение, так второй сигнала (не несущей) имеет обратный знак
        bit_0_signal_average = abs(np.mean(bit_0_signal_average_second_buffer))
        bit_1_signal_average = abs(np.mean(bit_1_signal_average_second_buffer))

        # bit_0_signal_average -= (bit_0_signal_average + bit_1_signal_average) / 2
        # bit_1_signal_average -= (bit_0_signal_average + bit_1_signal_average) / 2

        # Debug
        pll_type_2_average_value_debug.append(0)
        bit_0_signal_average_debug.append(bit_0_signal_average)
        bit_1_signal_average_debug.append(bit_1_signal_average)

        # Preamble search
        # Поиск преамбулы
        if preamble_lock_flag == 0:
            bit_0_signal_difference_previous = bit_0_signal_difference
            bit_1_signal_difference_previous = bit_1_signal_difference
            bit_0_signal_difference = bit_0_signal_average - bit_1_signal_average
            bit_1_signal_difference = bit_1_signal_average - bit_0_signal_average

            # bit_0_signal_average_debug.append(bit_0_signal_difference)
            # bit_1_signal_average_debug.append(bit_1_signal_difference)

            # Bit 1 signal rising, rising edge
            # Current value is greater than previous
            # Нарастание сигнала бита 1, передний фронт
            # Текщее значение больше предыдущего
            if (bit_1_signal_average > bit_signal_level_threshold) and (bit_1_signal_difference > bit_1_signal_difference_previous):
                 bit_1_signal_difference_rising_flag = 1
            else:
                # If there was an increase, and now there is no, then this is the maximum
                # This is the middle of bit 1 for the receiver and the end for the transmitter
                # Filter will give a maximum at the end
                # Если было нарастание, а теперь нет, то это максимум
                # Это середина бита 1 для приёмника и конец для передатчика т.к. фильтра даст максимум в конце
                if bit_1_signal_difference_rising_flag == 1:
                    bit_1_signal_difference_max = bit_1_signal_difference
                    bit_1_signal_difference_rising_flag = 0
                    bit_digital_value_debug_x.append(i)
                    bit_digital_value_debug_y.append(bit_1_signal_difference)
                    bit_digital_value_debug.append('1')
                    bit_digital_value = 1

            # Bit 0 signal rising, rising edge
            # Current value is greater than previous
            # Нарастание сигнала бита 0, передний фронт
            # Текщее значение больше предыдущего
            if (bit_0_signal_average > bit_signal_level_threshold) and (bit_0_signal_difference > bit_0_signal_difference_previous):
                bit_0_signal_difference_rising_flag = 1
            else:
                # If there was an increase, and now there is no, then this is the maximum
                # This is the middle of bit 1 for the receiver and the end for the transmitter
                # Filter will give a maximum at the end
                # Если было нарастание, а теперь нет, то это максимум
                # Это середина бита 0 для приёмника и конец для передатчика т.к. фильтра даст максимум в конце
                if bit_0_signal_difference_rising_flag == 1:
                    bit_0_signal_difference_max = bit_0_signal_difference
                    bit_0_signal_difference_rising_flag = 0
                    bit_digital_value_debug_x.append(i)
                    bit_digital_value_debug_y.append(bit_0_signal_difference)
                    bit_digital_value_debug.append('0')
                    bit_digital_value = 0

            # Check for bit alternation
            # Проверка на чередование бит
            if bit_digital_previous_value != bit_digital_value:
                bit_value_change_counter += 1
                bit_digital_previous_value = bit_digital_value
                if bit_value_change_counter > bit_value_change_lock_threshold:
                    preamble_lock_flag = 1
                    preamble_lock_sample_debug.append(i)
                    preamble_lock_value_debug.append((bit_0_signal_average + bit_1_signal_average) / 2)

                    # Remember the counter value for the middle of the bit
                    # Запомним значение счётчика для середины бита
                    bit_digital_buffer_counter_value = bits_signal_average_buffer_counter

        # Waiting for sync word
        # Ждём синхрослово
        if synchronization_word_lock_flag == 0:
            if (preamble_lock_flag == 1) and (bits_signal_average_buffer_counter == bit_digital_buffer_counter_value):
                # Compare bit levels to decide what is received
                # Сравнение уровней бит для решения, что принято
                if bit_0_signal_average >= bit_1_signal_average:
                    bit_digital_value = 0
                    bit_digital_value_debug_x.append(i)
                    bit_digital_value_debug_y.append(bit_0_signal_average)
                    bit_digital_value_debug.append('0')
                else:
                    bit_digital_value = 1
                    bit_digital_value_debug_x.append(i)
                    bit_digital_value_debug_y.append(bit_1_signal_average)
                    bit_digital_value_debug.append('1')

                # Sync word bit counter
                # Счётчик бит слова синхронизации
                if bit_digital_value == synchronization_word_bit_value:
                    synchronization_word_bit_counter += 1
                else:
                    synchronization_word_bit_counter = 0

                # Compare counter with threshold
                # Сравнение счётчика с порогом
                if synchronization_word_bit_counter == synchronization_word_bit_size:
                    synchronization_word_lock_flag = 1
                    synchro_lock_sample_debug.append(i)
                    synchro_lock_value_debug.append((bit_0_signal_average + bit_1_signal_average) / 2)
        else:
        # Receive data
        # Принимаем данные
        # if synchronization_word_lock_flag == 1:
            if bits_signal_average_buffer_counter == bit_digital_buffer_counter_value:
                # Compare bit levels to decide what is received
                # Сравнение уровней бит для решения, что принято
                if bit_0_signal_average >= bit_1_signal_average:
                    bit_digital_value = 0
                    bit_digital_value_debug_x.append(i)
                    bit_digital_value_debug_y.append(bit_0_signal_average)
                    bit_digital_value_debug.append('0')
                else:
                    bit_digital_value = 1
                    bit_digital_value_debug_x.append(i)
                    bit_digital_value_debug_y.append(bit_1_signal_average)
                    bit_digital_value_debug.append('1')

                # Put a bit into a byte
                # Помещаем бит в байт
                byte_value = byte_value | (bit_digital_value << output_bit_count)

                # Bit and byte counters, save byte, reset byte value
                # Счётчики бит и байт, сохраняем байт, сброс значения байта
                output_bit_count += 1
                if output_bit_count == 8:
                    output_stream_bytes.append(byte_value)
                    output_bit_count = 0
                    output_byte_count += 1
                    byte_digital_sample_x_debug.append(i)
                    byte_digital_sample_y_debug.append(0.55 + (output_byte_count % 2) / 30)
                    byte_digital_value_debug.append(byte_value)
                    byte_value = 0

        # Reset preamble flag if levels are low
        # Сброс флага преамбулы, если уровни малы
        if preamble_lock_flag == 1:
            if bits_signal_average_buffer_counter == bit_digital_buffer_counter_value:
                if (bit_1_signal_average < bit_signal_level_threshold) and (bit_0_signal_average < bit_signal_level_threshold):
                    pll_type_2_carrier_lock_flag = 0
                    pll_type_2_carrier_lock_counter = 0
                    preamble_lock_flag = 0
                    bit_value_change_counter = 0
                    bit_1_signal_difference_rising_flag = 0
                    bit_0_signal_difference_rising_flag = 0
                    synchronization_word_lock_flag = 0
                    synchronization_word_bit_counter = 0
                    bit_digital_value = 0
                    output_bit_count = 0
                    byte_value = 0

# Debug
# Для отладки
plt.figure("Time Во времени")
plt.plot(pll_type_2_average_value_debug, "-b", pll_type_2_average_value_debug, "-b")
plt.title('Receive BFSK, rate (приём BFSK сигнала со скоростью) {0} бит/сек'.format(transmit_bitrate))
plt.xlabel('Sample Номер отсчёта', color='gray')
plt.ylabel('Filter output Выход фильтра для значения захвата сигнала', color='gray')
plt.legend(['Bit 0 (бит 0), frequency (частота) {0} Hz (Гц)'.format(bit_0_frequency), 'Bit 1 (бит 1), frequency (частота) {0} Hz (Гц)'.format(bit_1_frequency)], loc = 2)

plt.plot(pll_type_2_carrier_lock_sample_debug, pll_type_2_carrier_lock_value_debug, 'rs')
for i in range(len(pll_type_2_carrier_lock_sample_debug)):
    plt.annotate('pll lock', (pll_type_2_carrier_lock_sample_debug[i], pll_type_2_carrier_lock_value_debug[i]), ha='center')

plt.plot(bit_0_signal_average_debug, "-b", bit_1_signal_average_debug, "-g")

plt.plot(bit_digital_value_debug_x, bit_digital_value_debug_y, 'ro')
for i in range(len(bit_digital_value_debug_x)):
    plt.annotate(bit_digital_value_debug[i], (bit_digital_value_debug_x[i], bit_digital_value_debug_y[i]), ha='center')

plt.plot(preamble_lock_sample_debug, preamble_lock_value_debug, 'rs')
for i in range(len(preamble_lock_sample_debug)):
    plt.annotate('preamble lock', (preamble_lock_sample_debug[i], preamble_lock_value_debug[i]), ha='center')

plt.plot(synchro_lock_sample_debug, synchro_lock_value_debug, 'rs')
for i in range(len(synchro_lock_sample_debug)):
    plt.annotate('sync lock', (synchro_lock_sample_debug[i], synchro_lock_value_debug[i]), ha='center')

plt.plot(byte_digital_sample_x_debug, byte_digital_sample_y_debug, 'rs')
for i in range(len(byte_digital_sample_x_debug)):
    plt.annotate("0x{0:x}".format(byte_digital_value_debug[i]), (byte_digital_sample_x_debug[i], byte_digital_sample_y_debug[i]), ha='center')
    plt.axvline(byte_digital_sample_x_debug[i], color='y', linestyle='dotted', label='axvline - full height')

plt.figure("I/Q plot")
for i in range(len(bit_digital_value_debug_y)):
    if bit_digital_value_debug[i] == '0':
        plt.plot(-bit_digital_value_debug_y[i], 0, 'r.')
    else:
        plt.plot(bit_digital_value_debug_y[i], 0, 'b.')

plt.grid()
plt.show()

# plt.figure(1)
# plt.plot([0], bit_0_signal_average_debug, '-g')
# plt.plot([0], bit_1_signal_average_debug, '-b')

# FFT для сигнала
# signal_fft = np.fft.fftshift(np.fft.fft(input_signal_data))
# signal_fft_magnitude = np.abs(signal_fft) / 32768
# signal_frequency = np.arange(wav_samplerate/-2, wav_samplerate/2, wav_samplerate/input_signal_length)
# plt.figure(1)
# plt.title('Спектр BFSK сигнала со скоростью {0} бит/сек'.format(transmit_bitrate))
# plt.xlabel('Частота, Гц', color='gray')
# plt.ylabel('Магнитуда сигнала', color='gray')
# plt.plot(signal_frequency, signal_fft_magnitude,'.-')

# Спектрограмма сигнала
'''signal_fft_size = 1024
signal_fft_rows = int(input_signal_length // signal_fft_size)
signal_spectrogram = np.zeros((signal_fft_rows, signal_fft_size))
for i in range(signal_fft_rows - 1):
    signal_spectrogram[i, :] = np.log10(np.abs(np.fft.fftshift(np.fft.fft(input_signal_data[i * signal_fft_size : (i + 1) * signal_fft_size]))) + 1)
    # signal_spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(input_signal_data[i * signal_fft_size: (i + 1) * signal_fft_size]))) ** 2)

plt.figure(1)
plt.imshow(signal_spectrogram, aspect='auto', extent = [wav_samplerate / -2, wav_samplerate / 2, 0, input_signal_length / wav_samplerate])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Time [s]")'''

# Convert to uint8
# Преобразуем в uint8
output_stream_bytes_uint8 = np.uint8(output_stream_bytes)

# Write the data file
# Записываем файл с данными
file = open(data_file_out_name, "wb")
file.write(output_stream_bytes_uint8)
file.close()
