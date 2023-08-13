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

# Input data file to transfer
# Имя файла с данными
data_file_name = 'transmit_data.txt'
data_file_name = os.path.join(os.path.dirname(__file__), data_file_name)

# Output audio file, modulated signal
# Имя выходного wav файла
wav_file_name = 'wav\\qfsk_out.wav'
wav_file_name = os.path.join(os.path.dirname(__file__), wav_file_name)

# Sampling rate of wav file, other values don't work
# Частота дискретизации файла wav
wav_samplerate = 44100

# Data transfer rate, bits per second
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

# Carrier frequency duration in bit durations, transmitted before the preamble
# Used for PLL operation and signal detection
# Размер несущей в символах
carrier_symbol_size = 24

# Frequency value for preamble transmission
# Значение частоты преамбулы, задано значением бита
carrier_symbol_value = 0

# Preamble (alternating 00, 01, 10, 11) duration, in bit durations
# Used for bit synchronization
# Размер преамбулы, симвоолов
preamble_symbol_size = 24

# Postamble duration, only needed for Windows player, loses end of wav file
# Размер постамбулы, бит
postamble_symbol_size = 16

# Synchronization word size, bits. Used for byte synchronization
# Размер слова синхронизации, символ
synchronization_word_symbol_size = 8

# Sync word bits value
# Значение символа слова синхронизации
synchronization_word_symbol_value = 1

# Number of wav sampling samples per data symbol
# Число отсчётов частоты дискретизации на каждый символ
samplerate_period_per_symbol = wav_samplerate / transmit_symbol_rate

# Open txt file for reading
# Открываем файл на чтение
data_file = open(data_file_name, "rb")

# Reading a file
# Читаем файл
input_signal_data = bytearray(data_file.read())
input_signal_length_bytes = len(input_signal_data)
input_signal_length_bits = input_signal_length_bytes * 8
input_signal_length_symbols = input_signal_length_bits // 2

# Data stream size with preamble, sync word and postamble, bits
# Размер потока данных с преамбулой,  словом синхронизации и постамбулой, бит
output_signal_data_symbols = input_signal_length_symbols + carrier_symbol_size + preamble_symbol_size + synchronization_word_symbol_size + postamble_symbol_size

# Calculate the duration of all data in wav file samples
# Считаем длительность посылки в отсчётах частоты дискретизации
output_signal_sample_count = output_signal_data_symbols * samplerate_period_per_symbol

# Output parameters to the console
# Вывод параметров
print("Частота дискретизации wav =", wav_samplerate)
print("Частота для символа 0, биты 00  =", symbol_0_frequency)
print("Частота для символа 1, биты 01  =", symbol_1_frequency)
print("Частота для символа 2, биты 10  =", symbol_2_frequency)
print("Частота для символа 3, биты 11  =", symbol_3_frequency)
print("Число периодов дискрертизации на символ =", samplerate_period_per_symbol);
print("Размер файла входных данных, байт =", input_signal_length_bytes)
print("Размер файла входных данных, бит =", input_signal_length_bits)
print("Размер несущей, символов =", carrier_symbol_size)
print("Значение несущей несущей, символ =", carrier_symbol_value)
print("Размер преамбулы, символов =", preamble_symbol_size)
print("Размер постамбулы, символов =", postamble_symbol_size)
print("Размер слова синхронизации, символов =", synchronization_word_symbol_size)
print("Значение символа слова синхронизации =", synchronization_word_symbol_value)
print("Размер выходных данных с преамбулой и словом синхронизации, символов =", output_signal_data_symbols)
print("Число отсчётов выходного wav файла =", output_signal_sample_count)

# Calculate an array with sinusoid samples
# Формируем отсчёты синусоид четырёх частот для символов
symbol_sin_samples_index = np.arange(samplerate_period_per_symbol)
symbol_0_sin_samples = np.sin(2 * np.pi * (symbol_0_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_1_sin_samples = np.sin(2 * np.pi * (symbol_1_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_2_sin_samples = np.sin(2 * np.pi * (symbol_2_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)
symbol_3_sin_samples = np.sin(2 * np.pi * (symbol_3_frequency / transmit_symbol_rate) * symbol_sin_samples_index / samplerate_period_per_symbol)

# Empty array for signal
# Формруем выходные данные, пока всё 0
output_signal = np.linspace(0, 0, int(output_signal_sample_count))

# Byte counter and bit counter for input file
# Счётчик байт и счётчик бит
input_byte_count = 0
input_bit_count = 0
output_symbol_count = 0

# Byte value and bit value, symbol value
# Значение байта, значение бит и значение символа
byte_value = 0
pair_bits_value = 0
symbol_value = 0

# Phase counter. Sine sample Counter
# Счётчик фазы
phase_cnt = 0

# Creating a QFSK signal
# Формируем выходной сигнал согласно битам, используя QFSK
for i in range(int(output_signal_sample_count)):
    # In the beginning, we make samples of the carrier frequency
    # Формируем несущую
    if output_symbol_count <= carrier_symbol_size:
        # Carrier signal frequency value
        # Частота зависит от значения бита
        if carrier_symbol_value == 0:
            symbol_value = 0
        elif carrier_symbol_value == 1:
            symbol_value = 1
        elif carrier_symbol_value == 2:
            symbol_value = 2
        else:
            symbol_value = 3

    # Make preamble, alternating frequency
    # Формируем биты преамбулы
    if (output_symbol_count > carrier_symbol_size) and (output_symbol_count < carrier_symbol_size + preamble_symbol_size):
        # Signal frequency alternating
        # Значение бита преамбулы, чередуем символы
        if phase_cnt == 0:
            symbol_value += 1
            if symbol_value >= 4:
                symbol_value = 0

    # Make sync word
    # Символы слова синхронизации
    if (output_symbol_count >= carrier_symbol_size + preamble_symbol_size) and (output_symbol_count < (carrier_symbol_size + preamble_symbol_size + synchronization_word_symbol_size)):
        if phase_cnt == 0:
            symbol_value = synchronization_word_symbol_value

    # Read input data bytes/bits
    # Биты входных данных
    if (output_symbol_count >= (carrier_symbol_size + preamble_symbol_size + synchronization_word_symbol_size)) and  (output_symbol_count < (carrier_symbol_size + preamble_symbol_size + synchronization_word_symbol_size + input_signal_length_symbols)):
        # Read one byte when phase counter is 0
        # Байт из входного файла читаем когда счётчик фазы равен 0
        if phase_cnt == 0:
            byte_value = input_signal_data[input_byte_count]

            # Get two bits from byte, make symbol
            # Два бита из байта
            pair_bits_value = (byte_value >> input_bit_count) & 0x03
            if pair_bits_value == 0:
                symbol_value = 0
            elif pair_bits_value == 1:
                symbol_value = 1
            elif pair_bits_value == 2:
                symbol_value = 2
            else:
                symbol_value = 3

            # Bits counter and bytes counter
            # Счётчики бит и байт
            input_bit_count += 2
            if input_bit_count >= 8:
                input_bit_count = 0
                input_byte_count += 1

    # Postamble
    # Постамбула
    if output_symbol_count >= (carrier_symbol_size + preamble_symbol_size + synchronization_word_symbol_size + input_signal_length_symbols):
        # Signal frequency inversion
        # Значение бита постамбулы, чередуем символы
        if phase_cnt == 0:
            symbol_value += 1
            if symbol_value >= 4:
                symbol_value = 0

    # Frequency depends on symbol value
    # Частота зависит от значения символв
    if symbol_value == 0:
        output_signal[i] = symbol_0_sin_samples[phase_cnt]
    elif symbol_value == 1:
        output_signal[i] = symbol_1_sin_samples[phase_cnt]
    elif symbol_value == 2:
        output_signal[i] = symbol_2_sin_samples[phase_cnt]
    else:
        output_signal[i] = symbol_3_sin_samples[phase_cnt]

    # Increment phase counter
    # Счётчик фазы
    phase_cnt += 1
    if phase_cnt >= samplerate_period_per_symbol:
        phase_cnt = 0
        output_symbol_count += 1

# Save wav file
# Сохраним в файл
output_signal *= 32765
output_signal_int = np.int16(output_signal)
wavfile.write(wav_file_name, wav_samplerate, output_signal_int)