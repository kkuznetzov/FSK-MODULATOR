from os.path import dirname, join as pjoin
from scipy.io import wavfile
import pdb
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Частота и амплитуда добавляемого сигнала
signal_frequency = 500
signal_amplitude = 1

# Имя входного wav файла
#wav_file_in_name = 'qfsk_out_and_noise.wav'
wav_file_in_name = 'bfsk_out_and_noise.wav'

# Имя выходного wav файла
wav_file_out_name = 'bfsk_out_and_noise_and_low_freq.wav'

# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_in_name)
input_signal_length = input_signal_data.shape[0]

# Число отсчётов на период сигнала
sample_per_period = input_signal_samplerate / signal_frequency

# Вывод параметров файла
print("Частота дискретизации =", input_signal_samplerate)
print("Число отсчётов =", input_signal_length)
print("Длительность секунд =", input_signal_length/input_signal_samplerate)
print("Частота добавляемого сигнала", signal_frequency)
print("Число отсчётов на период сигнала =", sample_per_period)

# Формируем отсчёты косинуса
cos_samples = np.arange(sample_per_period)
cos_signal = np.sin(2 * np.pi * cos_samples / sample_per_period) * signal_amplitude

# Масштабируем отсчёты входного сигнала, что бы были в интервале от -1 до 1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(input_signal_data[i]) > scale_value:
        scale_value = input_signal_data[i]
input_signal_data = input_signal_data/scale_value

# Формруем выходные данные, пока всё 0
output_signal = np.linspace(0, 0, int(input_signal_length))

# Добавляем сигнал к входному сигнала
phase_cnt = 0 # Счётчик фазы косинуса
for i in range(int(input_signal_length)):
    output_signal[i] = input_signal_data[i] + cos_signal[phase_cnt]
    
    # Счётчик фазы косинуса
    # Здесь же счётчик бит
    phase_cnt += 1
    if(phase_cnt >= sample_per_period):
        phase_cnt = 0

# Масштабируем отсчёты выходного сигнала, что бы были в интервале от -1 до 1
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(output_signal[i]) > scale_value:
        scale_value = output_signal[i]
output_signal = output_signal/scale_value

# Сохраним в файл
output_signal*= 32767
ountput_signal_int = np.int16(output_signal)
wavfile.write(wav_file_out_name, input_signal_samplerate, ountput_signal_int)
