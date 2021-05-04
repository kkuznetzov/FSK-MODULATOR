from os.path import dirname, join as pjoin
from scipy.io import wavfile
import pdb
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Имя входного wav файла
wav_file_in_name = 'qfsk_out.wav'
#wav_file_in_name = 'qfsk_out_and_noise_and_low_freq2.wav'

# Имя выходного wav файла
wav_file_out_name = 'qfsk_out_and_noise.wav'

# Требуемое соотношение сигнал/шум
traget_signal_noise_ration_db = 50

# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_in_name)
input_signal_length = input_signal_data.shape[0]

# Вывод параметров файла
print("Частота дискретизации =", input_signal_samplerate)
print("Число отсчётов =", input_signal_length)
print("Длительность секунд =", input_signal_length/input_signal_samplerate)

# Мощность сигнала
input_signal_data = input_signal_data/32767
input_signal_data_watts = input_signal_data ** 2

# Средняя мощность сигнала
input_signal_data_avg_watts = np.mean(input_signal_data_watts) * 32767
input_signal_data_avg_watts_db = 10 * np.log10(input_signal_data_avg_watts)

# Вывод параметров сигналов
print("Средняя мощность сигнала, дБ =", input_signal_data_avg_watts_db)

# Значение шума
noise_avg_db = input_signal_data_avg_watts_db - traget_signal_noise_ration_db
noise_avg_watts = 10 ** (noise_avg_db / 10)

# Вывод параметров шума
print("Средняя мощность шума, дБ =", noise_avg_db)

# Генерация белого шума
mean_noise = 0
noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(input_signal_data_watts))

# Добавим шум к сигналу
output_signal_noise_data = input_signal_data + noise_volts

# Масштабируем отсчёты, если превысят 1, что бы не было переполнения в int16
scale_value = 1
for i in range(int(input_signal_length)):
    if abs(output_signal_noise_data[i]) > scale_value:
       scale_value = output_signal_noise_data[i]
output_signal_noise_data = output_signal_noise_data/scale_value

#Запишем сигнал и шум
output_signal_noise_data *= 32767
output_signal_noise_data_int = np.int16(output_signal_noise_data)
wavfile.write(wav_file_out_name, input_signal_samplerate, output_signal_noise_data_int)
