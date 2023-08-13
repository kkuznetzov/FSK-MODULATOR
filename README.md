# FSK-MODULATOR
Frequency Shift Keying modulator and demodulator project

FSK modulator and demodulator
Code that implements modulation and demodulation using FSK.

The input of the modulator is a text file, the output is a wav file.
bfsk_modulator.py - FSK modulator with two frequencies.
qfsk_modulator.py - FSK modulator with four frequencies.

The demodulator input is a wav file, the output is a text file. 
bfsk_pll_type2_demodulator - BFSK demodulator, pll type 2.
qfsk_pll_type2_demodulator - QFSK demodulator, pll type 2.

Additional code:  
add_noise.py - adding noise to the signal.  
add_signal.py - adding a sine wave to the signal.

BPSK receiver debug output

![BFSK receiver debug output](https://github.com/kkuznetzov/FSK-MODULATOR/blob/master/IMG/bfsk_receive_debug.png)

BPSK receiver debug output zoomed

![BFSK receiver debug output](https://github.com/kkuznetzov/FSK-MODULATOR/blob/master/IMG/bfsk_receive_debug_zoomed.png)
