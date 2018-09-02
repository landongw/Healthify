from biosppy.signals import ecg
from biosppy import storage
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal as sig
import csv

def plotfft(signal, fs):
    T = 1 / fs  # Sampling Time
    yf = fft(signal)  # Fourier Transform
    xf = np.linspace(0.0, 1.0 / (2 * T), len(signal) // 2)  # Frequency Axis
    plt.plot(xf, 2. / len(signal) * np.abs(yf[0:len(signal) // 2]))


#  LOAD DATA ----------------------------------------
f = open("./Our_Device/morteza_abolghasemi/data.txt", "r")
reader = csv.reader(f)
signal1 = []
signal2 = []
for row in reader:
    sp = (row[0].split("\t"))
    signal1.append(float(sp[0]))
    signal2.append(float(sp[1]))

fs = 120
out = ecg.ecg(signal=signal2[1:1000], sampling_rate=fs, show=True)

