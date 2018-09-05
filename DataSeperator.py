# Our Libraries
import myplottools
import RPeak
from mpl_toolkits.mplot3d import Axes3D

# Third-Party Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import csv

# Laoding the signal

signal = []
with open("patient233/s0457_re-Data.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        signal.append(float(line[1]))   # ECG signal from "i" electrode

fs = 1000

# Frequency Filter
h = sig.firwin(101, [10, 50], width=None, window='hamming', pass_zero=False, scale=True, fs=fs)
#myplottools.mfreqz(h, 1, fs)  # bode plot
ECG_filtered = sig.fftconvolve(h, signal)

Signals = np.zeros((int(len(ECG_filtered)/(10*fs)), 10*fs))
size = int(len(ECG_filtered)/(10*fs))
for i in range(size):
    Signals[i, :] = ECG_filtered[i*10*fs:10*fs*(i+1)]


