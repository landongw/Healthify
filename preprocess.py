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


def mfreqz(b,a=1,fs=1):
    w,h = sig.freqz(b,a)
    h_dB = 20 * np.log10 (abs(h))
    plt.subplot(211)
    plt.plot(fs*w/(2*max(w)),h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
    plt.plot(fs*w/(2*max(w)),h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)

#  LOAD DATA ----------------------------------------

f = open("./Our_Device/mohammadreza_salehi/data.txt", "r")
reader = csv.reader(f)
signal1 = []
signal2 = []
for row in reader:
    sp = (row[0].split("\t"))
    signal1.append(float(sp[0]))
    signal2.append(float(sp[1]))

#  PLOT SIGNALS -------------------------------------

plt.figure()
plt.subplot(211)
plt.plot(signal1)
plt.xlim([0, 400])
plt.ylabel("Signal 1")
plt.subplot(212)
plt.plot(signal2)
plt.xlim([0, 400])
plt.ylabel("Signal 2")
plt.show()

# PLOT FOURIER TRANSFORMS ----------------------------

ECG = signal2
fs = 120
ECG = ECG - np.mean(ECG)  # Remove DC component
plotfft(ECG, fs)

# FILTER ---------------------------------------------

h = sig.firwin(101, [5, 40], width=None, window='hamming', pass_zero=False, scale=True, fs=fs)

mfreqz(h,1,fs)   # bode plot

ECG_filtered = sig.fftconvolve(h, ECG)

# before and after filtering
plt.figure()
plotfft(ECG, fs)
plotfft(ECG_filtered, fs)
plt.legend(['Original','Filtered'])
plt.show()


plt.figure()
plt.subplot(211)
plt.plot(ECG)
plt.xlim([0, 400])
plt.title('Original Signal')
plt.subplot(212)
plt.plot(ECG_filtered)


plt.title('Filtered Signal')
plt.xlim([0, 400])
plt.subplots_adjust(hspace=0.5)
plt.show()
