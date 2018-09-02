from scipy.fftpack import fft, ifft
from scipy import signal as sig
from matplotlib import pyplot as plt
import numpy as np


def plotfft(signal, fs):
    T = 1 / fs  # Sampling Time
    yf = fft(signal)  # Fourier Transform
    xf = np.linspace(0.0, 1.0 / (2 * T), len(signal) // 2)  # Frequency Axis
    plt.plot(xf, 2. / len(signal) * np.abs(yf[0:len(signal) // 2]))


def mfreqz(b, a=1, fs=1):
    w, h = sig.freqz(b, a)
    h_dB = 20 * np.log10(abs(h))
    plt.subplot(211)
    plt.plot(fs * w / (2 * max(w)), h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
    plt.plot(fs * w / (2 * max(w)), h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)

