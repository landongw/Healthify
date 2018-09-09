# Our Libraries
import myplottools
import RPeak

# Third-Party Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import scipy.fftpack as ffttools
from scipy import stats
import csv
import nolds

def ApEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))


# Laoding the signal

signal = []
with open("patient233/s0457_re-Data.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        signal.append(float(line[1]))  # ECG signal from "i" electrode

fs = 1000

# Frequency Filter

h = sig.firwin(101, [4, 50], width=None, window='hamming', pass_zero=False, scale=True, fs=fs)
# myplottools.mfreqz(h, 1, fs)  # bode plot
ECG_filtered = sig.fftconvolve(h, signal)

# R-Peak Detection

rpeaks = RPeak.QRS_detection(signal=ECG_filtered, sample_rate=fs, max_bpm=300)

plt.figure()
plt.plot(ECG_filtered[1:10000])
for xc in rpeaks:
    if xc < 10000:
        plt.axvline(x=xc, color='red')

plt.title("R-Peaks")
# plt.show()

# HeartBeat Extraction
j = 0
template = []

for i in rpeaks:
    template.append(ECG_filtered[int(np.floor(i - 0.2 * fs)): int(np.floor(i + 0.4 * fs))])
    j += 1
plt.figure()

for i in range(10):
    plt.plot(template[i])
# plt.show()

HRV = []
for i in range(len(rpeaks) - 2):
    HRV.append(rpeaks[i + 2] - rpeaks[i + 1])

seperated = []
N = 30
number = int(np.floor(len(HRV) / N))
for i in range(number):
    seperated.append(HRV[i:i + N])

feature_mean = []
feature_std = []

for i in range(len(seperated)):
    feature_mean.append(np.mean(seperated[i]))
    feature_std.append(np.mean(seperated[i]))

import numpy

HRV_seperated = numpy.zeros((number, N - 1))

for i in range(number):
    for j in range(N - 1):
        HRV_seperated[i, j] = seperated[i][j + 1] - seperated[i][j]
feature_hmean = []
feature_hstd = []
for i in range(number):
    feature_hmean.append(np.mean(HRV_seperated[i, :]))
    feature_hstd.append(np.mean(HRV_seperated[i, :]))

pNN50 = []
pNN10 = []
pNN5 = []

for i in range(number):
    pNN50.append([abs(x) for x in HRV_seperated[i][:] if abs(x) > 50 * fs / 1000])
    pNN10.append([abs(x) for x in HRV_seperated[i][:] if abs(x) > 10 * fs / 1000])
    pNN5.append([abs(x) for x in HRV_seperated[i][:] if abs(x) > 5 * fs / 1000])

h_high = sig.firwin(101, [0.15, 0.4], width=None, window='hamming', pass_zero=False, scale=True, fs=1)
h_low = sig.firwin(101, [0.04, 0.15], width=None, window='hamming', pass_zero=False, scale=True, fs=1)

HRV_high = sig.fftconvolve(h_high, HRV)
HRV_low = sig.fftconvolve(h_low, HRV)

HRV_Energy_Ratio = np.sum(HRV_high ** 2) / np.sum(HRV_low ** 2)
x = np.array(HRV[0:-1])
y = np.array(HRV[1:])

SD1 = np.std(np.abs(x - y))
SD2 = np.std(np.abs(x - y + 2 * np.mean(HRV)))

SD_Ratio = SD1 / SD2
ApEn_Feature = []

for i in range(number):
    ApEn_Feature.append(ApEn(np.array(HRV_seperated[i]), 2, 2 * np.std(HRV_seperated[i])))

HRV_fft = np.abs(ffttools.fft(HRV))
HRV_fft[0] = 0  # Remove
ProbDens = 2. / len(signal) * np.abs(HRV_fft[0:len(HRV_fft) // 2])
ProbDens = ProbDens / np.sum(ProbDens)
SpEn_Feature = stats.entropy(ProbDens, base=2)
Lya_Exp = nolds.lyap_r(np.array(HRV), emb_dim=2, lag=1, min_tsep=10, tau=1)

plt.figure()
nolds.lyap_r(np.array(HRV), emb_dim=2, lag=1, min_tsep=10, tau=1, debug_plot=True)
plt.show()

nolds.dfa(HRV)