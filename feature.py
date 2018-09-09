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

#plt.figure()
#plt.plot(ECG_filtered[1:10000])
#for xc in rpeaks:
#    if xc < 10000:
#        plt.axvline(x=xc, color='red')

#plt.title("R-Peaks")
#plt.show()

# HeartBeat Extraction
j = 0
template = []

for i in rpeaks:
    template.append(ECG_filtered[int(np.floor(i - 0.2 * fs)): int(np.floor(i + 0.4 * fs))])
    j += 1
#plt.figure()

#for i in range(10):
#    plt.plot(template[i])
# plt.show()

RR = []
for i in range(len(rpeaks) - 2):
    RR.append(rpeaks[i + 2] - rpeaks[i + 1])

seperated = []
N = 30
number = int(np.floor(len(RR) / N))
for i in range(number):
    seperated.append(RR[i:i + N])

####### RR Features #######
mean_Feature = []
std_Feature = []

for i in range(len(seperated)):
    mean_Feature.append(np.mean(seperated[i]))
    std_Feature.append(np.mean(seperated[i]))

import numpy

####### RR_Seperated Features #######
RR_seperated = numpy.zeros((number, N - 1))

for i in range(number):
    for j in range(N - 1):
        RR_seperated[i, j] = seperated[i][j + 1] - seperated[i][j]
RR_seperated_mean_Feature = []
RR_seperated_std_Feature = []
for i in range(number):
    RR_seperated_mean_Feature.append(np.mean(RR_seperated[i, :]))
    RR_seperated_std_Feature.append(np.mean(RR_seperated[i, :]))

####### pNN Feature #######
pNN50_Feature = []
pNN10_Feature = []
pNN5_Feature = []

for i in range(number):
    pNN50_Feature.append([abs(x) for x in RR_seperated[i][:] if abs(x) > 50 * fs / 1000])
    pNN10_Feature.append([abs(x) for x in RR_seperated[i][:] if abs(x) > 10 * fs / 1000])
    pNN5_Feature.append([abs(x) for x in RR_seperated[i][:] if abs(x) > 5 * fs / 1000])


####### Frequncy Energy Ratio Feature #######
h_high = sig.firwin(101, [0.15, 0.4], width=None, window='hamming', pass_zero=False, scale=True, fs=1)
h_low = sig.firwin(101, [0.04, 0.15], width=None, window='hamming', pass_zero=False, scale=True, fs=1)

RR_high = sig.fftconvolve(h_high, RR)
RR_low = sig.fftconvolve(h_low, RR)

RR_Energy_Ratio_Feature = np.sum(RR_high ** 2) / np.sum(RR_low ** 2)

####### Poincare Map Feature #######
x = np.array(RR[0:-1])
y = np.array(RR[1:])

SD1 = np.std(np.abs(x - y))
SD2 = np.std(np.abs(x - y + 2 * np.mean(RR)))

SD_Ratio_Feature = SD1 / SD2

####### Approximate Entropy Feature #######
ApEn_Feature = []

for i in range(number):
    ApEn_Feature.append(ApEn(np.array(seperated[i]), 2, 2 * np.std(seperated[i])))

####### Spectral Entropy Feature #######
RR_fft = np.abs(ffttools.fft(RR))
RR_fft[0] = 0  # Remove
ProbDens = 2. / len(signal) * np.abs(RR_fft[0:len(RR_fft) // 2])
ProbDens = ProbDens / np.sum(ProbDens)
SpEn_Feature = stats.entropy(ProbDens, base=2)
Lya_Exp_Feature = nolds.lyap_r(np.array(RR), emb_dim=2, lag=1, min_tsep=10, tau=1)

####### Detrended Fluctuation Analysis Feature #######
DFA_Slope_Feature = nolds.dfa(np.array(RR))

####### Sequential Trend Analysis Feature #######
DeltaRR = np.array(RR[1:]) - np.array(RR[0:-1])
print(DeltaRR)

PosSeqTrend_Feature = 0
NegSeqTrend_Feature = 0
for i in range(len(DeltaRR) - 1):
    if DeltaRR[i] > 0 and DeltaRR[i + 1] > 0:
        PosSeqTrend_Feature += 1
    if DeltaRR[i] < 0 and DeltaRR[i + 1] < 0:
        NegSeqTrend_Feature += 1

