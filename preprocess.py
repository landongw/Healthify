# Our Libraries
import myplottools
import RPeak


# Third-Party Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import csv


#  LOAD DATA ----------------------------------------
f = open("./Our_Device/mohammadreza_salehi/data.txt", "r")
reader = csv.reader(f)
signal1 = []
signal2 = []
for row in reader:
    sp = (row[0].split("\t"))
    signal1.append(float(sp[0]))
    signal2.append(float(sp[1]))


# PLOT FOURIER TRANSFORMS ----------------------------
ECG = signal2
fs = 120
ECG = ECG - np.mean(ECG)  # Remove DC component
myplottools.plotfft(ECG, fs)


# FILTER ---------------------------------------------
h = sig.firwin(101, [3, 45], width=None, window='hamming', pass_zero=False, scale=True, fs=fs)
myplottools.mfreqz(h, 1, fs)  # bode plot
ECG_filtered = sig.fftconvolve(h, ECG)



# Plot Filter Results
plt.figure()
myplottools.plotfft(ECG, fs)
myplottools.plotfft(ECG_filtered, fs)
plt.legend(['Original', 'Filtered'])


plt.figure()
plt.subplot(211)
plt.plot(ECG)
plt.xlim([0, 2000])
plt.title('Original Signal')
plt.subplot(212)
plt.plot(ECG_filtered)
plt.title('Filtered Signal')
plt.xlim([0, 2000])
plt.subplots_adjust(hspace=0.5)
plt.show()


# R Peaks : P.S. Hamilton, "Open Source ECG Analysis Software Documentation", E.P.Limited, 2002

rpeaks_new = RPeak.QRS_detection(signal=ECG_filtered[1:], sample_rate=fs, max_bpm=300)

plt.figure()
plt.plot(ECG_filtered[1:1000])
for xc in rpeaks_new:
    if xc < 1000:
        plt.axvline(x=xc, color='red')

plt.title("R-Peaks")
plt.show()

# HeartBeat Extraction
j = 0
template = []

for i in rpeaks_new:
    template.append(ECG_filtered[int(np.floor(i-0.2*fs)): int(np.floor(i+0.4*fs))])
    j += 1
plt.figure()

for i in range(10):
    plt.plot(template[i])
plt.show()

