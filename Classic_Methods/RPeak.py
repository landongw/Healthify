from matplotlib import pyplot as plt
from biosppy.signals import ecg as ecg



def QRS_detection(signal, fs):
    # R Peaks : P.S. Hamilton, "Open Source ECG Analysis Software Documentation", E.P.Limited, 2002
    rpeaks, = ecg.hamilton_segmenter(signal=signal, sampling_rate=fs)
    array = []
    for x in rpeaks:
        array.append(x)
    #plt.figure()
    #plt.plot(signal[1:1000])
    #for xc in array:
    #    if xc < 1000:
    #        plt.axvline(x=xc, color='red')
    #plt.show()

    return array

