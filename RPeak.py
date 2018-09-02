import numpy as np
import pywt as pw


def QRS_detection(signal, sample_rate, max_bpm):
    ## Stationary Wavelet Transform
    coeffs = pw.swt(signal, wavelet="haar", level=2, start_level=0, axis=-1)
    d2 = coeffs[1][1]  ##2nd level detail coefficients

    ## Threhold the detail coefficients
    avg = np.mean(d2)
    std = np.std(d2)
    sig_thres = [abs(i) if abs(i) > 2.0 * std else 0 for i in d2 - avg]

    ## Find the maximum modulus in each window
    window = int((60.0 / max_bpm) * sample_rate)
    sig_len = len(signal)
    n_windows = int(sig_len / window)
    modulus, qrs = [], []

    ##Loop through windows and find max modulus
    for i in range(n_windows):
        start = i * window
        end = min([(i + 1) * window, sig_len])
        mx = max(sig_thres[start:end])
        if mx > 0:
            modulus.append((start + np.argmax(sig_thres[start:end]), mx))

    ## Merge if within max bpm
    merge_width = int((0.2) * sample_rate)
    i = 0
    while i < len(modulus) - 1:
        ann = modulus[i][0]
        if modulus[i + 1][0] - modulus[i][0] < merge_width:
            if modulus[i + 1][1] > modulus[i][1]:  # Take larger modulus
                ann = modulus[i + 1][0]
            i += 1

        qrs.append(ann)
        i += 1
        ## Pin point exact qrs peak
    window_check = int(sample_rate / 6)
    # signal_normed = np.absolute((signal-np.mean(signal))/(max(signal)-min(signal)))
    r_peaks = [0] * len(qrs)

    for i, loc in enumerate(qrs):
        start = max(0, loc - window_check)
        end = min(sig_len, loc + window_check)
        wdw = np.absolute(signal[start:end] - np.mean(signal[start:end]))
        pk = np.argmax(wdw)
        r_peaks[i] = start + pk


    return r_peaks