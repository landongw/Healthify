# You can use "Signals" and "Labels" or use the csv file that this code generates

from matplotlib import pyplot as plt
from scipy import signal as sig
import csv
import os

location = '/media/bemoniri/d08ff677-13e1-4052-b55d-6337b6f8c634/PTBDB-DataBase/'  # Database location
header_data = './Label/data.csv'  # Labels location

dictionary = {}  # Patient's Disease Dictionary
with open("Label/data.csv") as tsv:
    for line in csv.reader(tsv):
        dictionary[int(float(line[0]))] = line[3]

patients = os.listdir(location)  # List of patients
patients = sorted(patients)
# patients = ['patient100']
raw_signal = []
Signals = []
Labels = []


for patient in patients:  # for each patient (avoid RECORDS.txt)
    print(patient)
    if os.listdir(location + patient):
        files = os.listdir(location + patient)  # load the patient's files
        for file in files:  # for each file
            if file.find("Data") != -1:  # if it is not a header file
                print(file)

                raw_signal = []
                with open(location + patient + "/" + file) as tsv:  # load the file
                    for line in csv.reader(tsv, dialect="excel-tab"):  # load the required signal
                        raw_signal.append(float(line[1]))  # ECG signal from  electrode "i"
                        fs = 1000

                h = sig.firwin(101, [10, 50], width=None, window='hamming', pass_zero=False, scale=True, fs=fs)
                ECG_filtered = sig.fftconvolve(h, raw_signal)  # filter
                ECG_filtered = ECG_filtered[0::10]  # Down sample
                fs = fs / 10   # fs = 100 Hz

                # Write in database
                size = int(len(ECG_filtered) / (3 * fs))

                for i in range(size):
                    Signals.append(ECG_filtered[int(i * fs * 3):   int(fs * (i + 1) * 3)])  # Patients Signal
                    Labels.append(dictionary[int(patient[7:])])



print(Signals)
print(Labels)
print("Number of Signals:")
print(len(Signals))
print(len(Labels))

with open("demofile.txt", "w+") as f:
    for i in range(len(Labels)):
        temp_sig = Signals[i]
        temp_label = Labels[i]
        f.write(temp_label+",")
        for j in range(len(temp_sig)):
            f.write(str(temp_sig[j])+',')
        f.write("\n")

plt.figure()
plt.plot(Signals[0])
plt.show()
