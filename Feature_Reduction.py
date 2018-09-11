from sklearn.decomposition import PCA
import csv
import os
import feature
import numpy as np

location = '/media/bemoniri/d08ff677-13e1-4052-b55d-6337b6f8c634/PTBDB-DataBase/'  # Database location
header_data = './Label/data.csv'  # Labels location

dictionary = {}  # Patient's Disease Dictionary
with open("Label/data.csv") as tsv:
    for line in csv.reader(tsv):
        dictionary[int(float(line[0]))] = line[3]

patients = os.listdir(location)  # List of patients
patients = sorted(patients)
Labels = []
Feature = []

error = 0
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
            try:
                a = feature.Feature_Exreaction(raw_signal, fs)
                Feature.append(a[0])
                Labels.append(dictionary[int(patient[7:])])
            except:
                error += 1

print(len(Feature))
print(len(Labels))
print(len(Feature[0]))
print(len(Feature[10]))
print(len(Feature[100]))
print(len(Feature[200]))