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
i = 1
dictionary = {}
with open("Label/data.csv") as tsv:
    for line in csv.reader(tsv):
        dictionary[int(float(line[0]))] = line[3]



print(dictionary)
