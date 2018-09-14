import csv
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

dictionary = {}  # Patient's Disease Dictionary
Labels = []
Signals = []
kappa = 0
with open("feature_list.txt") as tsv:
    for line in csv.reader(tsv):
        Labels.append(line[0])
        Signals.append([float(x) for x in line[1:-1]])

Signals = Signals[1::3]
Labels = Labels[1::3]

healthy = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Healthy control']
ill1 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Cardiomyopathy']
ill2 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Palpitation']
ill3 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Hypertrophy']
ill4 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Dysrhythmia']
ill5 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Bundle branch block']
ill6 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Myocarditis']
ill7 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Valvular heart disease']
ill8 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Heart failure (NYHA 2)']
ill9 = [Signals[x] for x in range(len(Signals)) if Labels[x] == 'Myocardial infarction']

print(list(set(Labels)))
print(len(healthy))
print(len(ill1), len(ill2), len(ill3), len(ill4), len(ill5), len(ill6), len(ill7), len(ill8), len(ill9))

# What we wish to classify
comp_feature = [Signals[x] for x in range(len(Signals)) if
                Labels[x] == 'Myocardial infarction' or Labels[x] == 'Healthy control']
comp_labels = [Labels[x] for x in range(len(Signals)) if
               Labels[x] == 'Myocardial infarction' or Labels[x] == 'Healthy control']


# Feature Reduction
X = np.array(comp_feature)
pca = PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
reduced = pca.fit_transform(X)


clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, reduced, comp_labels, cv=2)
print(scores)
