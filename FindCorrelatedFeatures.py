#import glob
import csv
import pickle
import numpy as np
from statistics import mean
from statistics import median

def FindCorrelatedFeatures(feature_dict_filter, molecule_dict_filter, activity_list):

    feature_dict_filter_corr = {}
    for feature, index in feature_dict_filter.items():
        feature_dict_filter_corr.update({feature: [index]})

    molecule_features = []
    for i in range(0,16):
        molecule_features.append([])
    for i in activity_list:
        molecule_activity = []
        #molecule_features[i] = []
        for molecule, values in molecule_dict_filter.items():
            molecule_activity.append(values[1][i])
            for j in values[0]:
                molecule_features[i].append(j)
        correlation = np.corrcoef(molecule_activity, molecule_features[i])

    return molecule_dict_filter