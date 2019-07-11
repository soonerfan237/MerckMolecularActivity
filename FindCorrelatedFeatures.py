#import glob
import csv
import pickle
import numpy as np
from statistics import mean
from statistics import median

def FindCorrelatedFeatures(feature_dict_filter, molecule_dict_filter, activity_list):

    correlation = []
    for i in range(0,16):
        correlation.append([])
        for feature, index in feature_dict_filter.items():
            correlation[i].append(0)

    feature_dict_filter_corr = {}
    for feature, index in feature_dict_filter.items():
        feature_dict_filter_corr.update({feature: [index]})

    molecule_features = []
    for i in range(len(feature_dict_filter)):
        molecule_features.append([])
    for i in activity_list:
        molecule_activity = []
        #molecule_features[i] = []
        for molecule, values in molecule_dict_filter.items():
            molecule_activity.append(values[1][i])
            for j in range(len(feature_dict_filter)):
                molecule_features[j].append(values[0][j])
        molecule_activity = np.array(molecule_activity)
        molecule_activity = molecule_activity.astype(float)
        for j in range(len(feature_dict_filter)):
            molecule_feature = np.array(molecule_features[j])
            molecule_feature = molecule_feature.astype(float)
            correlation[i][j] = np.corrcoef(molecule_activity, molecule_feature)[0,1]
            print(correlation[i][j])

    return molecule_dict_filter