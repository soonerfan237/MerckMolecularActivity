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
            #print(correlation[i][j])
        correlation_sorted = np.array(correlation[i])
        correlation_sorted = np.sort(np.absolute(correlation_sorted))
        correlation_cutoff = correlation_sorted[int(5*len(correlation_sorted)/10)]

        featureIndexToRemove = []
        correlation_abs = np.absolute(np.array(correlation[i]))
        #test = range(len(correlation_abs)-1,0)
        for j in reversed(range(len(correlation_abs))):
            if correlation_abs[j] < correlation_cutoff:
                featureIndexToRemove.append(j)
        featureIndexToRemove = sorted(featureIndexToRemove, key=float, reverse=True)

        for i in range(len(featureIndexToRemove)):
            for molecule, values in molecule_dict_filter.items():
                molecule_dict_filter[molecule][0].pop(featureIndexToRemove[i])

        print("REMOVED FEATURES: " + str(len(featureIndexToRemove)))
        print("REMAINING FEATURES: " + str(len(next(iter(molecule_dict_filter.values()))[0])))

    return molecule_dict_filter