#import glob
#import csv
import pickle
import numpy as np
from statistics import mean
from statistics import median

def FindFeatures(data_directory):
    #training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet/"

    fileObject = open(data_directory+"feature_dict.pickle",'rb')
    feature_dict = pickle.load(fileObject)
    fileObject.close()

    feature_dict_stdev = {}
    for feature, index in feature_dict.items():
        if feature not in feature_dict_stdev:
            feature_dict_stdev.update({feature: [index]})

    fileObject = open(data_directory+"molecule_dict.pickle",'rb')
    molecule_dict = pickle.load(fileObject)
    fileObject.close()

    #the idea here is for each feature, to get a list of all values
    #then i can analyze the features and determine which have low or high variability
    #the features with highest variability will be best used in the model

    print("GETTING STDEV LIST")
    stdev_list = []
    for feature, index in feature_dict.items():
        #print(feature)
        feature_values = []
        for molecule, values in molecule_dict.items():
            if str(molecule_dict[molecule][0][index]).isnumeric(): #excluding None type values
                feature_values.append(molecule_dict[molecule][0][index])
        stdev = np.std(np.array(feature_values, dtype=np.float64), dtype=np.float64)
        feature_dict_stdev[feature].append(stdev)
        stdev_list.append(stdev)

    print("LEN STDEV: " + str(len(stdev_list)))
    zero_count = 0
    for feature, values in feature_dict_stdev.items():
        #print(feature)
        if values[1] == 0:
            zero_count = zero_count + 1
    print("ZERO COUNT: " + str(zero_count))
    print("MAX STDEV")
    print(max(stdev_list))
    print("AVG STDEV")
    print(mean(stdev_list))
    print("MEDIAN STDEV")
    print(median(stdev_list))
    print("")

    featPortion = int(len(feature_dict_stdev)*0.2)
    stdev_list = sorted(stdev_list, key=float, reverse=True)
    stdevThreshold = stdev_list[featPortion]

    featureIndexToRemove = []
    for feature, values in feature_dict_stdev.items():
        if values[1] < stdevThreshold:
            featureIndexToRemove.append(values[0])
    featureIndexToRemove = sorted(featureIndexToRemove, key=float, reverse=True)

    molecule_dict_filter = molecule_dict
    for i in range(len(featureIndexToRemove)):
        for molecule, values in molecule_dict_filter.items():
            molecule_dict_filter[molecule][0].pop(featureIndexToRemove[i])

    fileObject = open(data_directory+"molecule_dict_filter.pickle",'wb') # open the file for writing
    pickle.dump(molecule_dict_filter,fileObject)
    fileObject.close()

    print("DONE!")