import glob
import csv
import pickle
import numpy as np
from statistics import mean
from statistics import median

training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet/"

fileObject = open(training_directory+"feature_dict",'rb')
feature_dict = pickle.load(fileObject)
fileObject.close()

feature_dict_stdev = {}
for feature, index in feature_dict.items():
    if feature not in feature_dict_stdev:
        feature_dict_stdev.update({feature: [index]})
    #feature_dict_stdev[feature] = [index,None]

fileObject = open(training_directory+"molecule_dict",'rb')
molecule_dict = pickle.load(fileObject)
fileObject.close()

#data_set = []

#the idea here is for each feature, to get a list of all values
#then i can analyze the features and determine which have low or high variability
#the features with highest variability will be best used in the model

#stdev_list = []
for feature, index in feature_dict.items():
    feature_values = []
    for molecule, values in molecule_dict.items():
        if str(molecule_dict[molecule][0][index]).isnumeric():
            feature_values.append(molecule_dict[molecule][0][index])
    stdev = np.std(np.array(feature_values, dtype=np.float64), dtype=np.float64)
    feature_dict_stdev[feature].append(stdev)

    #stdev_list.append(np.std(np.array(feature_values, dtype=np.float64), dtype=np.float64))
    #print(feature, np.std(np.array(values, dtype=np.float64), dtype=np.float64))

print("LEN STDEV: " + str(len(feature_dict_stdev)))
zero_count = 0
for feature, values in feature_dict_stdev:
    if values[1] == 0:
        zero_count = zero_count + 1
print("ZERO COUNT: " + str(zero_count))
#print("MAX STDEV")
#print(max(stdev_list))
#print("AVG STDEV")
#print(mean(stdev_list))
#print("MEDIAN STDEV")
#print(median(stdev_list))
#print("")

featToKeep = len(feature_dict_stdev)*0.2


