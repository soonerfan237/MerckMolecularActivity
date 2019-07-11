#import glob
import csv
import pickle
import numpy as np
from statistics import mean
from statistics import median

def FindFeatures(data_directory, feature_dict, molecule_dict):
    #training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet/"

    #fileObject = open(data_directory+"feature_dict.pickle",'rb')
    #feature_dict = pickle.load(fileObject)
    #fileObject.close()

    # molecule_dict = {}
    # with open(data_directory+"molecule_dict.csv", newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    #     try:
    #         next(reader)[1:]  # skipping header line
    #     except Exception:
    #         pass
    #     #for i in range(len(header)):
    #     for row in reader:
    #         molecule_name = row[0]
    #         molecule_activity = row[1:16]
    #         molecule_features = row[17:]
    #         #print("# OF FEATURES FROM CSV: " + str(len(molecule_features)))
    #         #molecule_dict.update({molecule_name : [molecule_features,molecule_activity]})

    feature_dict_stdev = {}
    for feature, index in feature_dict.items():
        if feature not in feature_dict_stdev:
            feature_dict_stdev.update({feature: [index]})

    # fileObject = open(data_directory+"molecule_dict1.pickle",'rb')
    # molecule_dict = pickle.load(fileObject)
    # fileObject.close()
    # fileObject = open(data_directory + "molecule_dict2.pickle", 'rb')
    # molecule_dict.update(pickle.load(fileObject))
    # fileObject.close()

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

    feature_dict_filter = {}
    for feature, values in feature_dict_stdev.items():
        if values[0] not in featureIndexToRemove:
            feature_dict_filter.update({feature: values[0]})

    #fileObject = open(data_directory+"molecule_dict_filter.pickle",'wb') # open the file for writing
    #pickle.dump(molecule_dict_filter,fileObject)
    #fileObject.close()
    feature_list = []
    for feature, value in feature_dict_filter.items():
        feature_list.append(feature)

    with open(data_directory+"molecule_dict_filter.csv", mode='w') as molecule_dict_filter_csv:
        molecule_dict_csv_writer = csv.writer(molecule_dict_filter_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_activity_list = ["ACT0","ACT1","ACT2","ACT3","ACT4","ACT5","ACT6","ACT7","ACT8","ACT9","ACT10","ACT11","ACT12","ACT13","ACT14","ACT15"]
        molecule_dict_csv_writer.writerow(["molecule"]+header_activity_list+feature_list)
        for molecule, features in molecule_dict_filter.items():
            molecule_dict_csv_writer.writerow([molecule]+features[1]+features[0])

    print("TOTAL FEATURES: " + str(len(feature_dict)))
    print("REMOVED FEATURES: " + str(len(featureIndexToRemove)))
    print("REMAINING FEATURES: " + str(len(next(iter(molecule_dict_filter.values()))[0])))
    print("DONE!")

    # TODO: normalize distribution of remaining features, boxcox?
    return molecule_dict_filter