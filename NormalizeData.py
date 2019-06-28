import glob
import csv
import re
import pickle
import numpy as np
from statistics import mean
from statistics import median

training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet/"
files = glob.glob(training_directory+"ACT*.csv")
#this part will just read the header row of each file and store a dictionary of all unique feature names and give them all a unique index
#ill use that index so that i can store each molecule's features in a consistent order later
print("READING IN FEATURE NAMES")
feature_dict = {}
feature_index = 0
for file in files:
    print(file)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)[2:] #skipping header line
        #print(header)
        header_novel = [] #these are the features that have not been seen yet
        header_novel_values = []
        for i in range(len(header)):
            if header[i] not in feature_dict:
                header_novel.append(header[i])
                header_novel_values.append(feature_index)
                feature_index = feature_index + 1
        header_novel_zip = zip(header_novel, header_novel_values)
        feature_dict.update(dict(header_novel_zip))

fileObject = open(training_directory+"feature_dict.pickle",'wb') # open the file for writing
pickle.dump(feature_dict,fileObject)
fileObject.close()

print("READING IN PROPERTY AND ACTIVITY VALUES")
#blank_feature_list = [0]*feature_index #by initializing to this, i am essentially imputing missing values with 0; maybe try other values here???? 0.5? mean of the feature?
#now ill read in the data from all files and store their features in a list across all datasets
molecule_dict = {} #this has molecule name as key, then an array. the first index is an array of all features and the second is an array of all activities
for file in files:
    print(file)
    match = re.search(r'ACT([0-9]+)',file)
    activity_index = int(match.group(1))
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)[2:] #reading header line
        #print(header)
        for row in reader:
            molecule_name = row[0][row[0].find("M_")+2:]
            molecule_activity = row[1]
            #print(molecule_name)
            if molecule_name not in molecule_dict:
                #molecule_dict.update({molecule_name : [[0]*feature_index,[]]}) #by initializing to this, i am essentially imputing missing values with 0; maybe try other values here???? 0.5? mean of the feature?
                molecule_dict.update({molecule_name : [[None]*feature_index,[None]*16]})
            molecule_dict[molecule_name][1][activity_index] = molecule_activity
            for i in range(len(row)-2):
                feature_name = header[i]
                molecule_dict[molecule_name][0][feature_dict[feature_name]] = row[i+2]
fileObject = open(training_directory+"molecule_dict.pickle",'wb') # open the file for writing
pickle.dump(molecule_dict,fileObject)
fileObject.close()

feature_list = []
for feature, value in feature_dict.items():
    feature_list.append(feature)

with open(training_directory+"molecule_dict.csv", mode='w') as molecule_dict_csv:
    molecule_dict_csv_writer = csv.writer(molecule_dict_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    molecule_dict_csv_writer.writerow(feature_list)
    for molecule, features in molecule_dict.items():
        molecule_dict_csv_writer.writerow(features[0])

print("DONE!")