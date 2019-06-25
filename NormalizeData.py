import glob
import csv
import numpy as np
from statistics import mean
from statistics import median

training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet/"
files = glob.glob(training_directory+"ACT*.csv")
#this part will just read the header row of each file and store a dictionary of all unique feature names and give them all a unique index
#ill use that index so that i can store each molecule's features in a consistent order later
feature_dict = {}
feature_index = 0
for file in files:
    print(file)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)[2:] #skipping header line
        print(header)
        header_novel = [] #these are the features that have not been seen yet
        header_novel_values = []
        for i in range(len(header)):
            if header[i] not in feature_dict:
                header_novel.append(header[i])
                header_novel_values.append(feature_index)
                feature_index = feature_index + 1
        header_novel_zip = zip(header_novel, header_novel_values)
        feature_dict.update(dict(header_novel_zip))
for feature, index in feature_dict.items():
    print(feature, index)
blank_feature_list = [0]*feature_index

#now ill read in the data from all files and store their features in a list across all datasets
molecule_dict = {}
for file in files:
    print(file)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)[2:] #skipping header line
        #print(header)
        for row in reader:
            molecule_name = row[0][row[0].find("M_")+2:]
            print(molecule_name)
            if molecule_name not in feature_dict:
                feature_dict.update({molecule_name : blank_feature_list})
            for i in range(len(row)-2):
                feature_name = header[i]
                feature_dict[molecule_name][feature_dict[feature_name]] = row[i+2]