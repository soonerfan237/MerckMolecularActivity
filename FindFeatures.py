import glob
import csv
import numpy as np
from statistics import mean
from statistics import median

training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet/"

data_set = []
feature_dict = {}

#the idea here is for each feature, to get a list of all values
#then i can analyze the features and determine which have low or high variability
#the features with highest variability will be best used in the model

files = glob.glob(training_directory+"ACT*.csv")
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
                header_novel_values.append([0])
        header_novel_zip = zip(header_novel, header_novel_values)
        #print(header_zip)
        #print(dict(header_zip))
        feature_dict.update(dict(header_novel_zip))
        #for feature in feature_dict:
        #    feature = []
        for row in reader:
            #label = int(str(row[1])[0])
            row_features = row[2:]
            for i in range(len(header)):
                feature_dict[header[i]].append(row_features[i])
            #feature_dict[]
            #data_set.append([row[2:],label])

stdev_list = []
for feature, values in feature_dict.items():
    stdev_list.append(np.std(np.array(values, dtype=np.float64), dtype=np.float64))
    print(feature, np.std(np.array(values, dtype=np.float64), dtype=np.float64))

print("LEN STDEV")
print(len(stdev_list))
zero_count = 0
for item in stdev_list:
    if item == 0:
        zero_count = zero_count+1
print("ZERO COUNT")
print(str(zero_count))
print("MAX STDEV")
print(max(stdev_list))
print("AVG STDEV")
print(mean(stdev_list))
print("MEDIAN STDEV")
print(median(stdev_list))
print("")