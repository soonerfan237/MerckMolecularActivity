import glob
import csv
import numpy as np

training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet/"

data_set = []
feature_dict = {}

#the idea here is for each feature, to get a list of all values
#then i can analyze the features and determine which have low or high variability
#the features with highest variability will be best used in the model

files = glob.glob(training_directory+"ACT7*.csv")
for file in files:
    print(file)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)[2:] #skipping header line
        print(header)
        header_values = []
        for i in range(len(header)):
            header_values.append([0])
        header_zip = zip(header, header_values)
        #print(header_zip)
        #print(dict(header_zip))
        feature_dict.update(dict(header_zip))
        #for feature in feature_dict:
        #    feature = []
        for row in reader:
            #label = int(str(row[1])[0])
            row_features = row[2:]
            for i in range(len(header)):
                feature_dict[header[i]].append(row_features[i])
            #feature_dict[]
            #data_set.append([row[2:],label])

    for feature, values in feature_dict.items():
        print(feature, np.std(np.array(values, dtype=np.float64), dtype=np.float64))