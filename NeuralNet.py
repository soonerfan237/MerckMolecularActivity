import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
import random
import tensorflow as tf

training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSet/"

data_set = []

files = glob.glob(training_directory+"ACT7*.csv")
for file in files:
    print(file)
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) #skipping header line
        for row in reader:
            #print(str(row[1]))
            if str(row[1])[0] != '-':
                label = int(str(row[1])[0])
                data_set.append([row[2:],label])

random.shuffle(data_set)

print(len(data_set))
features_length = len(data_set[0][0])
print(features_length)

training_set = data_set[:int(len(data_set)/2)]
test_set = data_set[int(len(data_set)/2):]

features_train = []
labels_train = []
for features, labels in training_set:
    features_train.append(features)
    labels_train.append(labels)

features_test = []
labels_test = []
for features, labels in test_set:
    features_test.append(features)
    labels_test.append(labels)

features_train = np.array(features_train)
features_train = features_train.reshape(-1, 1, features_length)
features_train = features_train.astype(float)
labels_train = np.array(labels_train)
labels_train = labels_train.astype(float)

features_test = np.array(features_test)
features_test = features_test.reshape(-1, 1, features_length)
features_test = features_test.astype(float)
labels_test = np.array(labels_test)
labels_test = labels_test.astype(float)

print("features_train type= " + str(type(features_train)))
print("features_train shape= " + str(features_train.shape))
print("labels_train type= " + str(type(labels_train)))
print("labels_train shape= " + str(labels_train.shape))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(features_train, labels_train, epochs=10)

model.save('merck.model')
new_model = tf.keras.models.load_model('merck.model')
predictions = new_model.predict(features_test)

print(np.argmax(predictions[0]))
print("DONE!!!!")
print(features_length)