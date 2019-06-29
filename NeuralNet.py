import numpy as np
#import os
import pickle
import glob
import csv
import random
import tensorflow as tf

def NeuralNet(data_directory, activity_to_predict, molecule_dict_filter):
    #training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet/"

    # fileObject = open(data_directory+"molecule_dict_filter.pickle",'rb')
    # molecule_dict_filter = pickle.load(fileObject)
    # fileObject.close()

    data_set = []
    for molecule, values in molecule_dict_filter.items():
        if values[1][activity_to_predict] is not None: #TODO: this is hardcoded to the 4th activity dataset. add support for other activities
            if str(values[1][activity_to_predict])[0] != '-': #TODO: handle negative activity levels
                label = int(str(values[1][activity_to_predict])[0]) #TODO: come up with better grouping instead of just taking first digit
                data = values[0]
                data_set.append([data,label])

    random.shuffle(data_set)

    #print(len(data_set))
    features_length = len(data_set[0][0])
    #print(features_length)

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

    #print("features_train type= " + str(type(features_train)))
    #print("features_train shape= " + str(features_train.shape))
    #print("labels_train type= " + str(type(labels_train)))
    #print("labels_train shape= " + str(labels_train.shape))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.InputLayer(input_tensor=features_train))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(features_train, labels_train, epochs=10)

    model.save("merck"+str(activity_to_predict)+".model")
    new_model = tf.keras.models.load_model('merck.model')
    predictions = new_model.predict(features_test)

    print("REAL VALUE: " + str(data_set[0][1]))
    print("PREDICTED VALUE: " + str(np.argmax(predictions[0])))
    print("DONE!!!!")