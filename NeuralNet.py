import numpy as np
#import os
import pickle
import glob
import csv
import random
import tensorflow as tf
import sklearn
from sklearn.metrics import r2_score
from tensorflow.python.keras.utils import to_categorical

def NeuralNet(data_directory, activity_to_predict, molecule_dict_filter):
    print("STARTING NeuralNet")
    #training_directory = "/Users/soonerfan237/Desktop/MerckActivity/TrainingSUBSet/"

    # fileObject = open(data_directory+"molecule_dict_filter.pickle",'rb')
    # molecule_dict_filter = pickle.load(fileObject)
    # fileObject.close()

    data_set = []
    for molecule, values in molecule_dict_filter.items():
        if values[1][activity_to_predict] is not None:
            if str(values[1][activity_to_predict])[0] != '-':
                label = int(str(values[1][activity_to_predict])[0])
                data = values[0]
                data_set.append([data,label])
            else:
                print("found negative activity")

    random.shuffle(data_set)

    #print(len(data_set))
    features_length = len(data_set[0][0])
    #print(features_length)

    training_set = data_set[:int(7*len(data_set)/10)]
    test_set = data_set[int(7*len(data_set)/10):]

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
    #labels_train = to_categorical(labels_train)

    features_test = np.array(features_test)
    features_test = features_test.reshape(-1, 1, features_length)
    features_test = features_test.astype(float)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(float)
    #labels_test = to_categorical(labels_test)

    #print("features_train type= " + str(type(features_train)))
    #print("features_train shape= " + str(features_train.shape))
    #print("labels_train type= " + str(type(labels_train)))
    #print("labels_train shape= " + str(labels_train.shape))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.Input(tensor=features_train))
    #model.add(tf.keras.InputLayer(input_tensor=features_train))
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(features_train, labels_train, epochs=5, validation_data=(features_test,labels_test))

    model.save("merck"+str(activity_to_predict)+".model")
    new_model = tf.keras.models.load_model("merck"+str(activity_to_predict)+".model")
    predictions = new_model.predict(features_test)
    new_model.summary()
    #print("REAL VALUE: " + str(labels_test[0]))
    #print("PREDICTED VALUE: " + str(np.argmax(predictions[0])))
    print(predictions[0])
    y_pred = []
    for i in range(0,len(labels_test)):
        #print("REAL: " + str(labels_test[i]) + " | PREDICTED: " + str(np.argmax(predictions[i])))
        y_pred.append(np.argmax(predictions[i]))

    y_true = np.array(labels_test)
    y_true.astype(float)
    y_pred = np.array(y_pred)
    y_pred.astype(float)
    r_squared = r2_score(y_true, y_pred)
    print("R2 = " + str(r_squared))
    print("DONE!!!!")