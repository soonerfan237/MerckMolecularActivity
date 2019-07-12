import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import pickle
import random
import numpy as np
from sklearn.metrics import r2_score

def ConvNeuralNet(data_directory, activity_to_predict, molecule_dict_filter):
    data_set = []
    for molecule, values in molecule_dict_filter.items():
        if values[1][activity_to_predict] is not None:
            if str(values[1][activity_to_predict])[0] != '-': #TODO: handle negative activity levels
                label = int(str(values[1][activity_to_predict])[0]) #TODO: come up with better grouping instead of just taking first digit
                data = values[0]
                data_set.append([data,label])

    random.shuffle(data_set)

    #print(len(data_set))
    features_length = len(data_set[0][0])
    #print(features_length)

    # TODO: put higher proportion in training set
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

    features_test = np.array(features_test)
    features_test = features_test.reshape(-1, 1, features_length)
    features_test = features_test.astype(float)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(float)

    #X = X/255.0
    print("SHAPE")
    print(features_train.shape)
    model = Sequential()
    #model.add(tf.keras.layers.Flatten())
    #INPUT LAYER
    #model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=features_train.shape))
    model.add(Conv1D(32, kernel_size=(30), activation='relu', padding='same', input_shape=(features_train.shape[1], features_train.shape[2])))
    model.add(MaxPooling1D(pool_size=(1)))

    #SECOND LAYER
    model.add(Conv1D(32, kernel_size=(30), activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=(1)))
    #model.add(Conv2D(32, kernel_size=(3, 3)))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    #THIRD LAYER
    #model.add(Conv1D(32, kernel_size=(30), activation='relu', padding='same'))
    #model.add(MaxPooling1D(pool_size=(1)))

    #FOURTH LAYER
    #model.add(Conv1D(32, kernel_size=(30), activation='relu', padding='same'))
    #model.add(MaxPooling1D(pool_size=(1)))

    #FIFTH LAYER
    model.add(Flatten())
    model.add(Dense(64))

    #OUTPUT LAYER
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(features_train, labels_train, batch_size=32, epochs=20, validation_split=0.3)

    model.save("merck" + str(activity_to_predict) + ".model")
    new_model = tf.keras.models.load_model("merck" + str(activity_to_predict) + ".model")
    predictions = new_model.predict(features_test)

    #print("REAL VALUE: " + str(labels_test[0]))
    #print("PREDICTED VALUE: " + str(np.argmax(predictions[0])))

    y_pred = []
    for i in range(0, len(labels_test)):
        #print("REAL: " + str(labels_test[i]) + " | PREDICTED: " + str(np.argmax(predictions[i])))
        y_pred.append(np.argmax(predictions[i]))

    y_true = np.array(labels_test)
    y_true.astype(float)
    y_pred = np.array(y_pred)
    y_pred.astype(float)
    r_squared = r2_score(y_true, y_pred)
    print("R2 = " + str(r_squared))
    print("DONE!!!!")