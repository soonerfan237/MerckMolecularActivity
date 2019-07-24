import numpy as np
import random
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

def generator_model(shape):
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=shape))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=shape, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

def discriminator_model(shape):
    discriminator = Sequential()
    discriminator.add(Dense(units=1024, input_dim=shape))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3)) #to prevent overfitting

    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3)) #to prevent overfitting

    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

def gan_model(discriminator, generator, shape):
    discriminator.trainable = False
    gan_input = Input(shape=(shape,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def generate_molecule(shape, generator, examples=100):
    random_noise = np.random.normal(loc=0, scale=1, size=[examples, shape])
    generated_molecules = generator.predict(random_noise)
    generated_molecules = np.array(generated_molecules)
    generated_molecules = np.reshape(generated_molecules, (generated_molecules.shape[0], generated_molecules.shape[1]))
    return generated_molecules

def predict_generated_molecule(generated_data, discriminator):
    #new_model = tf.keras.models.load_model("merck" + str(activity_to_predict) + ".model")
    predictions = discriminator.predict(generated_data)
    return predictions

def GenerativeAdversarialNetwork(data_directory, activity_to_predict, molecule_dict_filter, epochs):
    print("STARTING GenerativeAdversarialNetowrk")

    data_set = []
    for molecule, values in molecule_dict_filter.items():
        if values[1][activity_to_predict] is not None:
            if str(values[1][activity_to_predict])[0] != '-':
                label = int(str(values[1][activity_to_predict])[0])
                data = values[0]
                data_set.append([data, label])

    random.shuffle(data_set)

    # print(len(data_set))
    features_length = len(data_set[0][0])
    # print(features_length)

    #training_set = data_set[:int(7 * len(data_set) / 10)]
    #test_set = data_set[int(7 * len(data_set) / 10):]

    features_train = []
    #labels_train = []
    for features, labels in data_set:
        features_train.append(features)
    #    labels_train.append(labels)

    #features_test = []
    #labels_test = []
    #for features, labels in test_set:
    #    features_test.append(features)
    #    labels_test.append(labels)

    features_train = np.array(features_train)
    features_train = features_train.reshape(-1, 1, features_length)
    features_train = features_train.astype(float)
    #labels_train = np.array(labels_train)
    #labels_train = labels_train.astype(float)

    #features_test = np.array(features_test)
    #features_test = features_test.reshape(-1, 1, features_length)
    #features_test = features_test.astype(float)
    #labels_test = np.array(labels_test)
    #labels_test = labels_test.astype(float)

    batch_size = features_train.shape[0]
    shape = features_train.shape[2]

    generator = generator_model(shape)
    discriminator = discriminator_model(shape)
    gan = gan_model(discriminator, generator, shape)

    for e in range(epochs):
        print("Epoch " + str(e+1) + "/" + str(epochs))
        # TRAINING THE DISCRIMINATOR
        #generating fake molecules with the generator
        random_noise = np.random.normal(0, 1, [batch_size, shape]) #initializing with random data
        generated_molecules = generator.predict(random_noise) #generating molecular features from random data
        #picking a random set of real molecule features
        random_molecule_indices = np.random.randint(low=0, high=features_train.shape[0], size=batch_size)
        molecule_feature_batch = features_train[random_molecule_indices]
        molecule_feature_batch = np.reshape(molecule_feature_batch,(generated_molecules.shape[0], generated_molecules.shape[1]))
        #molecule_label_batch = labels_train[random_molecule_indices]
        features_fakereal_train = np.concatenate([molecule_feature_batch, generated_molecules]) #combining fake and real molecules
        #so we're telling the discriminator straight up which data is real and fake and letting it learn on that
        labels_fakereal_train = np.concatenate([[1]*batch_size,[0]*batch_size]) #the labels for the first half are 1 because they are real molecules, then 0 for the fake molecules
        discriminator.trainable = True
        #discriminator.train_on_batch(features_fakereal_train, labels_fakereal_train) #TRAINING THE DISCRIMINATOR on how to detect fake vs real data
        discriminator_results = discriminator.fit(features_fakereal_train,labels_fakereal_train)  # TRAINING THE DISCRIMINATOR on how to detect fake vs real data

        #TRAINING THE GENERATOR
        random_noise = np.random.normal(0, 1, [batch_size, shape]) #now we're generating fake data
        labels_eval = np.ones(batch_size) #but giving it a label of 1, so we're lying to the discriminator and saying it's real data
        discriminator.trainable = False #but we don't want to discriminator function to change here, we're just evaluating it
        #gan.train_on_batch(random_noise, labels_eval) #sending the faked data to the gan
        gan_results = gan.fit(random_noise, labels_eval)  # sending the faked data to the gan

        generated_data = generate_molecule(shape, generator)
        predictions = predict_generated_molecule(generated_data, discriminator)
        r_squared = r2_score([0]*len(predictions), predictions)
        print("DISCRIMINATOR R2 = " + str(r_squared))

    print("DONE!")
    #return generated_data, predictions