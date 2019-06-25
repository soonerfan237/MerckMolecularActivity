import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0])
#print(x_train[0])
#plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("x_train type= " + str(type(x_train)))
print("x_train shape= " + str(x_train.shape))
print("y_train type= " + str(type(y_train)))
print("y_train shape= " + str(y_train.shape))
#x_train = x_train.reshape(-1,1,28*28)

model.fit(x_train, y_train, epochs=3)

#val_loss, val_acc = model.evaluate(x_test, y_test)
#print(val_loss)
#print(val_acc)

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))