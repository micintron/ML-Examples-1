#video tutorials on this example -
#  https://www.youtube.com/watch?v=OS0Ddkle0o4&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#grab datasets
data = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#seperate training data
train_images = train_images/255.0
test_images = test_images/255.0

#set up the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#set up the layers of the model this nural network starts with just 1
# the higher the epoch the longer it takes as weights and balences have more time to find coralations see what it takes to get 90% accuracy
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

#for printing all loss accuracy of the model
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested ACC: ", test_acc)

#for predicting the trained model
prediction = model.predict(test_images)
#for predicting a single image - also set range to 1
#prediction = model.predict(np.array([test_images[7]]))

#visualize the models results
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual : "+ class_names[test_labels[i]])
    index = np.argmax(prediction[i])
    name = class_names[index]
    plt.title("Prediction : "+ name)
    plt.show()
