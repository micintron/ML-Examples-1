#here we build out a basic nural network
#more info here
#https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#forceEdit=true&sandboxMode=true

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#look through data
fashion_mnist = keras.datasets.fashion_mnist  # load dataset
# split into tetsing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#see shape of data
#print(train_images.shape)

# let's have a look at one pixel
print(train_images[0,23,23])

# let's have a look at the first 10 training labels
print(train_labels[:10])

#set the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
#plt.show()


#next we prepeocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

#next we build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

#next we need to optimize and activate the model for hyper paramerter tuning
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#next we need to train the model
# we pass the data, labels and epochs and watch the magic!
model.fit(train_images, train_labels, epochs=1)

#last we run it on the real data and get the final result
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('Test accuracy:', test_acc)
print('done')

# next we are going to work on making some targeted predictions
# To make predictions we simply need to pass an array of data in the form
# we've specified in the input layer to .predict() method.
#remember to make predictions on a one of data you will need to format like this test_images[0]
predictions = model.predict(test_images)

#use this to get the index of the best pick out of all the outputs for this item
# index = np.argmax(predictions[0])
# print(index)
# print(class_names[index])

# #print out the index image result to check on accuracy
# plt.figure()
# plt.imshow(train_images[index])
# plt.colorbar()
# plt.grid(False)
# plt.show()


#next lets check data on its predicted result vresus actual result
# just check index of data you want to check as result l
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]
  print("Excpected: " + class_names[correct_label])
  print("Guess: " + predicted_class)
  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)