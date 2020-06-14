#here we build a basic CNN or convalutional nural network
#https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=

#region Documentation
# Convolutional Neural Network
# Note: I will use the term convnet and convolutional neural network interchangably.
#
# Each convolutional neural network is made up of one or many convolutional layers.
# These layers are different than the dense layers we have seen previously. Their goal is to find
# patterns from within images that can be used to classify the image or parts of it. But this may sound
# familiar to what our densly connected neural network in the previous section was doing, well that's becasue it is.
#
# The fundemental difference between a dense layer and a convolutional layer is that dense
#  layers detect patterns globally while convolutional layers detect patterns locally. When
# we have a densly connected layer each node in that layer sees all the data from the previous
#  layer. This means that this layer is looking at all the information and is only capable
# of analyzing the data in a global capacity. Our convolutional layer however will not be densly
# connected, this means it can detect local patterns using part of the input data to that layer.

#Main takeaway - Dence nural networks find global patterns while Convalutonal nural networks find local patterns
#and can then indentify them again in different sets of data with different global patterns

#These patterns are then sent as output to an output feature map
#
# Let's have a look at how a densly connected layer would look at an image vs how a convolutional layer would.
# the goal of our network will be to determine whether this image is a cat or not.

# We indentify these features by running filters over our data in segments

# Strides
# In the previous sections we assumed that the filters would be slid continously through the image such
# that it covered every possible position. This is common but sometimes we introduce the idea of a stride
#  to our convolutional layer. The stride size reprsents how many rows/cols we will move the filter each
# time. These are not used very frequently so we'll move on.
#
# Pooling
# You may recall that our convnets are made up of a stack of convolution and pooling layers.
#
# The idea behind a pooling layer is to downsample our feature maps and reduce their dimensions.
# They work in a similar way to convolutional layers where they extract windows from the feature map
#  and return a response map of the max, min or average values of each channel. Pooling is usually done
#  using windows of size 2x2 and a stride of 2. This will reduce the size of the feature map by a factor
# of two and return a response map that is 2x smaller.
#endregion

# Dataset
# The problem we will consider here is classifying 10 different everyday objects. The dataset we will
# use is built into tensorflow and called the CIFAR Image Dataset. It contains 60,000 32x32 color images
# with 6000 images of each class.
#
# The labels in this dataset are the following:
#
# Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
# We'll load the dataset and have a look at some of the images below.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# set class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images
plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# CNN Architecture
# A common architecture for a CNN is a stack of Conv2D and MaxPooling2D layers followed by a few
# denesly connected layers. To idea is that the stack of convolutional and maxPooling layers extract
#  the features from the image. Then these features are flattened and fed to densly connected layers
#  that determine the class of an image based on the presence of features.
# We will start by building the Convolutional Base.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Layer 1
# The input shape of our data will be 32, 32, 3 and we will process 32 filters of size 3x3 over
#  our input data. We will also apply the activation function relu to the output of each convolution operation.
# Layer 2
# This layer will perform the max pooling operation using 2x2 samples and a stride of 2.
# Other Layers
# The next set of layers do very similar things but take as input the feature map from the previous layer.
#  They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks in spacial
# dimensions as it passed through the layers, meaning we can afford (computationally) to add more depth.

#print(model.summary())  # let's have a look at our model so far

# Adding Dense Layers
# So far, we have just completed the convolutional base. Now we need to take these extracted features
#  and add a way to classify them. This is why we add the following layers to our model.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())  # let's have a look at our model so far


#next lets train the current model using our data - This will take some time with a CNN
#we are using the Sparse CategoricalCrossentropy function for this there are others you can look up
#keras loss functions - to see the full set of options
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))

#make predictions with the model useing the same code from our nural network example
#next review data augmenation