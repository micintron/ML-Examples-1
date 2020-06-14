#Link to file
#https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=true&scrollTo=nc9RyHPYUnSK

#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

# Pretrained Models
# You would have noticed that the model in Deep Computer Vision takes a few minutes to train in the NoteBook and only
#  gives an accuaracy of ~70%. This is okay but surely there is a way to improve on this.

# In this section we will talk about using a pretrained CNN as apart of our own custom network to
# improve the accuracy of our model. We know that CNN's alone (with no dense layers) don't do anything
#  other than map the presence of features from our input. This means we can use a pretrained CNN,
# one trained on millions of images, as the start of our model. This will allow us to have a very
# good convolutional base before adding our own dense layered classifier at the end. In fact, by using
#  this techique we can train a very good classifier for a realtively small dataset (< 10,000 images).
#  This is because the convnet already has a very good idea of what features to look for in an image
# and can find them very effectively. So, if we can determine the presence of features all the rest
# of the model needs to do is determine which combination of features makes a specific image.


# Using a Pretrained Model
# In this section we will combine the tecniques we learned above and use a pretrained model
# and fine tuning to classify images of dogs and cats using a small dataset.
# This tutorial is based on the following guide from the TensorFlow documentation:
#  https://www.tensorflow.org/tutorials/images/transfer_learning


# Dataset
# We will load the cats_vs_dogs dataset from the modoule tensorflow_datatsets.
# This dataset contains (image, label) pairs where images have different dimensions and 3 color channels.
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

#lets view some images from our data
get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
  #plt.show()


#Since the sizes of our images are all different, we need to convert them all to the
# same size. We can create a function that will do that for us below.
# All images will be resized to 160x160- note better to compress images than to expand
IMG_SIZE = 160

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


#apply to images
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
  #plt.show()

#orginal img versus resized
# for img, label in raw_train.take(2):
#   print("Original shape:", img.shape)
#
# for img, label in train.take(2):
#   print("New shape:", img.shape)


#Finally we will shuffle and batch the images.
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# Picking a Pretrained Model
# The model we are going to use as the convolutional base for our model is the MobileNet
# V2 developed at Google. This model is trained on 1.4 million images and has 1000 different classes.
# We want to use this model but only its convolutional base. So, when we load in the model, we'll specify
# that we don't want to load the top (classification) layer. We'll tell the model what input shape to
# expect and to use the predetermined weights from imagenet (Googles dataset).
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#print(base_model.summary())

#At this point this base_model will simply output a shape (32, 5, 5, 1280) tensor that is a
#  feature extraction from our original (1, 160, 160, 3) image. The 32 means that we have 32
# layers of differnt filters/features.

for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
#print(feature_batch.shape)

#Freezing the Base
#The term freezing refers to disabling the training property of a layer. It simply means we won’t make
#  any changes to the weights of any layers that are frozen during training. This is important as we
# don't want to change the convolutional base that already has learned weights.
base_model.trainable = False
#print(base_model.summary())

#Adding our Classifier
#Now that we have our base layer setup, we can add the classifier. Instead of flattening
# the feature map of the base layer we will use a global average pooling layer that will
# average the entire 5x5 area of each 2D feature map and return to us a single 1280 element
# vector per filter.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

#add all layers together to build the final model
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

print(model.summary())

#Training the Model
#Now we will train and compile the model. We will use a very small learning rate to ensure that the
# model does not have any major changes made to it.

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
#print(accuracy0)

# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
acc = history.history['accuracy']
print(acc)


#save it so we dont have to do it again
model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
#and this is how we load it
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

print('done model fully saved and loaded')