from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import logging
logging.getLogger().setLevel(logging.INFO)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#set up the test set and the training data set
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
#print(train.head(10))

#Now we can pop the species column off and use that as our label.
train_y = train.pop('Species')
test_y = test.pop('Species')

# the species column is now gone
print(train_y.head())
#here we can call this and view the full shape of our data
print(train.shape)


# here we define our input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#print out the generated feature columns
print(my_feature_columns)


# Building the Model
# And now we are ready to choose a model. For classification tasks there are variety of
# different estimators/models that we can pick from. Some options are listed below.
#
# DNNClassifier (Deep Neural Network)
# LinearClassifier
# We can choose either model but the DNN seems to be the best choice. This is because we may not be
# able to find a linear coorespondence in our data.
#
# So let's build a model!
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

#now lets train the model
# We include a lambda to avoid creating an inner function previously
classifier.train(input_fn=lambda:
           input_fn(train, train_y, training=True), steps=5000)

#evaluate training data
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


#run predictions on individual data
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

#take in user data
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

#run data through predictions
predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    print(pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))


#example data to run predictions on-
# Here is some example input and expected classes you can try above
# expected = [ 1 'Setosa', 2 'Versicolor', 3 'Virginica']
# predict_x = {        1    2    3
#     'SepalLength': [5.1, 5.9, 6.9],
#     'SepalWidth': [3.3, 3.0, 3.1],
#     'PetalLength': [1.7, 4.2, 5.4],
#     'PetalWidth': [0.5, 1.5, 2.1],}