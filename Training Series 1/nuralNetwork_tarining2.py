import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#retrive and train data
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)
print(train_data[0])

#set values for wrd indexing moive review data
wrd_index = data.get_word_index();
wrd_index = {k:(v+3) for k, v in wrd_index.items()}
wrd_index["<PAD>"]=0;
wrd_index["<START>"] =1;
wrd_index["<UNK>"]=2;
wrd_index["<UNUSED>"]=3
reverse_word_index = dict([(value, key) for (key, value) in wrd_index.items()])

#set all data to be the same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = wrd_index["<PAD>"],
                                                       padding ="post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = wrd_index["<PAD>"],
                                                       padding ="post",maxlen = 250)
print(len(test_data), " should be equal to ",  len(test_data))

#decode data tests
def decode_review_text(text):
    return " ".join([reverse_word_index.get(i,"?")for i in text])
#print("this translates to \n")
#print(decode_review_text(test_data[0]))
#print(len(test_data[0]), " compared to ",  len(test_data[2]))

#Models
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#validate data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# fit the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

#show single index of data
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review_text(test_review))
print("Prediction: "+str(predict[0]))
print("Actual: "+ str(test_labels[0])+'\n')

#show results
results = model.evaluate(test_data, test_labels)
print("Model Results")
print(results)
print('model test completed')

