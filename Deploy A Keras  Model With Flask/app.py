# our web app framework!

# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)

from flask import Flask, render_template, request, Response
#old code use PIL instead
#from scipy.misc import imsave, imread, imresize
import imageio as im

import PIL
from PIL import Image
import numpy as np
import keras.models

#for regular expretions
import re

#For system and operations data
import sys
import os

sys.path.append(os.path.abspath('./model'))
from load import *
import base64


# initalize our flask app
app = Flask(__name__)
global model
# initialize these variables
model = init()



def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/predict/',methods=['GET','POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    print("debug")
    x = Image.open('output.png')
    # make it the right size
    x = x.resize((28, 28))
    x.save('output.png')
    # read the image into memory
    x = im.imread('output.png', pilmode='L')
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    print("debug2")
    # set the scope
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    # perform the prediction
    out = model.predict([x])
    print(out)
    print(np.argmax(out,axis=1))
    print("debug3")
    # convert the response to a string
    response = np.array_str(np.argmax(out, axis=1))
    return response


if __name__ == "__main__":
        app.run(debug=True)

