
import tensorflow as tf
import pip as p
import numpy as np
import matplotlib.pyplot as plt

classifier_url = “https://tfhub.dev/google...”
IMAGE_SHAPE = (244,244)

# create Model
Classifier = tf.keras.Sequential([ hub.KearsLayer(classifier_url, input_shape= IMAGE_SHAPE+(3,))]

#set up URLs for image downloads
image_url1 = “http://raw.githubusercontent.com/Apress/artific....
..url 5

#download images
pip install wget
import wget
wget.download(image_url1, ‘image1.jpg’)

!pip install --upgrade tensorflow_hub

  import tensorflow_hub as hub

  model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
  embeddings = model(["The rain in Spain.", "falls",
                      "mainly", "In the plain!"])

  print(embeddings.shape)  #(4,128)