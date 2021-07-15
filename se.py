from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data

X_train = X_train/255
X_test = X_test/255
num_classes=10
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

L, W, H, C = X_train.shape
input_shape = [W, H, C]
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, ))