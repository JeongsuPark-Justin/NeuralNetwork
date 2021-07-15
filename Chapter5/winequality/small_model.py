import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/justin/PycharmProjects/PythonProject//Chapter5/Data/winequality-red.csv')

x = dataset.drop('quality', axis= 1)
y = dataset['quality']

#creating dataset
from sklearn.model_selection import train_test_split

x_train_1, x_test, y_train_1, y_test = train_test_split(x, y, test_size=0.15, random_state= 0)
x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, test_size=0.05, random_state=0)

#Scaling Dataset
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train_new = sc_x.fit_transform(x_train)

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20, 10))
# ax1.scatter(x_train.index, x_train['fixed acidity'], color='b', label='raw', alpha=0.4, marker='o')
# ax2.scatter(x_train.index, x_train_new[:, 1], color='c', label='adjusted', alpha=0.4, marker='o')
# ax1.set_title('Training dataset')
# ax2.set_title('Standardized training dataset')
#
# for ax in (ax1, ax2):
#     ax.set_xlabel('index')
#     ax.set_ylabel('fixed acidity')
#     ax.legend(loc='upper right')
#     ax.grid()
#
# plt.tight_layout()
# plt.show()


x_test_new = sc_x.transform(x_test)
x_val_new = sc_x.transform(x_val)
#Modeling
epoch = 30
def plot_learningCurve(history):
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['mae'])
    plt.plot(epoch_range, history.history['val_mae'])
    plt.ylim([0,2])
    plt.title('Model mae')
    plt.ylabel('mae')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc = 'upper right')
    plt.show()

    print("_____________________________________________________________")

    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.ylim([0, 4])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

small_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(1)
])
small_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_small = small_model.fit(x_train_new, y_train, batch_size=32, epochs=30, verbose=1, validation_data=(x_val_new, y_val))

plot_learningCurve(history_small)

s_test_loss, s_test_mae = small_model.evaluate(x_test_new, y_test, batch_size=32, verbose=1)
print("small model test_loss : {}".format(s_test_loss))
print("small model test_mae : {}".format(s_test_mae))

unseen_data = np.array([[6.0, 0.28, 0.22, 12.15, 0.048, 4.20, 163.0, 0.99570, 3.20, 0.46, 10.1]])
y_small = small_model.predict(sc_x.transform(unseen_data))
print("wine quality on unseen data (small model) : ", y_small[0][0])