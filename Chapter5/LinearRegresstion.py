import tensorflow as tf
from tensorflow import keras
import pandas as pd

df = pd.read_csv('C:/Users/justin/PycharmProjects/PythonProject\
/Chapter5/Data/Simple-linear-regression.csv')


dataset = df.values
x = dataset[:,1]
y = dataset[:,0]

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss= 'mean_squared_error')
model.fit(x, y, epochs=15)

result = model.predict([5.0])
print("Expected SAT score for GPA 5.0 : {:.0f}".format(result[0][0]))

result = model.predict([1.2])
print("Expected SAT score for GPA 1.2 : {:.0f}".format(result[0][0]))