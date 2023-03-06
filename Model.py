import tensorflow as tf
from keras.datasets import mnist
import numpy as np


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[...,np.newaxis]
x_test = x_test[..., np.newaxis]

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8,(3,3),activation ='relu',input_shape = x_train[0].shape),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    # tf.keras.layers.Dense(128,activation='relu',kernel_regularizer='l2'),
    # tf.keras.layers.Dense(64,activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(10,activation='softmax')
])
print (x_train[0].shape)
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy','mae'],optimizer='adam')
model.fit(x_train,y_train,epochs=10, validation_split= 0.2)
eval =model.evaluate(x_train,y_train)
print(eval)
model.save('MNIST/MNIST_Model')
# model.predict(x_test[10])
print(model.weights)
