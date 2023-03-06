import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


model=tf.keras.models.load_model('MNIST/MNIST_Model')
(x_train,y_train),(x_test,y_test)=mnist.load_data()

image=Image.open() #'picture'
image_gray = ImageOps.grayscale(image)
image_small = image_gray.resize((28,28))

# n = np.random.randint(len(x_train))
# img = x_train[n]
# print(y_train[n])
# x_trial = x_train[y_train==2]
# y_trial = y_train[y_train ==2]

#img = x_trial[0]

img=np.asarray(image_small)
plt.imshow(img)
plt.show()
#print(img.shape)

tf.keras.utils.normalize(img)
img=img.reshape(1,28,28,1)
prediction = model.predict(img)[0]
prediction_list = prediction.tolist()
max_value = max(prediction_list)
max_index = prediction_list.index(max_value)
print("The Predicted number is {}".format(max_index))

#print(y_trial)
#print(model.weights)



