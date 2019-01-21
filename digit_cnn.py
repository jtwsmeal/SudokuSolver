import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import numpy as np
mnist = tf.keras.datasets.mnist


def digit_recognizer(): #input_img
	batch_size = 128

	# 9 classes so that we don't predict 0.
	num_classes = 9
	epochs = 5

	# Input image dimensions
	input_shape = (28, 28, 1)

	num_filters = 64
	pool_size = (2, 2)
	kernel_size = (3, 3)

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)

	# Delete the training and test data corresponding to a label of 0, because we want to predict between 1 and 9.
	x_train = np.delete(x_train, np.where(y_train == 0), axis=0)
	y_train = np.delete(y_train, np.where(y_train == 0))
	x_test = np.delete(x_test, np.where(y_test == 0), axis=0)
	y_test = np.delete(y_test, np.where(y_test == 0))

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = models.Sequential()

	# Define our model.
	model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=input_shape))
	model.add(layers.MaxPooling2D(pool_size))
	model.add(layers.Conv2D(64, kernel_size, activation='relu'))
	model.add(layers.MaxPooling2D(pool_size))
	model.add(layers.Conv2D(64, kernel_size, activation='relu'))
	model.add(layers.MaxPooling2D(pool_size))
	model.add(layers.Dropout(0.5))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dropout(0.5))

	# Add a softmax layer with 9 output units:
	model.add(layers.Dense(num_classes, activation='softmax'))

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
	test_loss, test_accuracy = model.evaluate(x_test, y_test)
	print('Test loss:  {}'.format(test_loss))
	print('Test accuracy:  {}'.format(test_accuracy))

	model.save('digits.h5')


if __name__ == '__main__':
	digit_recognizer()
