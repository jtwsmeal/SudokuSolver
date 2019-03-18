import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import numpy as np
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt


# Plot the training and validation loss and accuracy.
def plot_history(history, epochs):

	xaxis = [1, epochs]

	# Plot the accuracy.
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('Digit CNN Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.xlim(xaxis)
	plt.legend(['Train', 'Validation'])
	plt.savefig('accuracy_plot.png', bbox_inches='tight')
	plt.clf()

	# Plot the loss.
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Digit CNN Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.xlim(xaxis)
	plt.legend(['Train', 'Validation'])
	plt.savefig('loss_plot.png', bbox_inches='tight')


# Definition of the CNN.
def digit_recognizer():
	batch_size = 128

	# 9 classes so that we don't predict 0.
	num_classes = 9
	epochs = 13

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
	train_zero_indices = np.where(y_train == 0)
	test_zero_indices = np.where(y_test == 0)
	x_train = np.delete(x_train, train_zero_indices, axis=0)
	y_train = np.delete(y_train, train_zero_indices)
	x_test = np.delete(x_test, test_zero_indices, axis=0)
	y_test = np.delete(y_test, test_zero_indices)

	# Normalize the datasets.
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# Subtract 1 from y_train and y_test because Keras assumes that
	# class labels start at 0, and we removed the label 0.
	y_train = keras.utils.to_categorical(y_train - 1, num_classes)
	y_test = keras.utils.to_categorical(y_test - 1, num_classes)

	model = models.Sequential()

	# Define our model.
	model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=input_shape))
	model.add(layers.MaxPooling2D(pool_size))
	model.add(layers.Conv2D(64, kernel_size, activation='relu'))
	model.add(layers.MaxPooling2D(pool_size))
	model.add(layers.Conv2D(64, kernel_size, activation='relu'))
	model.add(layers.MaxPooling2D(pool_size))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(64, activation='relu'))

	# Add a softmax layer with 9 output units:
	model.add(layers.Dense(num_classes, activation='softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle= True, verbose=1)

	plot_history(history.history, epochs)
	test_loss, test_accuracy = model.evaluate(x_test, y_test)
	print('Test loss:  {}'.format(test_loss))
	print('Test accuracy:  {}'.format(test_accuracy))

	model.save('digits.h5')


if __name__ == '__main__':
	digit_recognizer()
