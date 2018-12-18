import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

samples = []
with open('/opt/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	first = True
	for line in reader:
		if first:
			first = False
			continue
		samples.append(line)
import os.path

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

correction = 0.2
g_batch_size = 32

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for index in range(3):
					name = '/opt/data/IMG/'+batch_sample[index].split('/')[-1]
					center_image = cv2.imread(name)
					center_angle = float(batch_sample[index + 3])
					if index == 1:
						center_angle += correction
					elif index == 2:
						center_angle -= correction
					images.append(center_image)
					angles.append(center_angle)
					if index == 0 or index == 1 or index == 2:
						images.append(cv2.flip(center_image, 1))
						angles.append(-1.0 * center_angle)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=g_batch_size)
validation_generator = generator(validation_samples, batch_size=g_batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Activation
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x : x / 127.5 - 1.0, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation(activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Activation(activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation(activation = "relu"))
model.add(Dropout(0.3))
model.add(Activation(activation = "relu"))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples)/g_batch_size, validation_data=validation_generator, nb_val_samples=len(validation_samples)/g_batch_size, nb_epoch=5)

model.save('model.h5')

