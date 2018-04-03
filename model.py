import csv
import cv2
import random
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

import numpy as np

# read log
lines = []
dirs = ['data', 'data_lap2&3', 'data_counter_clockwise', 'data_side']
for data_dir in dirs:
  with open(data_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)
random.shuffle(lines)

# balance data
nbins = 2000
max_examples = 200
grouped_lines = [[] for _ in range(nbins)]
for line in lines:
  index = min(int(abs(float(line[3]) * nbins)), nbins - 1)
  if len(grouped_lines[index]) < max_examples:
    grouped_lines[index].append(line)
lines = []
for sub_lines in grouped_lines:
  lines += sub_lines

# augment data
augmented_images = []
augmented_measurements = []
correction = [0, 0.2, -0.2]
for line in lines:
  for i in range(3):
    source_path = line[i]
    current_path = ''
    if len(source_path.split('/')) < 3:
      current_path = 'data/IMG/' + source_path.split('/')[-1]
    else:
      current_path = '/'.join(source_path.split('/')[-3:])
    image = cv2.imread(current_path)
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image, 1))
    measurement = float(line[3])
    augmented_measurements.append(measurement + correction[i])
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# set up model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
best_model = ModelCheckpoint('model_best.h5', verbose=2, save_best_only=True)
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
                           nb_epoch=30, verbose=1, callbacks=[best_model])

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
