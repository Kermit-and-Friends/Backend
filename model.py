from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

Training = pd.read_csv("sign_mnist_train.csv")
Testing = pd.read_csv("sign_mnist_test.csv")

yAxisTraining = Training["label"]
yAxisTesting = Testing["label"]
del Training["label"]
del Testing["label"]

LabelBinarizer = LabelBinarizer()
yAxisTraining = LabelBinarizer.fit_transform(yAxisTraining)
yAxisTesting = LabelBinarizer.fit_transform(yAxisTesting)

xAxisTraining = Training.values
xAxisTesting = Testing.values

xAxisTraining = xAxisTraining / 255
xAxisTesting = xAxisTesting / 255

xAxisTraining = xAxisTraining.reshape(-1, 28, 28, 1)
xAxisTesting = xAxisTesting.reshape(-1, 28, 28, 1)

DataGenerator = ImageDataGenerator(
    samplewise_center=False,
    featurewise_std_normalization=False,
    horizontal_flip=False,
    vertical_flip=False,
    featurewise_center=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

DataGenerator.fit(xAxisTraining)

TrainingModel = Sequential()
TrainingModel.add(Conv2D(75, (3, 3), input_shape=(28, 28, 1), strides=1, padding='same', activation='relu'))
TrainingModel.add(BatchNormalization())
TrainingModel.add(MaxPool2D((2, 2), strides=2, padding='same'))
TrainingModel.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
TrainingModel.add(Dropout(0.2))
TrainingModel.add(BatchNormalization())
TrainingModel.add(MaxPool2D((2, 2), strides=2, padding='same'))
TrainingModel.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
TrainingModel.add(BatchNormalization())
TrainingModel.add(MaxPool2D((2, 2), strides=2, padding='same'))
TrainingModel.add(Flatten())
TrainingModel.add(Dense(units=512, activation='relu'))
TrainingModel.add(Dropout(0.3))
TrainingModel.add(Dense(units=24, activation='softmax'))

TrainingModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
TrainingModel.summary()

history = TrainingModel.fit(DataGenerator.flow(xAxisTraining, yAxisTraining, batch_size=128), epochs=20, validation_data
=(xAxisTraining, yAxisTraining))

TrainingModel.save('smnist.h5')
