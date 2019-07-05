from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint


# data collecting and preprossecing
datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory('dataset/training_set', # path of your trainging dataset
                                                 target_size = (400, 400),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
valid_generator = test_datagen.flow_from_directory('dataset/test_set', # path of your test dataset
                                            target_size = (400, 400),
                                            batch_size = 32,
                                            class_mode = 'categorical')


#defining the model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (400, 400, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath = "model.h5" # saving the entire model
checkpoint = ModelCheckpoint(filepath, monitor = 'acc', verbose = 1, save_best_only = True, mode = 'max') # based on accuracy and the best model
#print("model saved")

model.fit_generator(
    train_generator,
    steps_per_epoch=8000,
    epochs=8,
    validation_data=valid_generator,
    validation_steps=2000,
    callbacks = [checkpoint])

