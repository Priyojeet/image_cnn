from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint


datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (400, 400),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
valid_generator = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (400, 400),
                                            batch_size = 32,
                                            class_mode = 'categorical')


new_model = load_model("model.h5") # loding the previous model.
filepath = "model1.h5" # saving the new model.
checkpoint = ModelCheckpoint(filepath, monitor = 'acc', verbose = 1, save_best_only = True, mode = 'max') 
new_model.fit_generator(
    train_generator,
    steps_per_epoch=8000,
    epochs=2,
    validation_data=valid_generator,
    validation_steps=2000,
    callbacks = [checkpoint])


