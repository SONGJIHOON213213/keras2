import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))


model = Sequential()
model.add(VGG16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

model.trainalbe = False

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))