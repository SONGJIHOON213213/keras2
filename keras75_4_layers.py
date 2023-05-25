import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16


model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
# #1.전체동결
# model.trainable = False
# #2.전체동결
# for layer in model.layers:
#     layer.trainable = False

# #3.전체동결
print(model.layers[0])

model.layers[0].trainable = False
model.layers[1].trainable = False
model.layers[2].trainable = False

model.summary()

import pandas as pd
pd.set_option('max_colwidth',-1)
layers = [(layer,layer.name,layer.trainable)for layer in model.layers]
# print(layers)
results = pd.DataFrame(layers,columns=['layer type','layer name','layer Trainable'])
print(results)