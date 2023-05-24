import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))



##################include_top=True##################

#1. FC layer 원래꺼 쓴다
#2. input_shzpe(224,224,3)고정값 바꿀수없다.