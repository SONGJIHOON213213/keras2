import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG19, Xception, InceptionResNetV2, EfficientNetB0, MobileNet,EfficientNetB0
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np
# 1. 데이터
save_path = 'd:/study_data/_save/cat_dog/'
datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
cat_dog = datagen.flow_from_directory('d:/study_data/_data/human/', target_size=(200, 200), batch_size=200, class_mode='binary', color_mode='rgb', shuffle=True)


cat_dog_x_train = np.load(save_path + 'keras56_cat_dog_x_train.npy')
cat_dog_x_test = np.load(save_path + 'keras56_cat_dog_x_test.npy')
cat_dog_y_train = np.load(save_path + 'keras56_cat_dog_y_train.npy')
cat_dog_y_test = np.load(save_path + 'keras56_cat_dog_y_test.npy')

# 2. 모델구성
model = Sequential()
model.add(VGG19(weights='imagenet', include_top=False, input_shape=(200, 200, 3)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(cat_dog_x_train, cat_dog_y_train, epochs=10, validation_data=(cat_dog_x_test, cat_dog_y_test))

import os
from sklearn.metrics import accuracy_score

pred_path = 'd:/study_data/_data/train_face/'
predict = datagen.flow_from_directory(pred_path, target_size=(200, 200), batch_size=1, class_mode='binary', color_mode='rgb', shuffle=False)

x_pred = predict[0][0]  # 한 개의 이미지만 예측
y_true = predict[0][1]

y_pred = model.predict(x_pred)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
y_pred_label = 'dog' if y_pred_binary == 1 else 'cat'

acc = accuracy_score(y_true, y_pred_binary)
print('Accuracy:', acc)
print('Predicted label:', y_pred_label)


# Accuracy: 0.7
# Predicted label: dog
# 내 결과 개