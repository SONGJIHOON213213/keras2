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
path = 'd:/study_data/_data/horse-or-human/'
save_path = 'd:/study_data/_save/horse-or-human/'

datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
horse_human = datagen.flow_from_directory('d:/study_data/_data/horse-or-human/', target_size=(200, 200), batch_size=200, class_mode='binary', color_mode='rgb', shuffle=True)

horse_human_x = horse_human[0][0]
horse_human_y = horse_human[0][1]

horse_human_x_train = np.load(save_path + 'keras56_horse_human_x_train.npy')
horse_human_x_test =  np.load(save_path + 'keras56_horse_human_x_test.npy')
horse_human_y_train =  np.load(save_path + 'keras56_horse_human_y_train.npy')
horse_human_y_test =  np.load(save_path + 'keras56_horse_human_y_test.npy')

horse_human_x_train, horse_human_x_test, horse_human_y_train, horse_human_y_test = train_test_split(horse_human_x, horse_human_y, train_size=0.7, shuffle=True, random_state=123)

# 2. 모델구성
model = Sequential()
model.add(VGG19(weights='imagenet', include_top=False, input_shape=(200, 200, 3)))  # Load the pre-trained VGG19 model
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(horse_human_x_train, horse_human_y_train, epochs=30, validation_data=(horse_human_x_test, horse_human_y_test))

import os
from sklearn.metrics import accuracy_score

pred_path = 'd:/study_data/_data/train_face/'
predict = datagen.flow_from_directory(pred_path, target_size=(200, 200), batch_size=1, class_mode='binary', color_mode='rgb', shuffle=False)

x_pred = predict[0][0]  # 한 개의 이미지만 예측
y_true = predict[0][1]

y_pred = model.predict(x_pred)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
y_pred_label = 'hourse' if y_pred_binary == 1 else 'human'

acc = accuracy_score(y_true, y_pred_binary)
print('Accuracy:', acc)
print('Predicted label:', y_pred_label)

# Accuracy: 6.0
# Predicted label 내결과: hourse