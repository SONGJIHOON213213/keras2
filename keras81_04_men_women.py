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
save_path = 'd:/study_data/_save/men_woman/'
datagen = ImageDataGenerator(rescale=1./255)
start = time.time()
men_woman = datagen.flow_from_directory('d:/study_data/_data/human/', target_size=(200, 200), batch_size=200, class_mode='binary', color_mode='rgb', shuffle=True)
men_woman_x_train = np.load(save_path + 'keras56_men_woman_x_train.npy')
men_woman_x_test = np.load(save_path + 'keras56_men_woman_x_test.npy')
men_woman_y_train = np.load(save_path + 'keras56_men_woman_y_train.npy')
men_woman_y_test = np.load(save_path + 'keras56_men_woman_y_test.npy')


# 2. 모델구성
model = Sequential()
model.add(VGG19(weights='imagenet', include_top=False, input_shape=(100, 100, 3)))  # Load the pre-trained VGG19 model
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(men_woman_x_train, men_woman_y_train, epochs=10, validation_data=(men_woman_x_test, men_woman_y_test))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']


import os
from sklearn.metrics import accuracy_score

pred_path = 'd:/study_data/_data/train_face/'
predict = datagen.flow_from_directory(pred_path, target_size=(100, 100), batch_size=1, class_mode='binary', color_mode='rgb', shuffle=False)

x_pred = predict[0][0]  # 한 개의 이미지만 예측
y_true = predict[0][1]

# 4. 평가, 예측
loss = model.evaluate(men_woman_x_test, men_woman_y_test)
print('loss : ', loss)

y_predict = np.round(model.predict(men_woman_x_test))
from sklearn.metrics import accuracy_score
acc = accuracy_score(men_woman_y_test, y_predict)
print('acc : ', acc) 

# 예측 결과 출력
class_labels = ['여자', '남자']
pred_label = class_labels[int(np.round(model.predict(x_pred)[0]))]
true_label = class_labels[int(y_true[0])]
print('예측 레이블:', pred_label)