import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# CIFAR-100 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# VGG16 모델 로드
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 모델 요약 정보 출력
base_model.summary()
print(len(base_model.weights))
print(len(base_model.trainable_weights))

# 새로운 Sequential 모델 정의
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(100, activation='softmax'))  # Update the number of units to 100

# 기존 VGG16 모델의 가중치 동결
model.trainable = False

# 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])  # loss, metrics(acc)
print('acc:', results[1])  # loss, metrics(acc)