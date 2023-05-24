import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# VGG16 모델 로드
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 모델 요약 정보 출력
base_model.summary()
print("Number of total weights:", len(base_model.weights))
print("Number of trainable weights:", len(base_model.trainable_weights))

# 새로운 Sequential 모델 정의
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# 기존 VGG16 모델의 가중치 동결
model.trainable = False

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)