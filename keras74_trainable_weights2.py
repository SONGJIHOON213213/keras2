import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 랜덤 시드 설정
np.random.seed(337)
tf.random.set_seed(337)

#1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2.모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(1))
model.trainable = False

model.compile(loss='mse',optimizer='adam')

model.fit(x,y,batch_size=1,epochs=1) 
y_predict = model.predict(x)
print(y_predict)


# [[ 0.28569695]
#  [ 0.20671399]
#  [ 0.12773107]
#  [ 0.048748  ]
#  [-0.03023495]]