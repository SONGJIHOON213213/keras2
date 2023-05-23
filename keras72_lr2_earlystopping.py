import numpy as np
import matplotlib.pyplot as plt

#[실습] 최솟값 넣을 변수하나 카운트할 변수하나
#다음 에포에 값과 최솟값을 비교 최솟값이 갱신되면 그변수에 최솟값을 넣어주고 카운트변수 초기화
#갱신이 안되면 카운트 변수 ++1 
# 운트 변수가 내가 원하는 얼리스탑핑 갯수에 도달하면 for 문을 stop

x = -10.0
y = 10
w = 11
epochs = 50
lr = 0.025
tolerance = 0.001
prev_loss = float('inf')
losses = []
min_loss = float('inf')
count = 0  
early_stopping_count = 3  

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2
    losses.append(loss)
    print('loss:', round(loss, 4), '\tPredict:', round(hypothesis, 4))

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if up_loss >= down_loss:
        w = w - lr
    else:
        w = w + lr

 
    if abs(loss - prev_loss) < tolerance:
        if loss < min_loss:
            min_loss = loss
            count = 0  
        else:
            count += 1  

        if count >= early_stopping_count:
            print('Early stopping at epoch', i+1)
            break

    prev_loss = loss

plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()