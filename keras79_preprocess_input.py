from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

# path = 'D:\study_data\_data\cat_dog\PetImages\Dog\\5.jpg'
path = 'D:/study_data/_data/suit/6.jpg'

img = image.load_img(path, target_size=(224, 224))
print(img)

x = image.img_to_array(img)
print("=================================== image.img_to_array(img) =======================")
print(x, '\n', x.shape)
print(np.min(x), np.max(x))

x = np.expand_dims(x, axis=0)
print(x.shape)

################################# -1 에서 1사이로 정규화 ###########################
x = preprocess_input(x)
print(x.shape)
print(np.min(x), np.max(x))

print("==============================")
x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape) 

print("결과는 : ",decode_predictions(x_pred,top=5)[0])