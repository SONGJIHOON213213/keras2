import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG19, Xception, InceptionResNetV2, EfficientNetB0, MobileNet,EfficientNetB0
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

models = {
    'VGG19': VGG19,
    'DenseNet121': DenseNet121,
    'MobileNet': MobileNet,
}

for model_name, model_class in models.items():
    base_model = model_class(weights='imagenet', include_top=False, input_shape=(71, 71, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(100, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print('Model:', model_name)
    x_test_resized = tf.image.resize(x_test, (71, 71))
    predictions = model.predict(x_test_resized)
    correct_predictions = 0
    total_images = len(x_test)
    for i in range(total_images):
        class_index = np.argmax(predictions[i])
        if class_index == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy:.4f}")
    


    #  'VGG19
    #  'Resnet50
    #  NasNetmobile
    # 'ResNet101': ResNet101,
    # 'DenseNet121': DenseNet121,
    # 'DenseNet169': DenseNet169,
    # 'InceptionV3': InceptionV3,
    # 'InceptionResNetV2': InceptionResNetV2,
    # 'MobileNet': MobileNet,
    # 'EfficientNetB0': EfficientNetB0,
    # 'Xception': Xception