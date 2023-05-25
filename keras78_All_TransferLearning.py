from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.applications import ResNet50,ResNet50V2
from tensorflow.keras.applications import ResNet101,ResNet101V2,ResNet152,ResNet152V2
from tensorflow.keras.applications import DenseNet201,DenseNet121,DenseNet169
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small,MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile,NASNetLarge
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB7
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

model_names = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'ResNet101': ResNet101,
    'ResNet101V2': ResNet101V2,
    'ResNet152': ResNet152,
    'ResNet152V2': ResNet152V2,
    'DenseNet201': DenseNet201,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'MobileNetV3Small': MobileNetV3Small,
    'MobileNetV3Large': MobileNetV3Large,
    'NASNetMobile': NASNetMobile,
    'NASNetLarge': NASNetLarge,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB7': EfficientNetB7,
    'Xception': Xception
}

for model_name, model_class in model_names.items():
    model = model_class(weights='imagenet')
    model.trainable = False

    print("===========================================")
    print("모델명: ", model_name)
    print("전체 가중치 갯수: ", len(model.weights))
    print("훈련 가능 갯수: ", len(model.trainable_weights))

