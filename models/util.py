import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
)
from tensorflow.keras.optimizers import (
    RMSprop,
    Adagrad,
    Adadelta,
    Adam
)
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Conv3D,
    ReLU,
    Dropout,
    BatchNormalization,
    Conv3DTranspose,
    ZeroPadding3D,
    MaxPooling3D,
    Reshape,
)

from tensorflow.keras.models import load_model

def build_feature_extractor():
    return
    # model = load_model('saved_models/discriminator')
    # flattened final features
    # return keras.Model(inputs=model.layers[0].input, outputs=model.layers[-4].output)

    # after 2 conv / maxpol / 2 conv / max pool / 1 conv
    # return keras.Model(inputs=model.layers[0].input, outputs=model.layers[15].output)

def build_vgg16_feature_extractor():
    vgg16_input = Input(shape=(256, 256, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=vgg16_input)
    return keras.Model(inputs=vgg16_input, outputs=vgg16.layers[-4].output)

feature_extractor = build_feature_extractor()
vgg16_feature_extractor = build_vgg16_feature_extractor()

if __name__ == '__main__':
    vgg16_feature_extractor.summary()
    feature_extractor.summary()
