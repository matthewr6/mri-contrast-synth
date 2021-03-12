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
    Conv2D,
    ReLU,
    Dropout,
    BatchNormalization,
    Conv3DTranspose,
    ZeroPadding3D,
    MaxPooling3D,
    Reshape,
)

from tensorflow.keras.models import load_model

def CBR(prior_layer, filters, kernel_size, padding='same'):
    layer = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding)(prior_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    return layer

def CBR_2D(prior_layer, filters, kernel_size, padding='valid'):
    layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(prior_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    return layer
