from datetime import datetime

import tensorflow as tf
import nibabel as nib
import numpy as np
from tqdm import tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

from tensorflow.keras import losses
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from datagen import normalize
from models.layers import CBR
from models.util import feature_extractor, vgg16_feature_extractor

class SliceGenerator():
    
    def __init__(self, name='model', load=False, vgg_perceptual_loss=False, perceptual_loss=False, lr=0.001):
        self.name = name
        self.model = None
        self.input_shape = (256, 256, 5, 1)
        self.use_perceptual_loss = perceptual_loss
        self.vgg_perceptual_loss = vgg_perceptual_loss
        self.dropout_rate = 0.1
        self.lr = lr
        if load:
            self.model = load_model(self.name, compile=False)
        else:
            self.build()
            self.compile()

    def build(self):
        wmn_input = Input(shape=self.input_shape)

        block_A = CBR(wmn_input, 16, (3, 3, 3))
        block_A = CBR(block_A, 32, (3, 3, 3))

        block_B = MaxPooling3D(pool_size=(2, 2, 1))(block_A)
        block_B = CBR(block_B, 32, (3, 3, 3))
        block_B = CBR(block_B, 64, (3, 3, 3))

        block_C = MaxPooling3D(pool_size=(2, 2, 1))(block_B)
        block_C = CBR(block_C, 64, (3, 3, 3))
        block_C = CBR(block_C, 128, (3, 3, 3))
        block_C = Dropout(rate=self.dropout_rate)(block_C)
        block_C = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(block_C)

        block_D = Concatenate()([block_C, block_B])
        block_D = CBR(block_D, 64, (3, 3, 3))
        block_D = CBR(block_D, 64, (3, 3, 3))
        block_D = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(block_D)

        block_E = Concatenate()([block_D, block_A])
        block_E = CBR(block_E, 32, (3, 3, 3))
        block_E = CBR(block_E, 32, (3, 3, 3))

        block_E = Conv3D(filters=32, kernel_size=(1, 1, 5))(block_E)
        csfn_output = Conv3D(filters=1, kernel_size=(1, 1, 1))(block_E)
        csfn_output = Reshape((256, 256, 1))(csfn_output)
        csfn_output = ReLU()(csfn_output)
            
        self.csfn_output_name = csfn_output.name.split('/')[0]

        self.model = keras.Model(inputs=[wmn_input], outputs=[csfn_output], name='network')

    def gram_matrix(self, x, norm_by_channels=False):
        x = K.permute_dimensions(x, (0, 3, 1, 2)) # (B, H, W, C) --> (B, C, H, W)
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        if norm_by_channels:
            denominator = C * H * W # Normalization from Johnson
        else:
            denominator = H * W # Normalization from Google
        gram = gram /  K.cast(denominator, x.dtype)
        return gram

    def vgg16_perceptual_loss(self, y, yhat):
        mae = losses.mean_absolute_error(y, yhat)

        y_features = vgg16_feature_extractor(K.repeat_elements(y, 3, 3))
        yhat_features = vgg16_feature_extractor(K.repeat_elements(yhat, 3, 3))

        S = self.gram_matrix(y_features)
        C = self.gram_matrix(yhat_features)
        perceptual = K.mean(K.square(S - C), axis=(0, 1, 2))

        return mae + perceptual

    def perceptual_loss(self, y, yhat):
        y_features = feature_extractor(y)
        yhat_features = feature_extractor(yhat)
        return K.mean(K.abs(y - yhat), axis=(1, 2, 3)) + losses.mean_squared_error(y_features, yhat_features)
        # return K.sum(losses.mean_absolute_error(y, yhat), axis=(1, 2, 3)) + losses.mean_squared_error(y_features, yhat_features)

    def compile(self):
        optimizer = Adam(learning_rate=self.lr)
        loss = {self.csfn_output_name: 'mean_absolute_error'}
        if self.use_perceptual_loss:
            loss[self.csfn_output_name] = self.perceptual_loss
        elif self.vgg_perceptual_loss:
            loss[self.csfn_output_name] = self.vgg16_perceptual_loss
        self.model.compile(optimizer=optimizer, loss=loss)

    def summary(self):
        self.model.summary()

    def train(self, generator, batches_per_epoch, epochs=5, save=True):
        start_time = datetime.now()
        history = {
            'train_loss': [],
        }

        last_time = start_time
        for epoch in range(epochs):
            for X, y in tqdm(generator(), total=batches_per_epoch):
                loss = self.model.train_on_batch(X, y)

            now = datetime.now()

            total_time = now - start_time

            history['train_loss'].append(loss)
            print('[Epoch {}/{}] [Loss: {}] time: {} ({}s elapsed)'.format(
                epoch,
                epochs,
                loss,
                total_time,
                round((now - last_time).total_seconds(), 2),
            ))
            last_time = now

        if save:
            self.model.save(self.name)
            for name, values in history.items():
                plt.plot(range(0, epochs), values, label=name)
                plt.xlabel('Iters')
                plt.ylabel(name)
                plt.legend()
                plt.savefig('training_graphs/{}_{}_history.png'.format(self.name, name))
                plt.clf()

    def convert_slab(self, input_slab):
        pred = self.model.predict(input_slab)
        return pred

    def convert_from_array(self, csfn_input):
        if self.normalize:
            csfn_input = normalize(csfn_input)
        final = np.zeros(csfn_input.shape)
        for i in range(0, csfn_input.shape[1] - self.input_shape[2] + 1):
            slab = csfn_input[:, i:i + self.input_shape[2], :]
            slab = np.rollaxis(slab, 2)
            slab = np.array([slab[:,:,:,None]])
            converted_slab = self.convert_slab(slab)[0, :, :, 0]
            final[:, i + int(self.input_shape[2]/2), :] = np.rollaxis(converted_slab, 0, 2)
        return final

    def convert_from_nifti(self, nifti):
        converted = self.convert_from_array(nifti.get_fdata())
        converted = nib.Nifti1Image(converted, affine=nifti.affine, header=nifti.header)
        return converted

    def convert_from_path(self, path, out_path=None):
        nifti = nib.load(path)
        out_image = self.convert_from_nifti(nifti)
        if out_path:
            nib.save(out_image, out_path)
        return out_image

if __name__ == '__main__':
    model = Generator()
    model.summary()
