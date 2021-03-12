from datetime import datetime

import tensorflow as tf
import nibabel as nib
import numpy as np

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
    Flatten,
    Dense,
)

from tensorflow.keras import losses
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from models.util import CBR

class SlabDiscriminator():
    
    def __init__(self, name='model', load=False, perceptual_loss=False, lr=0.001):
        self.name = name
        self.model = None
        self.input_shape = (256, 256, 5, 1)
        self.lr = lr
        if load:
            self.model = load_model(self.name, compile=False)
        else:
            self.build()
        self.compile()

    def build(self):
        wmn_input = Input(shape=self.input_shape)

        block_A = CBR(wmn_input, 16, (3, 3, 2), padding='valid')
        block_A = CBR(block_A, 32, (3, 3, 1), padding='valid')

        block_B = MaxPooling3D(pool_size=(2, 2, 1))(block_A)
        block_B = CBR(block_B, 32, (3, 3, 2), padding='valid')
        block_B = CBR(block_B, 64, (3, 3, 1), padding='valid')

        block_C = MaxPooling3D(pool_size=(2, 2, 1))(block_B)
        block_C = CBR(block_C, 64, (3, 3, 2), padding='valid')
        block_C = CBR(block_C, 128, (3, 3, 1), padding='valid')

        block_D = MaxPooling3D(pool_size=(2, 2, 1))(block_C)
        block_D = CBR(block_D, 128, (3, 3, 2), padding='valid')
        block_D = CBR(block_D, 64, (3, 3, 1), padding='valid')

        fc = Flatten()(block_D)
        fc = Dropout(0.25)(fc)
        fc = Dense(256, activation='relu')(fc)
        discrim_output = Dense(1, activation='sigmoid')(fc)
        self.output_name = discrim_output.name.split('/')[0]
        self.model = keras.Model(inputs=[wmn_input], outputs=[discrim_output], name='network')

    def compile(self):
        optimizer = Adam(learning_rate=self.lr)
        loss = 'binary_crossentropy'
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])

    def summary(self):
        self.model.summary()

    def binarize_prediction(self, pred):
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        return pred.flatten()

    def eval(self, pos_generator, neg_generator, batches):
        for iteration in range(batches):
            pos = next(pos_generator)
            neg = next(neg_generator)
            pos_results = self.binarize_prediction(self.model.predict_on_batch(pos))
            neg_results = self.binarize_prediction(self.model.predict_on_batch(neg))
            pos_correct = (pos_results == 1).sum()
            neg_correct = (neg_results == 0).sum()
            correct = pos_correct + neg_correct
            total = len(pos) + len(neg)
            print(pos_correct, neg_correct, total)

    def train(self, generator, iters=10, save=True):
        start_time = datetime.now()
        history = {
            'train_loss': [],
        }

        last_time = start_time
        for iteration in range(iters):
            X, y = next(generator)

            loss, acc = self.model.train_on_batch(X, y)

            now = datetime.now()

            total_time = now - start_time

            history['train_loss'].append(loss)
            print('[Iter {}/{}] [Loss: {}] [Acc: {}] time: {} ({}s elapsed)'.format(
                iteration,
                iters,
                loss,
                acc,
                total_time,
                round((now - last_time).total_seconds(), 2),
            ))
            last_time = now

        if save:
            for name, values in history.items():
                plt.plot(range(0, iters), values, label=name)
                plt.xlabel('Iters')
                plt.ylabel(name)
                plt.legend()
                plt.savefig('discriminator_training_graphs/{}_{}_history.png'.format(self.name, name))
                plt.clf()
            self.model.save(self.name)

if __name__ == '__main__':
    model = Generator()
