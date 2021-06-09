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
    MaxPooling2D,
    Reshape,
    Flatten,
    Dense,
)

from tensorflow.keras import losses
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from models.layers import CBR_2D

class SliceDiscriminator():
    
    def __init__(self, name='model', load=False, lr=0.001):
        self.name = name
        self.model = None
        self.input_shape = (256, 256, 1)
        self.lr = lr
        if load:
            self.model = load_model(self.name, compile=False)
        else:
            self.build()
        self.compile()

    def build(self):
        wmn_input = Input(shape=self.input_shape)

        block_A = CBR_2D(wmn_input, 16, (3, 3), padding='valid')
        block_A = CBR_2D(block_A, 32, (3, 3), padding='valid')

        block_B = MaxPooling2D(pool_size=(2, 2))(block_A)
        block_B = CBR_2D(block_B, 32, (3, 3), padding='valid')
        block_B = CBR_2D(block_B, 64, (3, 3), padding='valid')

        block_C = MaxPooling2D(pool_size=(2, 2))(block_B)
        block_C = CBR_2D(block_C, 64, (3, 3), padding='valid')
        block_C = CBR_2D(block_C, 128, (3, 3), padding='valid')

        block_D = MaxPooling2D(pool_size=(2, 2))(block_C)
        block_D = CBR_2D(block_D, 128, (3, 3), padding='valid')
        block_D = CBR_2D(block_D, 64, (3, 3), padding='valid')

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

    def eval(self, pos_generator, pos_batches, neg_generator, neg_batches):
        true_pos = 0
        total_pos = 0
        true_neg = 0
        total_neg = 0
        for batch in tqdm(pos_generator(), total=pos_batches):
            pos_results = self.binarize_prediction(self.model.predict_on_batch(batch))
            pos_correct = (pos_results == 1).sum()
            true_pos += pos_correct
            total_pos += len(batch)
        for batch in tqdm(neg_generator(), total=neg_batches):
            neg_results = self.binarize_prediction(self.model.predict_on_batch(batch))
            neg_correct = (neg_results == 0).sum()
            true_neg += neg_correct
            total_neg += len(batch)
        print('Pos acc: {}'.format(true_pos / total_pos))
        print('Neg acc: {}'.format(true_neg / total_neg))
        print('Total acc: {}'.format((true_pos + true_neg) / (total_pos + total_neg)))

    def train(self, generator, batches_per_epoch, epochs=5, save=True):
        start_time = datetime.now()
        history = {
            'train_loss': [],
        }

        last_time = start_time
        for epoch in range(epochs):
            for X, y in tqdm(generator(), total=batches_per_epoch):
                loss, acc = self.model.train_on_batch(X, y)

            now = datetime.now()

            total_time = now - start_time

            history['train_loss'].append(loss)
            print('[Epoch {}/{}] [Loss: {}] [Acc: {}] time: {} ({}s elapsed)'.format(
                epoch + 1,
                epochs,
                loss,
                acc,
                total_time,
                round((now - last_time).total_seconds(), 2),
            ))
            last_time = now

        if save:
            self.model.save(self.name)
            for name, values in history.items():
                plt.plot(range(0, epochs), values, label=name)
                plt.xlabel('Epochs')
                plt.ylabel(name)
                plt.legend()
                plt.savefig('discriminator_training_graphs/{}_{}_history.png'.format(self.name, name))
                plt.clf()
