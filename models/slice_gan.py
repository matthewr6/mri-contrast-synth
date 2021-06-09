from datetime import datetime
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow import keras

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


class SliceGAN():

    def __init__(self, generator, discriminator, name='gan'):
        self.name = name
        self.generator = generator
        self.generator.model._name = 'generator'
        self.discriminator = discriminator
        self.discriminator.model._name = 'discriminator'

        self.pure_gan = False

        self.compile()

    def compile(self):
        slab_input = Input(shape=self.generator.input_shape)
        slice_output = self.generator.model(slab_input)

        discrim_output = self.discriminator.model([slab_input, slice_output])

        # TODO: either modify discriminator to include both I/O slices, or add MAE loss
        if self.pure_gan:
            self.full_model = keras.Model(inputs=slab_input, outputs=discrim_output, name='combined')
            self.full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            outputs = {'slice': slice_output, 'discrim': discrim_output}
            losses = {'slice': 'mean_absolute_error', 'discrim': 'binary_crossentropy'}
            metrics = {'slice': 'mean_absolute_error', 'discrim': 'accuracy'}
            self.full_model = keras.Model(inputs=slab_input, outputs=outputs, name='combined')
            self.full_model.compile(optimizer='adam', loss=losses, metrics=metrics)

        self.generator.model.summary()
        self.discriminator.model.summary()
        self.full_model.summary()

    def train_on_batch(self, X, y, reset_metrics=False):
        num_examples = X.shape[0]
        real = np.ones((num_examples, ))
        fake = np.zeros((num_examples, ))

        yhat = self.generator.model.predict(X)

        self.discriminator.model.trainable = True
        self.generator.model.trainable = False
        _, _ = self.discriminator.model.train_on_batch([X, y], real, reset_metrics=reset_metrics)
        discrim_loss, discrim_acc = self.discriminator.model.train_on_batch([X, yhat], fake)

        self.discriminator.model.trainable = False
        self.generator.model.trainable = True
        metrics = self.full_model.train_on_batch(X, {'slice': y, 'discrim': real}, return_dict=True, reset_metrics=reset_metrics)
        # generator_mean_absolute_error, loss, generator_loss, discriminator_loss, discriminator_accuracy
        # generator_mean_absolute_error = generator_loss
        # loss = generator_loss + discriminator_loss
        generated_acc = metrics['discriminator_accuracy']
        generated_loss = metrics['loss']
        return discrim_loss, generated_loss, discrim_acc, generated_acc

    def train(self, generator, batches_per_epoch, epochs=5, save=True):
        start_time = datetime.now()
        history = {
            'discrim_loss': [],
            'generated_loss': [],
            'discrim_acc': [],
            'generated_acc': [],
        }

        last_time = start_time
        for epoch in range(epochs):
            for idx, (X, y) in tqdm(enumerate(generator()), total=batches_per_epoch):
                discrim_loss, generated_loss, discrim_acc, generated_acc = self.train_on_batch(X, y, reset_metrics=(idx == 0))

            # if preloading gen/discrim - try training discrim on full epoch and then training generator on full epoch?

            now = datetime.now()

            total_time = now - start_time

            history['discrim_loss'].append(discrim_loss)
            history['generated_loss'].append(generated_loss)
            history['discrim_acc'].append(discrim_acc)
            history['generated_acc'].append(generated_acc)
            print('[Epoch {}/{}] [Discrim loss: {}, gen loss: {}] time: {} ({}s elapsed)'.format(
                epoch,
                epochs,
                round(discrim_loss, 3),
                round(generated_loss, 3),
                total_time,
                round((now - last_time).total_seconds(), 2),
            ))
            print('              [Discrim acc: {}, gen acc: {}] '.format(
                round(discrim_acc, 3),
                round(generated_acc, 3),
            ))
            last_time = now

        self.discriminator.model.save('{}/discriminator'.format(self.name))
        self.generator.model.save('{}/generator'.format(self.name))
