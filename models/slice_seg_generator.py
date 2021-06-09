from datetime import datetime
import os
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

from data.datagen import normalize
from data.seg_datagen import n_structures, invert_corrected_ids
from models.layers import CBR

# maxs out at around 35 epochs
class SliceSegGenerator():
    
    def __init__(self, name='model', load=False, continue_from=False, lr=0.0001):
        self.name = name
        self.model = None
        self.input_shape = (256, 256, 5, 1)
        self.dropout_rate = 0.1
        self.lr = lr
        self.num_classes = n_structures + 1
        self.weights = None
        self.true_output_model = None
        if load or continue_from:
            if continue_from:
                self.model = load_model(continue_from, compile=False)
            else:
                self.model = load_model(self.name, compile=False)
            self.csfn_output_name = self.model.layers[-1].name.split('/')[0]
        else:
            self.build()
        self.compile()

    def build(self):
        wmn_input = Input(shape=self.input_shape)

        block_A = CBR(wmn_input, 16, (3, 3, 3)) # more filters?
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
        csfn_output = Conv3D(filters=self.num_classes, kernel_size=(1, 1, 1), activation='softmax')(block_E)
        csfn_output = Reshape((256, 256, self.num_classes))(csfn_output)
        csfn_output = Reshape((256 * 256, self.num_classes))(csfn_output)
            
        self.csfn_output_name = csfn_output.name.split('/')[0]

        self.model = keras.Model(inputs=[wmn_input], outputs=[csfn_output], name='network')

    def weighted_sparse_categorical_crossentropy(self, labels, predictions):
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)
        weights = tf.convert_to_tensor(self.weights, predictions.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(losses * weights), tf.reduce_sum(weights))

    def compile(self):
        optimizer = Adam(learning_rate=self.lr)
        loss = 'sparse_categorical_crossentropy'
        metrics = 'sparse_categorical_accuracy'
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary(line_length=150)

    def train(self, generator, batches_per_epoch, weights, epochs=5, save=True):
        start_time = datetime.now()
        history = {
            'train_loss': [],
            'train_acc': [],
        }

        last_time = start_time
        for epoch in range(epochs):
            for idx, (X, y) in tqdm(enumerate(generator()), total=batches_per_epoch, dynamic_ncols=True):
                loss, acc = self.model.train_on_batch(X, y, class_weight=weights, reset_metrics=(idx == 0))

            now = datetime.now()

            total_time = now - start_time

            history['train_loss'].append(loss)
            history['train_acc'].append(acc)
            print('[Epoch {}/{}] [Loss: {}] [Acc: {}] time: {} ({}s elapsed)'.format(
                epoch,
                epochs,
                loss,
                acc,
                total_time,
                round((now - last_time).total_seconds(), 2),
            ))
            last_time = now

        if save:
            self.model.save(os.path.join('trained_models', self.name))
            for name, values in history.items():
                plt.plot(range(0, epochs), values, label=name)
                plt.xlabel('Iters')
                plt.ylabel(name)
                plt.legend()
                plt.savefig('training_graphs/generator/{}_{}_history.png'.format(self.name, name))
                plt.clf()

    def convert_from_array(self, wmn_input, batch_size=64):
        if self.true_output_model is None:
            self.true_output_model = keras.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

        wmn_input = normalize(wmn_input)
        final = np.zeros(wmn_input.shape)

        batches = []
        ranges = []

        cur_batch = []
        for i in range(0, wmn_input.shape[1] - self.input_shape[2] + 1):
            if i % batch_size == 0 and len(cur_batch):
                batches.append(cur_batch)
                cur_batch = []
            slab = wmn_input[:, i:i + self.input_shape[2], :]
            slab = np.rollaxis(slab, 2)[:,:,:,None]
            cur_batch.append(slab)
        if len(cur_batch):
            batches.append(cur_batch)
        batches = [np.array(b) for b in batches]
        idx = 0
        for batch in batches:
            res = self.true_output_model.predict(batch).argmax(axis=-1)#[:,:,:,0]
            # res = res.reshape(batch.shape[:3])
            res = np.rollaxis(res, 0, 2)
            res = np.swapaxes(res, 0, 2)
            final[:, idx:idx + batch.shape[0], :] = res
            idx += batch.shape[0]
        return invert_corrected_ids(final)

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
    model = SliceSegGenerator()
    model.summary()
