import os
import sys
import glob
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train and/or evaluate WMn --> CSFn model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', dest='name', default='model', help='Model name to save/load')
parser.add_argument('--mode', dest='mode', default=None, choices=['train-discriminator', 'train-paired-discriminator', 'train-generator', 'eval-discriminator', 'eval-generator', 'batch-eval'])
parser.add_argument('--identity-mode', dest='identity_mode', default=None, choices=['csfn', 'wmn'], help='Train on identity transformation')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Epochs to train for, if training')
parser.add_argument('--inverse', dest='inverse', action='store_true', help='Train inverse direction (CSFn --> WMn)')
parser.add_argument('--input-mode', dest='input_mode', default='full', help='Whether to use full or stripped volumes', choices=['full', 'stripped_wmn', 'stripped_csfns'])
parser.add_argument('--gpu', dest='gpu', default=None, help='Passed to CUDA_VISIBLE_DEVICES')
parser.add_argument('--seg', dest='seg', action='store_true', help='Segmentation model')
parser.add_argument('--perceptual-loss', dest='perceptual_loss', action='store_true', help='Custom perceptual loss for training')
parser.add_argument('--vgg-perceptual-loss', dest='vgg_perceptual_loss', action='store_true', help='VGG16-based perceptual loss for training')
parser.add_argument('--gan', dest='gan', action='store_true', help='Train a GAN.')
parser.add_argument('--gan-discriminator-name', dest='discriminator_name', default=None, help='Pretrained discriminator for GAN architecture')
parser.add_argument('--gan-generator-name', dest='generator_name', default=None, help='Pretrained generator for GAN architecture')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Adam learning rate')
parser.add_argument('--no-normalize', dest='normalize', action='store_false', help=argparse.SUPPRESS)
parser.add_argument('--single-gpu', dest='multi_gpu', action='store_false', help=argparse.SUPPRESS)
parser.add_argument('--n-volumes', dest='n_volumes', type=int, default=60, help=argparse.SUPPRESS)
parser.add_argument('--dynamic-load', dest='preload_data', action='store_false', help=argparse.SUPPRESS)
parser.add_argument('--in', dest='infile', default=None, help='Volume to load for generator evaluation')
parser.add_argument('--out', dest='outfile', default=None, help='Output save path for generator evaluation')
parser.add_argument('--continue-training', dest='continue_training', action='store_true', help='writeme')
parser.add_argument('--continue-training-from', dest='continue_training_from', default=False, help='writeme')
parser.add_argument('--viz', dest='viz', action='store_true', help='writeme')

args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
import visualkeras

from data.datagen import (
    set_formats,
    paired_generator,
    corrupted_with_wmn_generator,
    eval_generator,
    paired_with_corruption_generator
)
from data.seg_datagen import seg_generator
from models.slice_generator import SliceGenerator
from models.slice_seg_generator import SliceSegGenerator
from models.slice_discriminator import SliceDiscriminator
from models.slice_with_slab_discriminator import SliceWithSlabDiscriminator
from models.slice_gan import SliceGAN

set_formats(args.input_mode)

def viz():
    model = SliceSegGenerator().model
    visualkeras.graph_view(model).show()

def train_gan():
    if args.discriminator_name:
        discriminator = SliceWithSlabDiscriminator(name=args.discriminator_name, load=True)
    else:
        discriminator = SliceWithSlabDiscriminator(name='discriminator')
    if args.generator_name:
        generator = SliceGenerator(name=args.generator_name, load=True)
    else:
        generator = SliceGenerator(name='generator')

    model = SliceGAN(generator, discriminator, name=args.name)

    generator, batches_per_epoch = paired_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        batch_size=12,
        n_volumes=args.n_volumes
    )
    model.train(generator, batches_per_epoch, epochs=args.epochs)

def evaluate_generator():
    if args.seg:
        model = SliceSegGenerator(name=args.name, load=True)
    else:
        model = SliceGenerator(name=args.name, load=True)
    if args.infile and args.outfile:
        model.convert_from_path(args.infile, out_path=args.outfile)

def train_generator(batch_size=10):
    if args.seg:
        model = SliceSegGenerator(
            name=args.name,
            lr=args.lr,
            load=args.continue_training,
            continue_from=args.continue_training_from,
        )
        generator, batches_per_epoch, weights = seg_generator(
            '/data/mradovan/7T_WMn_3T_CSFn_pairs',
            batch_size=batch_size,
            n_volumes=args.n_volumes,
        )
        model.train(generator, batches_per_epoch, weights, epochs=args.epochs)
    else:
        model = SliceGenerator(
            name=args.name,
            vgg_perceptual_loss=args.vgg_perceptual_loss,
            perceptual_loss=args.perceptual_loss,
            lr=args.lr,
            load=args.continue_training,
            continue_from=args.continue_training_from,
        )
        generator, batches_per_epoch = paired_generator(
            '/data/mradovan/7T_WMn_3T_CSFn_pairs',
            batch_size=batch_size,
            n_volumes=args.n_volumes,
            inverse=args.inverse,
            identity=args.identity_mode,
        )
        model.train(generator, batches_per_epoch, epochs=args.epochs)

def batch_eval():
    if args.seg:
        model = SliceSegGenerator(name=args.name, load=True)
    else:
        model = SliceGenerator(name=args.name, load=True)
    
    paths = glob.glob(args.infile)
    for subj_infile in tqdm(paths):
        subj_path = os.path.dirname(subj_infile)
        subj_outfile = os.path.join(subj_path, args.outfile)
        # if not os.path.exists(subj_outfile):
        model.convert_from_path(subj_infile, subj_outfile)

def train_discriminator(batch_size):
    model = SliceDiscriminator(
        name=args.name,
        perceptual_loss=args.perceptual_loss,
        lr=args.lr)
    generator, batches_per_epoch = corrupted_with_wmn_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        batch_size=batch_size,
        _normalize=args.normalize,
        n_volumes=args.n_volumes)
    model.train(generator, batches_per_epoch, epochs=args.epochs)

def train_paired_discriminator(batch_size=20):
    model = SliceWithSlabDiscriminator(
        name=args.name,
        lr=args.lr)
    generator, batches_per_epoch = paired_with_corruption_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        batch_size=batch_size,
        n_volumes=args.n_volumes)
    model.train(generator, batches_per_epoch, epochs=args.epochs)

def evaluate_discriminator():
    model = SliceDiscriminator(name=args.name, load=True)

    csfn_generator, csfn_batches_per_epoch = eval_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        'csfn',
        batch_size=50,
        n_volumes=args.n_volumes)
    wmn_generator, wmn_batches_per_epoch = eval_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        'wmn',
        batch_size=50,
        n_volumes=args.n_volumes)
    csfn_corrupted_generator, csfn_corr_batches_per_epoch = eval_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        'csfn',
        corrupted=True,
        batch_size=50,
        n_volumes=args.n_volumes)
    wmn_corrupted_generator, wmn_corr_batches_per_epoch = eval_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        'wmn',
        corrupted=True,
        batch_size=50,
        n_volumes=args.n_volumes)

    print('Evaluating on csfn and csfn_corrupted')
    model.eval(csfn_generator, csfn_batches_per_epoch, csfn_corrupted_generator, csfn_corr_batches_per_epoch)
    print('Evaluating on wmn and wmn_corrupted')
    model.eval(wmn_generator, wmn_batches_per_epoch, wmn_corrupted_generator, wmn_corr_batches_per_epoch)

num_gpus = 4
if args.gpu:
    num_gpus = len(args.gpu.split(','))
def main(multi_gpu=True):
    if args.viz:
        viz()
    elif args.mode == 'eval-generator':
        evaluate_generator()
    elif args.mode == 'eval-discriminator':
        evaluate_discriminator()
    elif args.mode == 'train-generator':
        if args.gan:
            train_gan()
        else:
            train_generator(batch_size=12 if num_gpus > 1 else 10)
    elif args.mode == 'train-discriminator':
        train_discriminator(batch_size=20)
    elif args.mode == 'train-paired-discriminator':
        train_paired_discriminator()
    elif args.mode == 'batch-eval':
        batch_eval()
    else:
        print('Command not recognized.')

if __name__ == '__main__':
    if args.multi_gpu:
        with MirroredStrategy().scope():
            main()
    else: 
        main(multi_gpu=False)

