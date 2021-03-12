import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser(description='Train and/or evaluate WMn --> CSFn model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', dest='name', default='model', help='Model name to save/load')
parser.add_argument('--mode', dest='mode', default=None, choices=['train-discriminator', 'train-generator', 'eval-discriminator', 'eval-generator', 'batch-eval'])
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Epochs to train for, if training')
parser.add_argument('--gpu', dest='gpu', default=None, help='Passed to CUDA_VISIBLE_DEVICES')
parser.add_argument('--perceptual-loss', dest='perceptual_loss', action='store_true', help='Custom perceptual loss for training')
parser.add_argument('--vgg-perceptual-loss', dest='vgg_perceptual_loss', action='store_true', help='VGG16-based perceptual loss for training')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Adam learning rate')
parser.add_argument('--no-normalize', dest='normalize', action='store_false', help=argparse.SUPPRESS)
parser.add_argument('--single-gpu', dest='multi_gpu', action='store_false', help=argparse.SUPPRESS)
parser.add_argument('--n-volumes', dest='n_volumes', type=int, default=60, help=argparse.SUPPRESS)
parser.add_argument('--dynamic-load', dest='preload_data', action='store_false', help=argparse.SUPPRESS)
parser.add_argument('--in', dest='infile', default=None, help='Volume to load for generator evaluation')
parser.add_argument('--out', dest='outfile', default=None, help='Output save path for generator evaluation')

args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy

from datagen import paired_generator, corrupted_with_wmn_generator, eval_generator
from models.slice_generator import SliceGenerator
from models.slice_discriminator import SliceDiscriminator

def evaluate_generator():
    model = SliceGenerator(name=args.name, load=True, normalize=args.normalize)
    if args.infile and args.outfile:
        model.convert_from_path(args.infile, out_path=args.outfile, mode=args.conversion_mode)

def train_generator(batch_size=10):
    model = SliceGenerator(
        name=args.name,
        vgg_perceptual_loss=args.vgg_perceptual_loss,
        perceptual_loss=args.perceptual_loss,
        lr=args.lr
    )
    generator, batches_per_epoch = paired_generator(
        '/data/mradovan/7T_WMn_3T_CSFn_pairs',
        batch_size=batch_size,
        n_volumes=args.n_volumes
    )
    model.train(generator, batches_per_epoch, epochs=args.epochs)

def batch_eval():
    model = SliceGenerator(name=args.name, load=True, perceptual_loss=args.perceptual_loss, normalize=args.normalize)
    path = '/data/mradovan/7T_WMn_3T_CSFn_pairs'
    input_vol = 'WMn.nii.gz'
    output_vol = 'CSFn_predicted.nii.gz'
    for subj_dir in glob.glob(path + '/*'):
        subj_input = '{}/{}'.format(subj_dir, input_vol)
        subj_output = '{}/{}'.format(subj_dir, output_vol)
        # if not os.path.exists(subj_output):
        model.convert_from_path(subj_input, subj_output, mode=args.conversion_mode)

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

def main(multi_gpu=True):
    if args.mode == 'eval-generator':
        evaluate_generator()
    elif args.mode == 'eval-discriminator':
        evaluate_discriminator()
    elif args.mode == 'train-generator':
        train_generator(batch_size=12)
    elif args.mode == 'train-discriminator':
        train_discriminator(batch_size=20)
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

