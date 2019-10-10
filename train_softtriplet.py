import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from softtriple import train
from softtriple import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hdml with triplet loss.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('-s', '--image_size', type=int, default=227, help="The size of input images.")
    parser.add_argument('-l', '--lr', type=float, default=7e-5, help="Initial learning rate.")
    parser.add_argument('-m', '--max_steps', type=int, default=80000, help="The maximum step number.")
    parser.add_argument('-c', '--n_class', type=int, default=99, help="Number of classes.")
    parser.add_argument('-d', '--dataset', type=str, default='cars196', choices=['cars196', 'cub200_2011'], help='Choose a dataset.')
    parser.add_argument('-p', '--pretrained', type=str, default=None, help='Use pretrained weight.')
    args = parser.parse_args()
    if args.dataset == 'cars196':
        streams = dataset.get_streams('data/CARS196/cars196.hdf5', args.batch_size, crop_size=args.image_size)
    elif args.dataset == 'cub200_2011':
        streams = dataset.get_streams('data/cub200_2011/cub200_2011.hdf5', args.batch_size, crop_size=args.image_size)
    else:
        raise ValueError("`dataset` must be 'cars196' or 'cub200_2011'.")
    writer = SummaryWriter()
    train.train_softtriplet(streams, writer, args.max_steps, args.n_class, args.lr,
                            pretrained=args.pretrained,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    writer.close()