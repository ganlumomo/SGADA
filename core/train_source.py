import argparse
import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision import transforms
from models.resnet50off import CNN
from core.trainer import train_source_cnn
from utils.utils import get_logger
from utils.altutils import get_mscoco, get_flir, get_flir_from_list_wdomain
from utils.altutils import setLogger
import logging


def main(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'train_source.log'))
    logger.info(args)

    # data loaders
    #dataset_root = os.environ["DATASETDIR"]
    dataset_root = './dataset_dir/'
    source_train_loader = get_mscoco(dataset_root, args.batch_size, train=True)
    source_val_loader = get_mscoco(dataset_root, args.batch_size, train=False)

    args.classInfo = {'classes': torch.unique(torch.tensor(source_train_loader.dataset.targets)),
                    'classNames': source_train_loader.dataset.classes}

    logger.info('Source training')

    # train source CNN
    source_cnn = CNN(in_channels=args.in_channels, srcTrain=True).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        source_cnn.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    source_cnn = train_source_cnn(
        source_cnn, source_train_loader, source_val_loader,
        criterion, optimizer, logger, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/train_source')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--tgt_conf_cat', type=str, default='flir_confident')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    main(args)
