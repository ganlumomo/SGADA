import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet50off import CNN, Discriminator
from core.trainer import train_target_cnnP_domain, validate_pseudo_label
from utils.utils import get_logger
from utils.altutils import get_mscoco, get_flir, get_flir_from_list_wdomain
from utils.altutils import setLogger
import logging


def run(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'train_sgada.log'))
    logger.info(args)

    # data loaders
    #dataset_root = os.environ["DATASETDIR"]
    dataset_root = './dataset_dir/'
    source_train_loader = get_mscoco(dataset_root, args.batch_size, train=True)
    source_val_loader = get_mscoco(dataset_root, args.batch_size, train=False)
    target_train_loader, target_train_path = get_flir(dataset_root, args.batch_size, train=True)
    target_val_loader, _ = get_flir(dataset_root, args.batch_size, train=False)

    args.classInfo = {'classes': torch.unique(torch.tensor(target_train_loader.dataset.targets)),
                    'classNames': target_train_loader.dataset.classes}

    logger.info('Testing')

    # load target CNN
    target_cnn = CNN(in_channels=args.in_channels, target=True, srcTrain=False).to(args.device)
    if os.path.isfile(args.trained):
        c = torch.load(args.trained)
        target_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.trained))

    # load discriminator
    discriminator = Discriminator(args=args).to(args.device)
    if os.path.isfile(args.d_trained):
        c = torch.load(args.d_trained)
        discriminator.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.d_trained))
    
    # generate label files
    criterion = nn.CrossEntropyLoss()
    validation = validate_pseudo_label(target_cnn, discriminator, target_val_loader, None, criterion, args=args)
    clsNames = validation['classNames']
    for cls_idx, clss in enumerate(clsNames):
        logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
    log = '[Val] Target/Loss {:.4f} Target/Acc {:.4f} '.format(
        validation['loss'], validation['acc'])
    logger.info(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--d_trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    parser.add_argument('--lam', type=float, default=0.25)
    parser.add_argument('--thr', type=float, default=0.79)
    parser.add_argument('--thr_domain', type=float, default=0.87)
    parser.add_argument('--num_val', type=int, default=3)  # number of val. within each epoch
    # misc
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/sgada_domain')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--tgt_conf_cat', type=str, default='flir_confident')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    run(args)
