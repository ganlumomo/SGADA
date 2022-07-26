import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet50off import CNN, Discriminator
from models import networks
from core.domain_trainer import train_generator_domain
from utils.utils import get_logger
from utils.altutils import get_mscoco, get_flir, get_flir_from_list_wdomain
from utils.altutils import setLogger
import logging
import wandb


def run(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'train_sgada.log'))
    logger.info(args)
    wandb.init(
        project='domain-exp-classification', 
        config={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
        })

    # data loaders
    #dataset_root = os.environ["DATASETDIR"]
    dataset_root = './dataset_dir/'
    source_train_loader = get_mscoco(dataset_root, args.batch_size, train=True)
    target_train_loader = get_flir(dataset_root, args.batch_size, train=True)
    target_val_loader = get_flir(dataset_root, args.batch_size, train=False)
    target_conf_train_loader = get_flir_from_list_wdomain(dataset_root, args.batch_size, train=True)

    args.classInfo = {'classes': torch.unique(torch.tensor(source_train_loader.dataset.targets)),
                    'classNames': source_train_loader.dataset.classes}

    logger.info('SGADA training')

    # train source CNN
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    if os.path.isfile(args.trained):
        c = torch.load(args.trained)
        source_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.trained))
    for param in source_cnn.parameters():
        param.requires_grad = False

    '''
    # train target CNN
    target_cnn = CNN(in_channels=args.in_channels, target=True, srcTrain=False).to(args.device)
    target_cnn.load_state_dict(source_cnn.state_dict())
    for param in target_cnn.classifier.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(
        target_cnn.encoder.parameters(), 
        lr=args.lr, betas=args.betas, 
        weight_decay=args.weight_decay)
    '''
    
    # train generator
    generator = networks.define_G(input_nc=args.in_channels, output_nc=args.in_channels, ngf=64,
        netG='resnet_9blocks', norm='instance', use_dropout=False,
        init_type='xavier', init_gain=0.02, no_antialias=False,
        no_antialias_up=False, gpu_ids=[int(args.device[-1])], opt=None)
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=args.g_lr, betas=args.betas)
    projector = networks.define_F(input_nc=args.in_channels, netF='mlp_sample', norm='instance',
        use_dropout=False, init_type='xavier', init_gain=0.02,
        no_antialias=False, gpu_ids=[int(args.device[-1])], opt=args)
    
    sample_discriminator = networks.define_D(input_nc=args.in_channels, ndf=64, netD='basic',
        n_layers_D=3, norm='instance', init_type='xavier',
        init_gain=0.02, no_antialias=False, gpu_ids=[int(args.device[-1])], opt=args)
    feature_discriminator = Discriminator(args=args).to(args.device)
    sd_optimizer = optim.Adam(
        sample_discriminator.parameters(),
        lr=args.d_lr, betas=args.betas)
    fd_optimizer = optim.Adam(
        feature_discriminator.parameters(),
        lr=args.d_lr, betas=args.betas)

    criterion = nn.CrossEntropyLoss()
    sd_criterion = networks.GANLoss('lsgan').to(args.device)
    fd_criterion = nn.CrossEntropyLoss()
    best_acc, best_class, classNames = train_generator_domain(
        source_cnn, generator, projector, sample_discriminator, feature_discriminator,
        criterion, sd_criterion, fd_criterion, g_optimizer, sd_optimizer, fd_optimizer,
        source_train_loader, target_conf_train_loader, target_val_loader,
        logger, wandb, args=args)
    bestClassWiseDict = {}
    for cls_idx, clss in enumerate(classNames):
        bestClassWiseDict[clss] = best_class[cls_idx].item()
    logger.info('Best acc.: {}'.format(best_acc))
    logger.info('Best acc. (Classwise):')
    logger.info(bestClassWiseDict)
    
    return best_acc, bestClassWiseDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--g_lr', type=float, default=1e-3)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    parser.add_argument('--netF_nc', type=int, default=256)
    parser.add_argument('--num_patches', type=int, default=256)
    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    parser.add_argument('--lam', type=float, default=0.25)
    parser.add_argument('--lam_NCE', type=float, default=1.0)
    parser.add_argument('--thr', type=float, default=0.79)
    parser.add_argument('--thr_domain', type=float, default=0.87)
    parser.add_argument('--num_val', type=int, default=6)  # number of val. within each epoch
    parser.add_argument('--num_visual', type=int, default=20)  # number of val. within each epoch
    # misc
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/domain_exp')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--tgt_conf_cat', type=str, default='flir_confident')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    run(args)
