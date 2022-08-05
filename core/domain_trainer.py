import os
import sys
sys.path.append(os.path.abspath('.'))
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from utils.utils import AverageMeter, save, d_save, tensor2im
from models.patchnce import PatchNCELoss
from torch import optim
from models import networks

def normalize(x, mean=(0.4017, 0.3791, 0.3656), std=(0.2093, 0.2019, 0.1996)):
    x[:, 0] -= mean[0]
    x[:, 1] -= mean[1]
    x[:, 2] -= mean[2]
    x[:, 0] /= std[0]
    x[:, 1] /= std[1]
    x[:, 2] /= std[2]
    return x

def train_generator_domain(
    source_cnn,
    target_cnn,
    generator,
    projector,
    sample_discriminator,
    feature_discriminator,
    criterion,
    sd_criterion,
    fd_criterion,
    g_optimizer,
    sd_optimizer,
    fd_optimizer,
    optimizer,
    source_train_loader,
    target_train_loader,
    target_test_loader,
    logger,
    wandb,
    args=None
):
    validation = validate(source_cnn, generator, target_test_loader, criterion, args=args)
    log_source = 'Source/Acc {:.3f} '.format(validation['avgAcc'])

    try:
        best_score = None
        best_class_score = None
        p_optimizer = None

        for epoch_i in range(1, 1 + args.epochs):
            # training
            start_time = time()
            training = generative_adversarial_domain(
                source_cnn, target_cnn, generator, projector, sample_discriminator, feature_discriminator,
                source_train_loader, target_train_loader, target_test_loader,
                criterion, sd_criterion, fd_criterion,
                g_optimizer, p_optimizer, sd_optimizer, fd_optimizer, optimizer,
                best_score, best_class_score, epoch_i, 
                logger, wandb, args=args
            )
            best_score = training['best_score']
            best_class_score = training['best_class_score']
            n_iters = training['n_iters']
            p_optimizer = training['p_optimizer']
            
            # validation
            validation = validate(
                target_cnn, generator, target_test_loader, criterion, args=args)
            clsNames = validation['classNames']
            log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
            log += 'sD/Loss {:.3f} fD/Loss {:.3f} G/Loss {:.3f} '.format(
                training['sd/loss'], training['fd/loss'], training['g/loss'])
            log += '[Val] Target/Loss {:.3f} Target/Acc {:.3f} '.format(
                validation['loss'], validation['acc'])
            log += log_source
            log += 'Time {:.2f}s'.format(time() - start_time)
            logger.info(log)
            # track validation metrics
            wandb.log({
                'val/loss': validation['loss'],
                'val/acc': validation['acc'],
                'val/avg_acc': validation['avgAcc'],
            })

            # save
            is_best = (best_score is None or validation['avgAcc'] > best_score)
            best_score = validation['avgAcc'] if is_best else best_score
            best_class_score = validation['classAcc'] if is_best else best_class_score
            state_dict = {
                'model': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
                'epoch': epoch_i,
                'val/acc': best_score,
            }
            save(args.logdir, state_dict, is_best)
            for cls_idx, clss in enumerate(clsNames):
                logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
            logger.info('Current val. acc.: {}'.format(validation['avgAcc']))
            logger.info('Best val. acc.: {}'.format(best_score))
            classWiseDict = {}
            for cls_idx, clss in enumerate(clsNames):
                classWiseDict[clss] = validation['classAcc'][cls_idx].item()
    except KeyboardInterrupt as ke:
        logger.info('\n============ Summary ============= \n')
        logger.info('Classwise accuracies: {}'.format(best_class_score))
        logger.info('Best val. acc.: {}'.format(best_score))
    
    return best_score, best_class_score, clsNames


def generative_adversarial_domain(
    source_cnn, target_cnn, generator, projector, sample_discriminator, feature_discriminator, 
    source_loader, target_loader, target_test_loader,
    criterion, sd_criterion, fd_criterion,
    g_optimizer, p_optimizer, sd_optimizer, fd_optimizer, optimizer,
    best_score, best_class_score, epoch_i, 
    logger, wandb, args=None
):
    source_cnn.eval()
    target_cnn.encoder.train()
    generator.train()
    projector.train()
    sample_discriminator.train()
    feature_discriminator.train()
    
    best_score = best_score
    best_class_score = best_class_score

    g_losses, sd_losses, fd_losses = AverageMeter(), AverageMeter(), AverageMeter()
    n_iters = min(len(source_loader), len(target_loader))
    valSteps = n_iters//args.num_val
    valStepsList = [valSteps+(x*valSteps) for x in range(args.num_val)]
    vals = valStepsList[:-1]
    visualSteps = n_iters//args.num_visual
    visualStepsList = [visualSteps+(x*visualSteps) for x in range(args.num_visual)]
    visuals = visualStepsList[:-1]

    source_iter, target_iter = iter(source_loader), iter(target_loader)
    for iter_i in range(n_iters):
        source_data, source_target = source_iter.next() # rgb
        target_data, target_target, target_conf, target_domain, target_domain_conf = target_iter.next() # thermal
        source_data = source_data.to(args.device)  # real_B
        source_target = source_target.to(args.device)
        target_data = target_data.to(args.device)  # real_A
        target_target = target_target.to(args.device)
        target_conf = target_conf.to(args.device)
        target_domain = target_domain.to(args.device)
        target_domain_conf = target_domain_conf.to(args.device)
        bs = source_data.size(0)

        
        # forward Generator
        #G_input = torch.cat((target_data, source_data), dim=0)
        #G_output = generator(G_input)
        #G_output_source = G_output[bs:] # idt_B
        #G_output_target = G_output[:bs] # fake_B
        G_output_target = generator(target_data)
        
        ## Train sample discriminator
        for param in sample_discriminator.parameters():
            param.requires_grad = True
        sd_optimizer.zero_grad()
        # compute D loss
        sd_output_source = sample_discriminator(source_data)
        sd_output_target = sample_discriminator(G_output_target.detach())
        sd_loss = (sd_criterion(sd_output_source, False).mean() + sd_criterion(sd_output_target, True).mean()) * 0.5
        sd_loss.backward()
        sd_optimizer.step()
        sd_losses.update(sd_loss.item(), bs)

        ## Train generator
        for param in sample_discriminator.parameters():
            param.requires_grad = False
        g_optimizer.zero_grad()
        if p_optimizer is not None:
            p_optimizer.zero_grad()
        # compute G loss
        sd_output_target = sample_discriminator(G_output_target)
        loss_sGAN = sd_criterion(sd_output_target, False).mean()
        loss_NCE = calculate_NCE_loss(generator, projector, target_data, G_output_target, args) 
        #+ calculate_NCE_loss(generator, projector, source_data, G_output_source, args)) * 0.5
        g_loss = loss_sGAN + args.lam_NCE * loss_NCE
        g_loss.backward()
        if p_optimizer is None:
            p_optimizer = optim.Adam(projector.parameters(), lr=args.g_lr, betas=args.betas)
            p_optimizer.zero_grad()
            print('p: ', networks.count_parameters(projector))
        g_optimizer.step()
        p_optimizer.step()
        g_losses.update(g_loss.item(), bs)
        
        ## Train feature discriminator
        fd_input_source = source_cnn.encoder(normalize(source_data))
        fd_input_target = target_cnn.encoder(G_output_target.detach())
        #fd_input_target = target_cnn.encoder(target_data)
        fd_target_source = torch.tensor(
            [0] * bs, dtype=torch.long).to(args.device)
        fd_target_target = torch.tensor(
            [1] * bs, dtype=torch.long).to(args.device)

        fd_output_source = feature_discriminator(fd_input_source)
        fd_output_target = feature_discriminator(fd_input_target)
        fd_output = torch.cat([fd_output_source, fd_output_target], dim=0)
        fd_target = torch.cat([fd_target_source, fd_target_target], dim=0)
        fd_loss = fd_criterion(fd_output, fd_target)
        fd_optimizer.zero_grad()
        fd_loss.backward()
        fd_optimizer.step()
        fd_losses.update(fd_loss.item(), bs)

        # Train target cnn
        fd_input_target = target_cnn.encoder(G_output_target.detach())
        #fd_input_target = target_cnn.encoder(target_data)
        fd_output_target = feature_discriminator(fd_input_target)
        loss_fGAN = fd_criterion(fd_output_target, fd_target_source)

        # compute self training loss
        '''
        fd_output_target_P = target_cnn.classifier(fd_input_target)
        validSource = (target_domain == 0) & (target_conf >= args.thr)
        validMaskSource = validSource.nonzero(as_tuple=False)[:, 0]
        validTarget = (target_domain == 1) & (target_domain_conf <= args.thr_domain) & (target_conf >= args.thr)
        validMaskTarget = validTarget.nonzero(as_tuple=False)[:, 0]
        validIndexes = torch.cat((validMaskSource, validMaskTarget), 0)
        loss_ST = criterion(fd_output_target_P[validIndexes], target_target[validIndexes])
        t_loss = loss_fGAN + args.lam * loss_ST
        '''
        optimizer.zero_grad()
        loss_fGAN.backward()
        optimizer.step()

        # track training metrics
        wandb.log({
            'train/sd_loss': sd_loss.item(),
            'train/g_loss_sgan': loss_sGAN.item(),
            'train/g_loss_nce': loss_NCE.item(),
            'train/g_loss': g_loss.item(),
            'train/fd_loss': fd_loss.item(),
            'train/loss_fgan': loss_fGAN.item(),
            #'train/loss_st' : loss_ST.item(),
            #'train/t_loss': t_loss.item()
        })

        if iter_i in visuals:
            wandb.log({
                'real_target': wandb.Image(tensor2im(target_data)),
                'fake_source': wandb.Image(tensor2im(G_output_target)),
                'real_source': wandb.Image(tensor2im(source_data)),
                #'idt_source': wandb.Image(tensor2im(G_output_source))
            })

        if iter_i in vals:
            validation = validate(
                target_cnn, generator, target_test_loader, 
                criterion, args=args)
            clsNames = validation['classNames']
            is_best = (best_score is None or validation['avgAcc'] > best_score)
            best_score = validation['avgAcc'] if is_best else best_score
            best_class_score = validation['classAcc'] if is_best else best_class_score
            state_dict = {
                'model': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
                'epoch': epoch_i,
                'val/acc': best_score,
            }
            save(args.logdir, state_dict, is_best)
            logger.info('Epoch_{} Iter_{}'.format(epoch_i, iter_i))
            for cls_idx, clss in enumerate(clsNames):
                logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
            logger.info('Current val. acc.: {}'.format(validation['avgAcc']))
            logger.info('Best val. acc.: {}'.format(best_score))
            classWiseDict = {}
            for cls_idx, clss in enumerate(clsNames):
                classWiseDict[clss] = validation['classAcc'][cls_idx].item()

            # track validation metrics
            wandb.log({
                'val/loss': validation['loss'],
                'val/acc': validation['acc'],
                'val/avg_acc': validation['avgAcc'],
            })
            source_cnn.eval()
            target_cnn.encoder.train()
            generator.train()
            projector.train()
            sample_discriminator.train()
            feature_discriminator.train()

    return {'p_optimizer': p_optimizer, 'sd/loss': sd_losses.avg, 'fd/loss': fd_losses.avg, 'g/loss': g_losses.avg, 'best_score': best_score, 'best_class_score': best_class_score, 'n_iters': n_iters}


def step(model, generator, data, target, criterion, args):
    data, target = data.to(args.device), target.to(args.device)
    output = model(generator(data))
    loss = criterion(output, target)
    return output, loss


def validate(model, generator, dataloader, criterion, args=None):
    model.eval()
    generator.eval()
    losses = AverageMeter()
    targets, probas = [], []
    if args.classInfo == None:
        classes = torch.unique(torch.tensor(dataloader.dataset.targets))
        classNames = dataloader.dataset.classes
    else:
        classes = args.classInfo['classes']
        classNames = args.classInfo['classNames']
    class_acc = torch.zeros(len(classes))
    class_len = torch.zeros(len(classes))
    acc_ev = 0
    with torch.no_grad():
        for iter_i, (data, target) in enumerate(dataloader):
            bs = target.size(0)
            output, loss = step(model, generator, data, target, criterion, args)
            pred_cls = output.data.max(1)[1]
            acc_ev += pred_cls.cpu().eq(target.data).cpu().sum()
            for class_idx, class_id in enumerate(classes):
                idxes = torch.nonzero(target==class_id.to(target.device), as_tuple=False)
                class_acc[class_idx] += pred_cls[idxes].cpu().eq(target[idxes].data).cpu().sum()
                class_len[class_idx] += len(idxes)
            output = torch.softmax(output, dim=1)
            losses.update(loss.item(), bs)
            targets.extend(target.cpu().numpy().tolist())
            probas.extend(output.cpu().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)
    acc = accuracy_score(targets, preds)
    class_acc /= class_len
    avgAcc = 0.0
    for i in range(len(class_acc)):
        avgAcc += class_acc[i]
    avgAcc = avgAcc / len(class_acc)
    return {
        'loss': losses.avg, 'acc': acc, 'avgAcc': avgAcc, 'classAcc': class_acc, 'classNames': classNames,
    }

def calculate_NCE_loss(generator, projector, src, tgt, args):
    nce_layers = [0, 4, 8, 12]
    criterion = PatchNCELoss(args).to(args.device) 
    n_layers = len(nce_layers)
    
    feat_q = generator(tgt, nce_layers, encode_only=True)
    feat_k = generator(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = projector(feat_k, args.num_patches, None)
    feat_q_pool, _ = projector(feat_q, args.num_patches, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, nce_layers):
        loss = criterion(f_q, f_k) * args.lam_NCE
        total_nce_loss += loss.mean()
    return total_nce_loss / n_layers

