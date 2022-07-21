import os
import sys
sys.path.append(os.path.abspath('.'))
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from utils.utils import AverageMeter, save, d_save
from models.patchnce import PatchNCELoss
from torch import optim

def train_generator_domain(
    source_cnn,
    generator,
    projector,
    discriminator,
    criterion,
    criterionGAN,
    g_optimizer,
    d_optimizer,
    source_train_loader,
    target_train_loader,
    target_test_loader,
    logger,
    args=None
):
    validation = validate(source_cnn, generator, target_test_loader, criterion, args=args)
    log_source = 'Source/Acc {:.3f} '.format(validation['avgAcc'])

    try:
        best_score = None
        best_class_score = None
        for epoch_i in range(1, 1 + args.epochs):
            start_time = time()
            training = generative_adversarial_domain(
                source_cnn, generator, projector, discriminator,
                source_train_loader, target_train_loader, target_test_loader,
                criterion, criterionGAN,
                g_optimizer, d_optimizer, 
                best_score, best_class_score, epoch_i, logger, args=args
            )
            best_score = training['best_score']
            best_class_score = training['best_class_score']
            n_iters = training['n_iters']
            validation = validate(
                target_cnn, target_test_loader, criterion, args=args)
            clsNames = validation['classNames']
            log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
            log += 'D/Loss {:.3f} Target/Loss {:.3f} '.format(
                training['d/loss'], training['target/loss'])
            log += '[Val] Target/Loss {:.3f} Target/Acc {:.3f} '.format(
                validation['loss'], validation['acc'])
            log += log_source
            log += 'Time {:.2f}s'.format(time() - start_time)
            logger.info(log)

            # save
            is_best = (best_score is None or validation['avgAcc'] > best_score)
            best_score = validation['avgAcc'] if is_best else best_score
            best_class_score = validation['classAcc'] if is_best else best_class_score
            state_dict = {
                'model': target_cnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_i,
                'val/acc': best_score,
            }
            d_state_dict = {
                'model': discriminator.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'epoch': epoch_i,
                'loss': training['d/loss'],
            }
            save(args.logdir, state_dict, is_best)
            d_save(args.logdir, d_state_dict)
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
    source_cnn, generator, projector, discriminator,
    source_loader, target_loader, target_test_loader,
    criterion, d_criterion,
    g_optimizer, d_optimizer, 
    best_score, best_class_score, epoch_i, logger, args=None
):
    source_cnn.eval()
    generator.train()
    projector.train()
    discriminator.train()

    best_score = best_score
    best_class_score = best_class_score

    g_losses, d_losses = AverageMeter(), AverageMeter()
    n_iters = min(len(source_loader), len(target_loader))
    valSteps = n_iters//args.num_val
    valStepsList = [valSteps+(x*valSteps) for x in range(args.num_val)]
    vals = valStepsList[:-1]
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

        #D_input_source = source_cnn.encoder(source_data)
        #D_input_target = target_cnn.encoder(target_data)
        #D_target_source = torch.tensor(
        #    [0] * bs, dtype=torch.long).to(args.device)
        #D_target_target = torch.tensor(
        #    [1] * bs, dtype=torch.long).to(args.device)

        # forward Generator
        G_input = torch.cat((target_data, source_data), dim=0)
        G_output = generator(G_input)
        G_output_source = G_output[bs:] # idt_B
        G_output_target = G_output[:bs] # fake_B

        # train Discriminator
        for param in discriminator.parameters():
            param.requires_grad = True
        d_optimizer.zero_grad()
        # compute D loss
        D_output_source = discriminator(source_data)
        D_output_target = discriminator(G_output_target.detach())
        #D_output = torch.cat([D_output_source, D_output_target], dim=0)
        #D_target = torch.cat([D_target_source, D_target_target], dim=0)
        d_loss = (d_criterion(D_output_source, True).mean() + d_criterion(D_output_target, False).mean()) * 0.5
        d_loss.backward()
        d_optimizer.step()
        d_losses.update(d_loss.item(), bs)

        # train Generator
        for param in discriminator.parameters():
            param.requires_grad = False
        g_optimizer.zero_grad()
        if iter_i > 0:
            p_optimizer.zero_grad()
        # compute G loss
        D_output_target = discriminator(G_output_target)
        loss_GAN = d_criterion(D_output_target, True).mean()
        loss_NCE = calculate_NCE_loss(generator, projector, target_data, G_output_target, args)
        # compute self training loss
        G_output_target_P = source_cnn(G_output_target)
        validSource = (target_domain == 0) & (target_conf >= args.thr)
        validMaskSource = validSource.nonzero(as_tuple=False)[:, 0]
        validTarget = (target_domain == 1) & (target_domain_conf <= args.thr_domain) & (target_conf >= args.thr)
        validMaskTarget = validTarget.nonzero(as_tuple=False)[:, 0]
        validIndexes = torch.cat((validMaskSource, validMaskTarget), 0)
        loss_ST = criterion(G_output_target_P[validIndexes], target_target[validIndexes])
        g_loss = loss_GAN + args.lam * loss_ST + args.lam_NCE * loss_NCE
        g_loss.backward()
        if iter_i == 0:
            p_optimizer = optim.Adam(projector.parameters(), lr=args.g_lr, betas=args.betas, weight_decay=args.weight_decay)
        g_optimizer.step()
        p_optimizer.step()
        g_losses.update(g_loss.item(), bs)
        if iter_i in vals:
            validation = validate(
                source_cnn, generator, target_test_loader, 
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
            source_cnn.eval()
            generator.train()
            projector.train()
            discriminator.train()

    return {'d/loss': d_losses.avg, 'target/loss': losses.avg, 'best_score': best_score, 'best_class_score': best_class_score, 'n_iters': n_iters}


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
    nce_layers = [0, 4, 8, 12, 16]
    criterion = PatchNCELoss(args).to(args.device) 
    n_layers = len(nce_layers)
    
    feat_q = generator(tgt, nce_layers, encode_only=True)
    feat_k = generator(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = projector(feat_k, 256, None)
    feat_q_pool, _ = projector(feat_q, 256, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, nce_layers):
        loss = criterion(f_q, f_k) * args.lam_NCE
        total_nce_loss += loss.mean()
    return total_nce_loss / n_layers

