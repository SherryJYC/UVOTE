import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean
from datetime import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from resnet_var_morebranch import resnet50 as resnet50_var

from loss import *
from datasets_gradual import AgeDB
from utils import *
from ranksim import batchwise_ranking_regularizer

import os
os.environ["KMP_WARNINGS"] = "FALSE"


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# imbalanced related
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=9, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# more model setting
parser.add_argument('--num_branch', type=int, default=2)
parser.add_argument('--reweight', type=int, default=2, help='cost-sensitive reweighting scheme')
parser.add_argument('--dynamic_loss', action='store_true', default=False, help='whether to combine loss (like two) dynamically')

# batchwise ranking regularizer
parser.add_argument('--regularization_weight', type=float, default=0, help='weight of the regularization term')
parser.add_argument('--interpolation_lambda', type=float, default=1.0, help='interpolation strength')

# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False, help='whether to retrain last regression layer (regressor)')

# training/optimization related
parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['l1nll', 'l1', 'mse', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=90, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=4, help='number of workers used in data loading')
# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5

args.reweight = args.num_branch
print(f'reweight = {args.reweight} = num_branch = {args.num_branch}')
assert args.reweight >= 0, 'args.reweight & args.num_branch should >= 0'

if args.num_branch < 2:
    args.dynamic_loss = False
    print(f'num_branch = {args.num_branch}, so set dynamic_loss as False')

if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if args.reweight != 'none':
    args.store_name += f'_{args.reweight}'

if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
if args.dynamic_loss:
    args.store_name += f'_dyL'
if args.regularization_weight > 0:
    args.store_name += f'_reg{args.regularization_weight}_il{args.interpolation_lambda}'

# for evaluation
if 'nll' in args.loss:
    combine_sets = ['avg', 'avgvar', 'minvar', 'oracle']
else:
    combine_sets = ['avg', 'oracle']

date = datetime.now()
date = "{}-{}-{}-{}-{}-{}".format(date.day, date.month, date.year, date.hour, date.minute, date.second)
args.store_name = f"{date}_{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

# fix randomess
set_seed()
def main():
    if not args.evaluate:
        wandb.init(project='imbalanced-ensemble', name=args.store_name, config=args)

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train',
                        reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    model = resnet50_var(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth,
                     kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt, 
                     num_branch=args.num_branch, return_features=(args.regularization_weight > 0))
    model = torch.nn.DataParallel(model).cuda()

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")

        if args.num_branch <= 1:
            validate(test_loader, model, train_labels=train_labels, prefix='Test')
        else:
            print(f"Loaded latest model, epoch {checkpoint['epoch']}")
            for c in ['avg', 'avgvar', 'minvar', 'oracle']: #'avg', 'avgvar', 'minvar', 'oracle']:
                test_loss_mse, test_loss_l1, test_loss_gmean = validate_mul(test_loader, model, train_labels=train_labels, prefix='Test', 
                                                            combine=c)
                print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        parameters = []
        parameters += [{'params': [p for n, p in model.named_parameters() if 'logvar_linear' not in n and p.requires_grad],
                        'lr':     args.lr}]
        parameters += [{'params': [p for n, p in model.named_parameters() if 'logvar_linear' in n and p.requires_grad],
                        'lr':     args.lr * 0.1}]
        
        optimizer = torch.optim.Adam(parameters) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None, [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=train_labels, umetric=False if epoch < 10 else True)

        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        wandb.log({'train_loss': train_loss,
                   'val_loss_mse': val_loss_mse,
                   'val_loss_l1': val_loss_l1,
                   'val_loss_gmean': val_loss_gmean})

    # test with best checkpoint
    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    if args.num_branch <= 1:
        validate(test_loader, model, train_labels=train_labels, prefix='Test')
    else:
        for c in combine_sets:
            test_loss_mse, test_loss_l1, test_loss_gmean = validate_mul(test_loader, model, train_labels=train_labels, prefix='Test', 
                                                        combine=c)
            print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")

    print("=" * 120)
    print("Test latest model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded latest model, epoch {checkpoint['epoch']}")
    if args.num_branch <= 1:
        validate(test_loader, model, train_labels=train_labels, prefix='Test')
    else:
        for c in combine_sets:
            test_loss_mse, test_loss_l1, test_loss_gmean = validate_mul(test_loader, model, train_labels=train_labels, prefix='Test', 
                                                        combine=c)
            print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')

    mus_loglist, log_vars_loglist = [], []
    for i in range(args.num_branch):
        mus_loglist.append(AverageMeter(f'Mu{i}', ':6.2f'))
        log_vars_loglist.append(AverageMeter(f'Log_var{i}', ':6.2f'))

    alllogs = [batch_time, data_time, losses]
    alllogs.extend(mus_loglist)
    alllogs.extend(log_vars_loglist)

    progress = ProgressMeter(
        len(train_loader),
        alllogs,
        prefix="Epoch: [{}]".format(epoch)
    )

    # compute alpha for dynamic learning
    alpha = 1 - (epoch / args.epoch) ** 2

    # compute t2, t3 for posterior annealing
    a = 100
    r = 0.1
    t2 = a*(1-r)**epoch
    t3 = a*(1-r)**epoch

    model.train()
    end = time.time()
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets = \
            inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        if isinstance(weights, list):
            for i in range(len(weights)):
                weights[i] = weights[i].cuda(non_blocking=True)
        else:
            weights = weights.cuda(non_blocking=True)

        if args.regularization_weight > 0:
            outputs, features = model(inputs, targets, epoch)
        elif args.fds:
            outputs, _ = model(inputs, targets, epoch)
        else:
            outputs = model(inputs, targets, epoch)

        loss = None
        for i in range(len(weights)):
            mu, log_var = outputs[i]
            log_var = torch.clamp(log_var, min=-5, max=5)

            # update logging outputs
            mus_loglist[i].update(mu.mean())
            log_vars_loglist[i].update(log_var.mean())

            # compute loss
            if loss is None:
                loss = globals()[f"weighted_{args.loss}_loss"](mu, log_var, targets, weights[i])
            else:
                if args.dynamic_loss:
                    loss += (1 - alpha) * globals()[f"weighted_{args.loss}_loss"](mu, log_var, targets, weights[i])
                else:
                    loss += globals()[f"weighted_{args.loss}_loss"](mu, log_var, targets, weights[i])

        # additional loss for feature
        if args.regularization_weight > 0:
            loss += (args.regularization_weight * batchwise_ranking_regularizer(features, targets, 
                args.interpolation_lambda))

        if epoch > 1:
            assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg

def validate_mul(val_loader, model, train_labels=None, prefix='Val', 
            combine='minvar'):
    print(f'********* COMBINE BY {combine} ********* ')
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix=f'{prefix}: '
    )

    model.eval()
    preds, labels, stds = [], [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)

            mus, logvars = [], []
            for i in range(len(outputs)):
                mu, logvar = outputs[i]
                mus.append(mu.data.cpu().numpy())
                logvars.append(logvar.data.cpu().numpy())

            mus, logvars = np.array(mus).squeeze(), np.array(logvars).squeeze()
            # avg
            if combine == 'avg':
                mu = np.mean(mus, axis=0).reshape((-1, ))
                logvar = np.mean(logvars, axis=0).reshape((-1, ))

            # avgvar
            elif combine == 'avgvar':
                w = np.power(np.exp(logvars), -1)
                mu = np.sum(mus * w, axis=0) / np.sum(w, axis=0)
                mu = mu.reshape((-1, ))
                logvar = np.sum(logvars * w, axis=0) / np.sum(w, axis=0)
                logvar = logvar.reshape((-1, ))

            # minvar
            elif combine == 'minvar':
                idxs = np.expand_dims(np.argmin(logvars, axis=0), axis=0)
                mu = np.take_along_axis(mus, idxs, axis=0).reshape((-1, ))
                logvar = np.take_along_axis(logvars, idxs, axis=0).reshape((-1, ))
            elif combine == 'oracle':
                diff = np.abs(mus - targets.data.cpu().numpy().squeeze())
                idxs = np.expand_dims(np.argmin(diff, axis=0), axis=0)
                mu = np.take_along_axis(mus, idxs, axis=0).reshape((-1, ))
                logvar = np.take_along_axis(logvars, idxs, axis=0).reshape((-1, ))
            else:
                raise "combine not in ['avg', 'avgvar', 'minvar', 'oracle']"

            preds.extend(mu)
            labels.extend(targets.data.cpu().numpy())
            stds.extend(np.sqrt(2) * np.exp(logvar))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)

        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        
        diffs_array = np.hstack(preds) -  np.hstack(labels)
        mae = np.mean(np.abs(diffs_array))
        mse = np.mean(np.power(diffs_array, 2))
        overall_gmean = gmean(np.abs(diffs_array), axis=None).astype(float)

        print(f" * Overall: MSE {mse:.3f}\tL1 {mae:.3f}\tG-Mean {overall_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

        # eval uncertainty
        uce = measureUCE(np.hstack(preds), np.hstack(stds), np.hstack(labels))
        print(f" * UCE {uce:.3f}")

        # shot-based uncertainty evaluation
        labels, preds, stds = np.hstack(labels), np.hstack(preds), np.hstack(stds)
        for s in ['many', 'median', 'low']:
            l_unique = shot_dict[s]['label']
            use_mask = np.isin(labels, l_unique)
            uce = measureUCE(preds[use_mask], stds[use_mask], labels[use_mask])
            print(f" * {s} UCE {uce:.3f}")

    return mse, mae, overall_gmean

def validate(val_loader, model, train_labels=None, prefix='Val', umetric=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels, logvars = [], [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            mu, log_var = outputs[0]

            preds.extend(mu.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())
            logvars.extend(log_var.data.cpu().numpy())

            loss_mse = criterion_mse(mu, targets)
            loss_l1 = criterion_l1(mu, targets)
            loss_all = criterion_gmean(mu, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)

        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

        # eval uncertainty
        if umetric:
            stds = np.sqrt(2) * np.exp(np.hstack(logvars))
            # eval uncertainty
            uce = measureUCE(np.hstack(preds), stds, np.hstack(labels))
            print(f" * UCE {uce:.3f}")

            # shot-based uncertainty evaluation
            labels, preds, stds = np.hstack(labels), np.hstack(preds), np.hstack(stds)
            for s in ['many', 'median', 'low']:
                l_unique = shot_dict[s]['label']
                use_mask = np.isin(labels, l_unique)
                uce = measureUCE(preds[use_mask], stds[use_mask], labels[use_mask])
                print(f" * {s} UCE {uce:.3f}")


    return losses_mse.avg, losses_l1.avg, loss_gmean


def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    l_unique = np.unique(labels)

    for l in l_unique:
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []
    many_l, median_l, low_l = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
            many_l.append(l_unique[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
            low_l.append(l_unique[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])
            median_l.append(l_unique[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    shot_dict['many']['label'] = many_l
    shot_dict['median']['label'] = median_l
    shot_dict['low']['label'] = low_l

    return shot_dict

if __name__ == '__main__':
    main()
