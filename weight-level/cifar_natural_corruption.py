from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models.cifar as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='test_checkpoint/', type=str)
parser.add_argument('--percent', default=0.6, type=float, help='percentage of weight to prune')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    test_data = dataloader(root='./data', train=False, download=True, transform=transform_test)
   

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    #print('\nEvaluation only')
    dataset_root = 'PATH TO CIFAR-10-C'
    
    distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                        'defocus_blur', 'glass_blur', 'motion_blur',
                        'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform',
                        'pixelate']
    
    test_acc = []
    ECE = []
    MCE = []
    for distortion_name in distortions:
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))
        
        # first 10,000 images in each .npy are the test set images corrupted at severity 1, and the last 10,000 images are the test set images corrupted at severity five

        test_data.data = test_data.data[0:10000]
        test_data.targets =test_data.targets[0:10000]

        testloader = data.DataLoader(test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
            
        test_loss0, test_acc0, preds_0, labels_oneh_0  = test(testloader, model, criterion, start_epoch, use_cuda)
        ECE_0, MCE_0 = get_metrics(preds_0, labels_oneh_0)
        
      
        
        test_acc.append(test_acc0)
        ECE.append(ECE_0)
        MCE.append(MCE_0)
        
   
    avg_test_acc = sum(test_acc)/len(test_acc)
    avg_ECE = sum(ECE)/len(ECE)
    avg_MCE = sum(MCE)/len(MCE) 
    print('Before pruning: average test accuracy:  %.4f' % (avg_test_acc))
    print('Before pruning: average ECE:  %.4f, average MCE:  %.4f' % (avg_ECE, avg_MCE))

    
    with open(os.path.join(args.save_dir, 'result_corrupted.txt'), 'w') as f:
        f.write('After pruning: Average Test Acc:  %.2f\n' % (avg_test_acc))
        f.write('After Pruning: average ECE:  %.4f\n, average MCE: %.4f\n' % (avg_ECE, avg_MCE))
        
        
def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    
    with torch.no_grad():
        #number_corrects = 0
        #number_samples = 0
        preds = []
        labels_oneh = []
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
            
            
            y_predicted = model(inputs)
            #labels_predicted = y_predicted.argmax(axis = 1)
            #number_corrects += (labels_predicted==targets).sum().item()
            #number_samples += targets.size(0)
            
            sm = nn.Softmax(dim=1)
            pred = sm(y_predicted)
            _, predicted_cl = torch.max(pred.data, 1)
            pred = pred.cpu().detach().numpy()
            
            label_oneh = torch.nn.functional.one_hot(targets, num_classes=10)
            label_oneh = label_oneh.cpu().detach().numpy()
            
            preds.extend(pred)
            labels_oneh.extend(label_oneh)

          # Count correctly classified samples for accuracy
            correct += sum(predicted_cl == targets).item()

        preds = np.array(preds).flatten()
        labels_oneh = np.array(labels_oneh).flatten()
        

        correct_perc = correct / 10000
        print(f'Acc: {correct_perc}')
        #print(f'Overall accuracy {(number_corrects / number_samples)*100}%')
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

            # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, preds, labels_oneh)


def calc_bins(preds,labels_oneh):
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)
    
    #print(bin_accs)
    #print(bin_confs)
    #print(bin_sizes)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE



def draw_reliability_graph_after_pruning(preds, labels_oneh):
    ECE, MCE = get_metrics(preds, labels_oneh)
    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)
    
    #print(bin_accs)
    #print(bin_confs)
    #print(bin_sizes)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

  # x/y limits
    #ax.set_xlim(0, 1.05)
    #ax.set_ylim(0, 1)

  # x/y labels
    #plt.xlabel('Confidence')
    #plt.ylabel('Accuracy')

  # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

  # Error bars
    plt.bar(bins, bin_accs,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
    plt.bar(bins, bin_confs, width=0.1, alpha=0.5, edgecolor='black', color='b', hatch='//')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.4f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='black', label='MCE = {:.4f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  
    plt.savefig('./calibrated_network_after_pruning_corrupted.png', bbox_inches='tight')


def calibration_curve(preds, labels_oneh):
    
    
    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)
    x, y = bin_confs, bin_accs

    fig = plt.figure(figsize=(8, 8))
    #ax = fig.gca()
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(x, y, "s-", label="90% pruned")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("calibration curve for DENSENET40 on CIFAR 10")

    plt.tight_layout()
    #plt.show()
    plt.savefig('./calibration_curve_after_pruning_corrupted.png', bbox_inches='tight')
    
    
if __name__ == '__main__':
    main()
