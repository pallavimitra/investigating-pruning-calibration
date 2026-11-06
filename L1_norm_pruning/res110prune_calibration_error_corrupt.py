import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import torch.utils.data as data

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=110,
                    help='depth of the resnet')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)

if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')

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

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(preds, labels_oneh):
    ECE, MCE = get_metrics(preds, labels_oneh)
    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

  # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

  # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

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
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='black', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  
    plt.savefig('calibrated_network_before_pruning.png', bbox_inches='tight')

    


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(test_loader, model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    '''if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")'''
    model.eval()
    correct = 0
    preds = []
    labels_oneh = []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        sm = nn.Softmax(dim=1)
        pred_ = sm(output)
        pred_ = pred_.cpu().detach().numpy()
            
        label_oneh = torch.nn.functional.one_hot(target, num_classes=10)
        label_oneh = label_oneh.cpu().detach().numpy()
            
        preds.extend(pred_)
        labels_oneh.extend(label_oneh)
        
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset)), preds, labels_oneh

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
 
        test_data.data = test_data.data[10000:20000]
        test_data.targets =test_data.targets[10000:20000]

        testloader = data.DataLoader(test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
            
        test_acc0, preds_0, labels_oneh_0  = test(testloader, model)
        ECE_0, MCE_0 = get_metrics(preds_0, labels_oneh_0)
        
        #test_acc0 = test_acc0.cpu().detach().numpy()
        #ECE_0 = ECE_0.cpu().detach().numpy()
        #MCE_0 = MCE_0.cpu().detach().numpy()
        
        test_acc.append(test_acc0)
        ECE.append(ECE_0)
        MCE.append(MCE_0)
        
    #test_acc = np.array(test_acc).flatten()
    #ECE = np.array(ECE).flatten()
    #MCE = np.array(MCE).flatten()
avg_test_acc = sum(test_acc)/len(test_acc)
avg_ECE = sum(ECE)/len(ECE)
avg_MCE = sum(MCE)/len(MCE) 

print('Before pruning: average test accuracy:  %.4f' % (avg_test_acc))
print('Before pruning: average ECE:  %.4f, average MCE:  %.4f' % (avg_ECE, avg_MCE))
skip = {
    'A': [36],
    'B': [36, 38, 74],
}

prune_prob = {
    'A': [0.5, 0.0, 0.0],
    'B': [0.5, 0.4, 0.3],
}

layer_id = 1
cfg = []
cfg_mask = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        if layer_id in skip[args.v]:
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
            layer_id += 1
            continue
        if layer_id % 2 == 0:
            stage = layer_id // 36
            if layer_id <= 36:
                stage = 0
            elif layer_id <= 72:
                stage = 1
            elif layer_id <= 108:
                stage = 2
            prune_prob_stage = prune_prob[args.v][stage]
            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1,2,3))
            num_keep = int(out_channels * (1 - prune_prob_stage))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            cfg.append(num_keep)
            layer_id += 1
            continue
        layer_id += 1

newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue
    elif isinstance(m0, nn.BatchNorm2d):
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
model_prune = newmodel


num_parameters_before_pruning = sum([param.nelement() for param in model.parameters()])
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters before pruning: \n" + str(num_parameters_before_pruning) + "\n")
    fp.write("Number of parameters after pruning: \n"+str(num_parameters)+"\n")
    fp.write("Average Test accuracy before pruning: \n"+str(avg_test_acc)+"\n")
    fp.write("Avg_ECE before pruning: \n"+str(avg_ECE)+"\n")
    fp.write("AVG_MCE before pruning: \n"+str(avg_MCE)+"\n")