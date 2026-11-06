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


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
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
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
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
        raise ValueError("No valid dataset is given.")
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

acc, preds, labels_oneh = test(model)
ECE, MCE = get_metrics(preds, labels_oneh)
draw_reliability_graph(preds, labels_oneh)

print('Before pruning: ECEs:  %.4f, MCE:  %.4f' % (ECE, MCE))

skip = {
    'A': [36],
    'B': [36, 38, 74],
}
# For Skip connection A, change the probability at 1st stage from 0.1-0.7
prune_prob = {
    'A': [0.7, 0.0, 0.0],
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
acc_after_pruning,_,_ = test(model_prune)



num_parameters_before_pruning = sum([param.nelement() for param in model.parameters()])
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters before pruning: \n" + str(num_parameters_before_pruning) + "\n")
    fp.write("Number of parameters after pruning: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy before pruning: \n"+str(acc)+"\n")
    fp.write("Test accuracy after pruning: \n"+str(acc_after_pruning)+"\n")
    fp.write("ECE before pruning: \n"+str(ECE)+"\n")
    fp.write("MCE before pruning: \n"+str(MCE)+"\n")