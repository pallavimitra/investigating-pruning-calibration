from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable

import models


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--save_dir', default='logs/', type=str)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('==> Preparing dataset %s' % args.dataset)
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100



    
dataset = dataloader(root='./data', train=True,
                                        download=True, transform=transform_train)

train_set, val_set = torch.utils.data.random_split(dataset, [45000, 5000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                              shuffle=True, **kwargs) 

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

history_score = np.zeros((args.epochs - args.start_epoch + 1, 3))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.data
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = train_acc / float(len(train_loader))

def val():
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return correct / float(len(val_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'finetune.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'finetune.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    train(epoch)
    prec1 = val()
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict()
    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')

def test(test_loader, model):
    model.eval()
    with torch.no_grad():
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

# Data

test_data = dataloader(root='./data', train=False, download=True, transform=transform_test)  

dataset_root = 'PATH TO CIFAR-10-C'
# All the distortions
distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                        'defocus_blur', 'glass_blur', 'motion_blur',
                        'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform',
                        'pixelate']
test_acc_1 = []
ECE_1 = []
MCE_1 = []
for distortion_name in distortions:
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))

        test_data.data = test_data.data[0:10000]
        test_data.targets =test_data.targets[0:10000]

        testloader = data.DataLoader(test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
            
        test_acc0, preds_0, labels_oneh_0  = test(testloader, model)
        ECE_0, MCE_0 = get_metrics(preds_0, labels_oneh_0)
        
        
        test_acc_1.append(test_acc0)
        ECE_1.append(ECE_0)
        MCE_1.append(MCE_0)
        

avg_test_acc_1 = sum(test_acc_1)/len(test_acc_1)
avg_ECE_1 = sum(ECE_1)/len(ECE_1)
avg_MCE_1 = sum(MCE_1)/len(MCE_1) 



    
with open(os.path.join(args.save_dir, 'result_corrupted_sev1.txt'), 'w') as f:
    f.write('After pruning: Average Test Acc:  %.6f\n' % (avg_test_acc_1))
    f.write('After Pruning: average ECE:  %.4f\n, average MCE: %.4f\n' % (avg_ECE_1, avg_MCE_1))

test_acc_2 = []
ECE_2 = []
MCE_2 = []
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
        
        test_acc_2.append(test_acc0)
        ECE_2.append(ECE_0)
        MCE_2.append(MCE_0)
        
    #test_acc = np.array(test_acc).flatten()
    #ECE = np.array(ECE).flatten()
    #MCE = np.array(MCE).flatten()
avg_test_acc_2 = sum(test_acc_2)/len(test_acc_2)
avg_ECE_2 = sum(ECE_2)/len(ECE_2)
avg_MCE_2 = sum(MCE_2)/len(MCE_2) 

#print('Before pruning: average test accuracy:  %.4f' % (avg_test_acc))
#print('Before pruning: average ECE:  %.4f, average MCE:  %.4f' % (avg_ECE, avg_MCE))

    
with open(os.path.join(args.save_dir, 'result_corrupted_sev2.txt'), 'w') as f:
    f.write('After pruning: Average Test Acc:  %.6f\n' % (avg_test_acc_2))
    f.write('After Pruning: average ECE:  %.4f\n, average MCE: %.4f\n' % (avg_ECE_2, avg_MCE_2))

    
test_acc_3 = []
ECE_3 = []
MCE_3 = []
for distortion_name in distortions:
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))

        test_data.data = test_data.data[20000:30000]
        test_data.targets =test_data.targets[20000:30000]

        testloader = data.DataLoader(test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
            
        test_acc0, preds_0, labels_oneh_0  = test(testloader, model)
        ECE_0, MCE_0 = get_metrics(preds_0, labels_oneh_0)
        
        #test_acc0 = test_acc0.cpu().detach().numpy()
        #ECE_0 = ECE_0.cpu().detach().numpy()
        #MCE_0 = MCE_0.cpu().detach().numpy()
        
        test_acc_3.append(test_acc0)
        ECE_3.append(ECE_0)
        MCE_3.append(MCE_0)
        
    #test_acc = np.array(test_acc).flatten()
    #ECE = np.array(ECE).flatten()
    #MCE = np.array(MCE).flatten()
avg_test_acc_3 = sum(test_acc_3)/len(test_acc_3)
avg_ECE_3 = sum(ECE_3)/len(ECE_3)
avg_MCE_3 = sum(MCE_3)/len(MCE_3) 

#print('Before pruning: average test accuracy:  %.4f' % (avg_test_acc))
#print('Before pruning: average ECE:  %.4f, average MCE:  %.4f' % (avg_ECE, avg_MCE))

    
with open(os.path.join(args.save_dir, 'result_corrupted_sev3.txt'), 'w') as f:
    f.write('After pruning: Average Test Acc:  %.6f\n' % (avg_test_acc_3))
    f.write('After Pruning: average ECE:  %.4f\n, average MCE: %.4f\n' % (avg_ECE_3, avg_MCE_3))
    
test_acc_4 = []
ECE_4 = []
MCE_4 = []
for distortion_name in distortions:
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))

        test_data.data = test_data.data[30000:40000]
        test_data.targets =test_data.targets[30000:40000]

        testloader = data.DataLoader(test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
            
        test_acc0, preds_0, labels_oneh_0  = test(testloader, model)
        ECE_0, MCE_0 = get_metrics(preds_0, labels_oneh_0)
        
        #test_acc0 = test_acc0.cpu().detach().numpy()
        #ECE_0 = ECE_0.cpu().detach().numpy()
        #MCE_0 = MCE_0.cpu().detach().numpy()
        
        test_acc_4.append(test_acc0)
        ECE_4.append(ECE_0)
        MCE_4.append(MCE_0)
        
    #test_acc = np.array(test_acc).flatten()
    #ECE = np.array(ECE).flatten()
    #MCE = np.array(MCE).flatten()
avg_test_acc_4 = sum(test_acc_4)/len(test_acc_4)
avg_ECE_4 = sum(ECE_4)/len(ECE_4)
avg_MCE_4 = sum(MCE_4)/len(MCE_4) 

#print('Before pruning: average test accuracy:  %.4f' % (avg_test_acc))
#print('Before pruning: average ECE:  %.4f, average MCE:  %.4f' % (avg_ECE, avg_MCE))

    
with open(os.path.join(args.save_dir, 'result_corrupted_sev4.txt'), 'w') as f:
    f.write('After pruning: Average Test Acc:  %.6f\n' % (avg_test_acc_4))
    f.write('After Pruning: average ECE:  %.4f\n, average MCE: %.4f\n' % (avg_ECE_4, avg_MCE_4))


test_acc_5 = []
ECE_5 = []
MCE_5 = []
for distortion_name in distortions:
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))

        test_data.data = test_data.data[40000:50000]
        test_data.targets =test_data.targets[40000:50000]

        testloader = data.DataLoader(test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
            
        test_acc0, preds_0, labels_oneh_0  = test(testloader, model)
        ECE_0, MCE_0 = get_metrics(preds_0, labels_oneh_0)
        
        #test_acc0 = test_acc0.cpu().detach().numpy()
        #ECE_0 = ECE_0.cpu().detach().numpy()
        #MCE_0 = MCE_0.cpu().detach().numpy()
        
        test_acc_5.append(test_acc0)
        ECE_5.append(ECE_0)
        MCE_5.append(MCE_0)
        
    #test_acc = np.array(test_acc).flatten()
    #ECE = np.array(ECE).flatten()
    #MCE = np.array(MCE).flatten()
avg_test_acc_5 = sum(test_acc_5)/len(test_acc_5)
avg_ECE_5 = sum(ECE_5)/len(ECE_5)
avg_MCE_5 = sum(MCE_5)/len(MCE_5) 

#print('Before pruning: average test accuracy:  %.4f' % (avg_test_acc))
#print('Before pruning: average ECE:  %.4f, average MCE:  %.4f' % (avg_ECE, avg_MCE))

    
with open(os.path.join(args.save_dir, 'result_corrupted_sev5.txt'), 'w') as f:
    f.write('After pruning: Average Test Acc:  %.6f\n' % (avg_test_acc_5))
    f.write('After Pruning: average ECE:  %.4f\n, average MCE: %.4f\n' % (avg_ECE_5, avg_MCE_5))
