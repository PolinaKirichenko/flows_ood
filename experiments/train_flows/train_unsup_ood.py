"""
Code adapted from https://github.com/chrischute/real-nvp, which is in turn
adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
from torch.nn import functional as F

import flow_ssl
from flow_ssl import FlowLoss
from tqdm import tqdm
from torch import distributions
import torch.nn as nn

from tensorboardX import SummaryWriter

from flow_ssl.data import make_sup_data_loaders
from sklearn.metrics import roc_auc_score


def linear_rampup(final_value, epoch, num_epochs, start_epoch=0):
    t = (epoch - start_epoch + 1) / num_epochs
    if t > 1:
        t = 1.
    return t * final_value


def draw_samples(net, writer, loss_fn, num_samples, device, img_shape, iter):
    images = utils.sample(net, loss_fn.prior, num_samples,
                          cls=None, device=device, sample_shape=img_shape)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    writer.add_image("samples/unsup", images_concat, iter)
    return images_concat


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, writer,
        num_samples=10, sampling=True, tb_freq=100):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    iter_count = 0
    batch_count = 0
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            iter_count += 1
            batch_count += x.size(0)
            x = x.to(device)
            optimizer.zero_grad()
            z = net(x)
            sldj = net.module.logdet()
            loss = loss_fn(z, sldj=sldj)
            loss.backward()
            utils.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

            if iter_count % tb_freq == 0 or batch_count == len(trainloader.dataset):
                tb_step = epoch*(len(trainloader.dataset))+batch_count
                writer.add_scalar("train/loss", loss_meter.avg, tb_step)
                writer.add_scalar("train/bpd", utils.bits_per_dim(x, loss_meter.avg), tb_step)
                if sampling:
                    net.eval()
                    draw_samples(net, writer, loss_fn, num_samples, device, tuple(x[0].shape), tb_step)
                    net.train()


def test(epoch, net, testloader, device, loss_fn, writer, 
        tb_name="test"):
    net.eval()
    loss_meter = utils.AverageMeter()
    loss_list = []
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z = net(x)
                sldj = net.module.logdet()
                losses = loss_fn(z, sldj=sldj, mean=False)
                loss_list.extend([loss.item() for loss in losses])
                
                loss = losses.mean()
                loss_meter.update(loss.item(), x.size(0))
                
                progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=utils.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    likelihoods = -torch.from_numpy(np.array(loss_list)).float()
    if writer is not None:
        writer.add_scalar("{}/loss".format(tb_name), loss_meter.avg, epoch)
        writer.add_scalar("{}/bpd".format(tb_name), utils.bits_per_dim(x, loss_meter.avg), epoch)
        writer.add_histogram('{}/likelihoods'.format(tb_name), likelihoods, epoch)
    return likelihoods

def get_percentile(arr, p=0.05):
    percentile_idx = int(len(arr) * 0.05)
    percentile = torch.sort(arr)[0][percentile_idx].item()
    return percentile


parser = argparse.ArgumentParser(description='RealNVP')

parser.add_argument('--dataset', type=str, default="CIFAR10", required=True, metavar='DATA',
                help='Dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')

parser.add_argument('--ood_dataset', type=str, default=None, required=False, metavar='DATA',
                help='OOD dataset name (default: None)')
parser.add_argument('--ood_data_path', type=str, default=None, required=False, metavar='PATH',
                help='path to ood datasets location (default: None)')

parser.add_argument('--logdir', type=str, default=None, required=True, metavar='PATH',
                help='path to log directory (default: None)')
parser.add_argument('--ckptdir', type=str, default=None, required=True, metavar='PATH',
                help='path to ckpt directory (default: None)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--num_samples', default=20, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--resume',  type=str, default=None, metavar='PATH', help='path to ckpt')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='L2 regularization (only applied to the weight norm scale factors)')
parser.add_argument('--use_validation', action='store_true', help='Use trainable validation set')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')
parser.add_argument('--lr_anneal', action='store_true')


#parser.add_argument('--flow', choices=['RealNVP', 'Glow', 'RealNVPNewMask', 'RealNVPNewMask2', 'RealNVPSmall'], default="RealNVP", help='Flow model to use (default: RealNVP)')
parser.add_argument('--flow', type=str, default="RealNVP", help='Flow model to use (default: RealNVP)')
parser.add_argument('--num_blocks', default=8, type=int, help='number of blocks in ResNet')
parser.add_argument('--num_scales', default=2, type=int, help='number of scales in multi-layer architecture')
parser.add_argument('--num_mid_channels', default=64, type=int, help='number of channels in coupling layer parametrizing network')
parser.add_argument('--st_type', choices=['highway', 'resnet', 'convnet', 'autoencoder_old', 'autoencoder', 'resnet_ae'], default='resnet')
parser.add_argument('--latent_dim', default=100, type=int, help='dim of bottleneck in autoencoder st-network')
parser.add_argument('--no_batchnorm', action='store_true')
parser.add_argument('--no_skip', action='store_true')
parser.add_argument('--aug', action='store_true')
parser.add_argument('--init_zeros', action='store_true')
parser.add_argument('--optim', choices=['Adam', 'RMSprop', 'AdamW'], default='Adam')

# Glow parameters
parser.add_argument('--num_coupling_layers_per_scale', default=8, type=int, help='number of coupling layers in one scale')
parser.add_argument('--no_multi_scale', action='store_true')

# for RealNVPTabular model
parser.add_argument('--no_sampling', action='store_true')
parser.add_argument('--dropout', action='store_true')


args = parser.parse_args()

def schedule(epoch):
    t = (epoch) / args.num_epochs
    if t <= 0.8:
        factor = 1.0
    elif t <=0.9:
        factor = 0.5
    else:
        factor = 0.5 ** 2
    return args.lr * factor

os.makedirs(args.ckptdir, exist_ok=True)
with open(os.path.join(args.ckptdir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

writer = SummaryWriter(log_dir=args.logdir)

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
start_epoch = 0

if args.dataset.lower() == "mnist" or args.dataset.lower() == "fashionmnist":
    img_shape = (1, 28, 28)
    if args.aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

elif args.dataset.lower() in ["cifar10", "svhn", "celeba"]:  # celeba has its own train_transform in make_sup_data_loaders
    img_shape = (3, 32, 32)
    if args.aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
elif args.dataset.lower() == 'numpy':
    transform_train = None
elif args.dataset.lower() in ["cifar10_transfer", "svhn_transfer", "celeba_transfer"]:  # features from pretrained EfficientNet
    feature_dim = 1792
    transform_train = transforms.Compose([transforms.ToTensor()])
else:
    raise ValueError("Unsupported dataset "+args.dataset)

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainloader, testloader, _ = make_sup_data_loaders(
        args.data_path, 
        args.batch_size, 
        args.num_workers, 
        transform_train, 
        transform_test, 
        use_validation=args.use_validation,
        shuffle_train=True,
        dataset=args.dataset.lower())

if args.ood_dataset:
    _, ood_testloader, _ = make_sup_data_loaders(
            args.ood_data_path, 
            args.batch_size, 
            args.num_workers, 
            transform_train, 
            transform_test, 
            use_validation=args.use_validation,
            shuffle_train=True,
            dataset=args.ood_dataset.lower())

if args.dataset.lower() == 'numpy':
    img_shape = testloader.dataset[0][0].shape
    print(img_shape)

# Model
print('Building {} model...'.format(args.flow))
model_cfg = getattr(flow_ssl, args.flow)
if args.flow == 'RealNVPTabular':
    net = model_cfg(in_dim=feature_dim, hidden_dim=args.num_mid_channels, num_layers=args.num_blocks,
                    num_coupling_layers=args.num_coupling_layers_per_scale, init_zeros=args.init_zeros, dropout=args.dropout)

elif 'RealNVP' in args.flow:
    net = model_cfg(in_channels=img_shape[0], init_zeros=args.init_zeros, mid_channels=args.num_mid_channels,
        num_blocks=args.num_blocks, num_scales=args.num_scales, st_type=args.st_type,
        use_batch_norm=not args.no_batchnorm, img_shape=img_shape, skip=not args.no_skip, latent_dim=args.latent_dim)

elif args.flow == 'Glow':
    net = model_cfg(image_shape=img_shape, mid_channels=args.num_mid_channels, num_scales=args.num_scales,
        num_coupling_layers_per_scale=args.num_coupling_layers_per_scale, num_layers=args.num_blocks,
        multi_scale=not args.no_multi_scale, st_type=args.st_type)
print("Model contains {} parameters".format(sum([p.numel() for p in net.parameters()])))


if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

if args.resume is not None:
    # Load checkpoint.
    print('Resuming from checkpoint at ckpts/best.pth.tar...')
    assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

D = np.prod(img_shape) if args.flow != 'RealNVPTabular' else feature_dim
D = int(D)

prior = distributions.MultivariateNormal(torch.zeros(D).to(device),
                                             torch.eye(D).to(device))

loss_fn = FlowLoss(prior)

if 'RealNVP' in args.flow and args.flow != 'RealNVPTabular':
    # We need this to make sure that weight decay is only applied to g -- norm parameter in Weight Normalization
    param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    if args.optim == 'Adam':
        optimizer = optim.Adam(param_groups, lr=args.lr)
    else:
        optimizer = optim.RMSprop(param_groups, lr=args.lr)

elif args.flow == 'Glow' or args.flow == 'RealNVPTabular':
    if args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(start_epoch, start_epoch + args.num_epochs + 1):
    if args.lr_anneal:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)


    train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm, writer, num_samples=args.num_samples)
    test_ll = test(epoch, net, testloader, device, loss_fn, writer)
    test_ll_percentile = get_percentile(test_ll)
    test_ll = test_ll.cpu().detach().numpy()

    if args.ood_dataset:
        ood_ll = test(epoch, net, ood_testloader, device, loss_fn, writer, tb_name="ood")
        ood_ll_percentile = get_percentile(ood_ll)
        ood_ll = ood_ll.cpu().detach().numpy()
        # AUC-ROC
        n_ood, n_test = len(ood_ll), len(test_ll)
        lls = np.hstack([ood_ll, test_ll])
        targets = np.ones((n_ood + n_test,), dtype=int)
        targets[:n_ood] = 0
        score = roc_auc_score(targets, lls)
        writer.add_scalar("ood/roc_auc", score, epoch)
        

    # plotting likelihood hists
    fig = plt.figure(figsize=(8, 8))
    sns.distplot(test_ll[test_ll > test_ll_percentile], label='test')
    if args.ood_dataset:
        sns.distplot(ood_ll[ood_ll > ood_ll_percentile], label='OOD')
    plt.legend()
    fig.canvas.draw()
    hist_img = torch.tensor(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''))
    hist_img = torch.tensor(hist_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))).transpose(0, 2).transpose(1, 2)
    writer.add_image("ll_hist", hist_img, epoch)

    # Save checkpoint
    if (epoch % args.save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(args.ckptdir, exist_ok=True)
        torch.save(state, os.path.join(args.ckptdir, str(epoch)+'.pt'))

    if not args.no_sampling:
        # Save samples and data
        os.makedirs(os.path.join(args.ckptdir, 'samples'), exist_ok=True)
        images_concat = draw_samples(net, writer, loss_fn, args.num_samples, device, img_shape, epoch*len(trainloader.dataset))
        os.makedirs(args.ckptdir, exist_ok=True)
        torchvision.utils.save_image(images_concat,
                                     os.path.join(args.ckptdir, 'samples/epoch_{}.png'.format(epoch)))
