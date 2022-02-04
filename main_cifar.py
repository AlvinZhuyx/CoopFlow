"""
Train CoopFlow on CIFAR-10.
"""
import argparse
import numpy as np
import os
import time
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import math
from models import FlowPlusPlus
from models import EBM_res as EBM
from tqdm import tqdm
from matplotlib import pyplot as plt


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since model expects inputs in (0, 1)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('Building model..')
    flow_net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                       in_shape=(3, 32, 32),
                       mid_channels=args.num_channels,
                       num_blocks=args.num_blocks,
                       num_dequant_blocks=args.num_dequant_blocks,
                       #num_dequant_blocks=-1,
                       num_components=args.num_components,
                       use_attn=args.use_attn,
                       drop_prob=args.drop_prob)
    ebm_net = EBM(n_c=3, n_f=128)
    flow_net = flow_net.to(device)
    ebm_net = ebm_net.to(device)
    if device == 'cuda':
        flow_net = torch.nn.DataParallel(flow_net, args.gpu_ids)
        ebm_net = torch.nn.DataParallel(ebm_net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        file_name = "./ckpt/cifar10.pth.tar"
        print('Resuming from checkpoint at {}...'.format(file_name))
        #assert os.path.isdir('save'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(file_name)
        flow_net.load_state_dict(checkpoint['flow_net'])
        ebm_net.load_state_dict(checkpoint['ebm_net'])
        #global best_loss
        global gstep
        #best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch'] + 1
        gstep = start_epoch * len(trainset)

    if not args.train:
        assert args.resume
        test(ebm_net, flow_net, testloader, device, args)
        return

    loss_fn = util.NLLLoss().to(device)
    flow_param_groups = util.get_param_groups(flow_net, args.weight_decay, norm_suffix='weight_g')
    flow_optimizer = optim.Adam(flow_param_groups, lr=args.lr_flow)
    ebm_optimizer = optim.Adam(ebm_net.parameters(), lr=args.lr_ebm, betas=[.5, .5])
    warm_up = args.warm_up * args.batch_size
    flow_scheduler = sched.LambdaLR(flow_optimizer, lambda s: min(1., s / warm_up))
    ebm_scheduler = sched.LambdaLR(ebm_optimizer, lambda s: min(1., s/warm_up))


    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, ebm_net, flow_net, trainloader, device, ebm_optimizer, ebm_scheduler, flow_optimizer, flow_scheduler,
              loss_fn, args.max_grad_norm, args, save_checkpoint=(epoch % 5 == 0))


@torch.enable_grad()
def train(epoch, ebm_net, flow_net, trainloader, device, ebm_optimizer, ebm_scheduler, flow_optimizer, flow_scheduler, loss_fn, max_grad_norm, args, save_checkpoint=True):
    global gstep
    print('\nEpoch: %d' % epoch)
    ebm_net.train()
    flow_net.train()
    flow_loss_meter = util.AverageMeter()
    ebm_loss_meter = util.AverageMeter()
    counter = 0
    start_time = time.time()
    num_iter = math.ceil(float(len(trainloader.dataset)) / args.batch_size)
    # with tqdm(total=len(trainloader.dataset)) as progress_bar:

    for x, _ in trainloader:
        # train flow model
        x = x.to(device)
        if epoch == 0 and counter < 200:
            flow_net(x.detach(), reverse=False)
            if counter % 20 == 0:
                print(counter, time.time() - start_time)
            counter += 1
            continue

        x_flow = sample_flow(flow_net, args.batch_size, device).detach()

        save_img = (counter % 200 == 0)
        x_ebm = ebm_sample(epoch, ebm_net, m=64, n_ch=3, im_w=32, im_h=32, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device=device, p_0=x_flow\
                           , num_sample=args.batch_size, save_images=save_img, save_dir=args.save_dir)
        mse_loss = torch.sum(torch.mean((x_ebm.detach() - x_flow) ** 2, dim=0))
        flow_optimizer.zero_grad()
        z, sldj = flow_net(x_ebm.detach(), reverse=False)
        flow_loss = loss_fn(z, sldj)
        flow_loss_meter.update(flow_loss.item(), x_ebm.size(0))
        flow_loss.backward()
        if max_grad_norm > 0:
            util.clip_grad_norm(flow_optimizer, max_grad_norm)
        flow_optimizer.step()
        flow_scheduler.step(gstep)
        # train ebm model
        ebm_optimizer.zero_grad()
        en_pos = ebm_net(x).mean()
        en_neg = ebm_net(x_ebm).mean()
        ebm_loss = en_neg - en_pos
        ebm_loss_meter.update(ebm_loss.item(), x_ebm.size(0))
        ebm_loss.backward()
        ebm_optimizer.step()
        ebm_scheduler.step(gstep)

        if torch.isnan(mse_loss) or torch.isnan(flow_loss) or torch.isnan(en_pos) or torch.isnan(en_neg) or torch.isnan(ebm_loss):
            x_array = x_flow.clone().detach().view(-1).cpu().numpy()
            os.makedirs(args.save_dir, exist_ok=True)
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_flow_before_blow.png'.format(args.save_dir))
            plt.close()
            x_ebm_array = x_ebm.clone().detach().view(-1).cpu().numpy()
            plt.figure()
            plt.hist(np.clip(x_ebm_array, -2, 2), bins=128)
            plt.savefig('{}/hist_ebm_before_blow.png'.format(args.save_dir))
            plt.close()
            assert False

        if counter % 200 == 0:
            x_array = x_flow.clone().detach().view(-1).cpu().numpy()
            os.makedirs(args.save_dir, exist_ok=True)
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_flow.png'.format(args.save_dir))
            plt.close()
            x_ebm_array = x_ebm.clone().detach().view(-1).cpu().numpy()
            plt.figure()
            plt.hist(np.clip(x_ebm_array, -2, 2), bins=128)
            plt.savefig('{}/hist_ebm.png'.format(args.save_dir))
            plt.close()
            x_array = x.clone().detach().view(-1).cpu().numpy()
            plt.figure()
            plt.hist(np.clip(x_array, -2, 2), bins=128)
            plt.savefig('{}/hist_ori.png'.format(args.save_dir))
            plt.close()
            torchvision.utils.save_image(torch.clamp(x_flow, -1., 1.), '{}/flow_epoch_{}.png'.format(args.save_dir, epoch),
                                         normalize=True, nrow=int(args.batch_size ** 0.5))
            torchvision.utils.save_image(torch.clamp(x, -1., 1.), '{}/ori_{}.png'.format(args.save_dir, epoch),
                                         normalize=True, nrow=int(args.batch_size ** 0.5))
        if counter % 20 == 0:
            print('Epoch {} iter {}/{} time{:.3f} FLOW: mse loss {:.3f} flow_loss {:.3f} EBM: pos en {:.3f} neg en{:.3f} en diff {:.3f}' \
                .format(epoch, counter, num_iter, time.time() - start_time, mse_loss, flow_loss, en_pos, en_neg, ebm_loss))

        if save_checkpoint and counter == num_iter - 1:
            print('Saving...')
            state = {
                'ebm_net': ebm_net.state_dict(),
                'flow_net': flow_net.state_dict(),
                'epoch': epoch,
            }
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(state, '{}/{}_{}.pth.tar'.format(args.ckpt_dir, epoch, counter))

        '''
        progress_bar.set_postfix(nll=loss_meter.avg,
                                 bpd=util.bits_per_dim(x, loss_meter.avg),
                                 lr=optimizer.param_groups[0]['lr'])
        progress_bar.update(x.size(0))
        '''
        gstep += x.size(0)
        counter += 1


def ebm_sample(epoch, net, m=64, n_ch=3, im_w=32, im_h=32, K=10, step_size=0.02, device='cpu', p_0=None, num_sample=100, save_images=True, save_dir='./'):

    if p_0 is None:
        sample_p_0 = lambda: torch.FloatTensor(m, n_ch, im_w, im_h).uniform_(-1, 1).to(device)
    else:
        sample_p_0 = lambda:  p_0.clone().to(device)

    x_k = torch.autograd.Variable(sample_p_0(), requires_grad=True)
    #print(K, step_size)
    for k in range(K):
        net_prime = torch.autograd.grad(net(x_k).sum(), [x_k])[0]
        #x_k.data += net_prime
        #x_k.data = x_k.data - 0.5 * step_size * step_size * (x_k.data/ (0.05 ** 2) - net_prime)
        delta = -0.5 * step_size * step_size * (net_prime) # + step_size * (1.0 - 1.0/(K - k)) * torch.randn_like(x_k).data
        #delta = torch.clamp(delta, -0.05, 0.05)
        delta = torch.clamp(delta, -0.1, 0.1)
        x_k.data -= delta
        #x_k = torch.clamp(x_k, -0.999, 0.999)
        x_k = torch.clamp(x_k, -1.0, 1.0)
    if save_images:
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(torch.clamp(x_k, -1., 1.), '{}/ebm_epoch_{}.png'.format(save_dir, epoch),
                                     normalize=True, nrow=int(num_sample ** 0.5))

    return x_k.detach()


@torch.no_grad()
def sample_flow(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    #print(x.min(), x.max(), torch.abs(x).mean())

    x = 2.0 * torch.sigmoid(x) - 1.0

    return x

#@torch.no_grad()
def test(ebm_net, flow_net, testloader, device, args):
    flow_net.eval()
    ebm_net.eval()
    ori_dir = './exp_cifar/ori_samples'
    gen_dir = './exp_cifar/gen_samples'
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    total_imgs = 50000
    counter = 0
    grid_img = []
    grid_img_flow = []
    start_time = time.time()
    for x, _ in testloader:
        batch_size = len(x)
        x_flow = sample_flow(flow_net,batch_size, device)
        x_ebm = ebm_sample(0, ebm_net, m=64, n_ch=3, im_w=32, im_h=32, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device=device, p_0=x_flow, num_sample=batch_size, save_images=False)
        # print(x_ebm.max(), x_ebm.min(), x.max(), x.min())
        for i in range(batch_size):
            torchvision.utils.save_image((x[i] + 1.0) * 0.5, os.path.join(ori_dir, '{}.png'.format(counter)))
            torchvision.utils.save_image((x_ebm[i] + 1.0) * 0.5, os.path.join(gen_dir, '{}.png'.format(counter)))
            if len(grid_img) < 100:
                grid_img.append((x_ebm[i] + 1.0) * 0.5)
                grid_img_flow.append((x_flow[i] + 1.0) * 0.5)
            counter += 1
            if counter % 1000 == 0:
                print(counter, time.time() - start_time)
            if counter >= total_imgs:
                images_concat = torchvision.utils.make_grid(grid_img, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                torchvision.utils.save_image(images_concat, './exp_cifar/ebm.png')
                images_concat = torchvision.utils.make_grid(grid_img_flow, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                torchvision.utils.save_image(images_concat, './exp_cifar/flow.png')
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoopFlow on cifar10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU')
    parser.add_argument('--image_size', default=32, type=int, help='Image size')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr_flow', default=1e-4, type=float, help='Peak learning rate')
    parser.add_argument('--lr_ebm', default=1e-4, type=float, help='Peak learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_dequant_blocks', default=2, type=int, help='Number of blocks in dequantization')
    parser.add_argument('--num_channels', default=96, type=int, help='Number of channels in Flow++')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='exp_cifar/Coop_samples', help='Directory for saving samples')
    parser.add_argument('--ckpt_dir', type=str, default='exp_cifar/save', help='Directory for saving samples')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--num_steps_Langevin_coopNet', default=30, type=int,
                        help='number of Langevin steps in CoopNets')
    parser.add_argument('--step_size', default=0.04, type=float, help='Langevin step size')
    parser.add_argument('--train', default=False, type=bool, help='whether to train the model or to generate image for computing fid')

    best_loss = 0
    gstep = 0

    main(parser.parse_args())
