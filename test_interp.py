'''
##########################################################
Do interpolation between two given images or do interpolation
between two random z.
##########################################################
'''
import argparse
import numpy as np
import os
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import math
import multiprocessing
from sklearn import metrics
from models import EBM_res as EBM
from models import FlowPlusPlus
from matplotlib import pyplot as plt

def main(args):
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed_all(12)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transform_test = transforms.Compose([transforms.Resize(args.image_size),
                                        transforms.CenterCrop(args.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = torchvision.datasets.ImageFolder(root='./data/celeba', transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ebm_net = EBM(n_c=3, n_f=128)
    flow_net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                       in_shape=(3, 32, 32),
                       mid_channels=args.num_channels,
                       num_blocks=args.num_blocks,
                       num_dequant_blocks=args.num_dequant_blocks,
                       num_components=args.num_components,
                       use_attn=args.use_attn,
                       drop_prob=args.drop_prob)
    ebm_net = ebm_net.to(device)
    flow_net = flow_net.to(device)
    ebm_net = torch.nn.DataParallel(ebm_net, [0])
    flow_net = torch.nn.DataParallel(flow_net, [0])
    print('Resuming from checkpoint at {}...'.format(args.load_path))
    checkpoint = torch.load(args.load_path)
    ebm_net.load_state_dict(checkpoint['ebm_net'])
    flow_net.load_state_dict(checkpoint['flow_net'])
    ebm_net.eval()
    flow_net.eval()
    counter = 0

    for imgs, labels in testloader:
        if counter > 5:
            break
        counter += 1

    imgs = imgs.cuda()
    
    start_time = time.time()
    z = torch.randn((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32, device='cuda', requires_grad=True)
    with torch.no_grad():
        x0, _ = flow_net(z, reverse=True)
        x0 = 2.0 * torch.sigmoid(x0) - 1.0
    x0 = torch.autograd.Variable(x0.detach().clone(), requires_grad=True) 
    
    # first do image reconstruction
    for j in range(args.num_steps_recons):
        xk = ebm_sample(ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device='cuda', x_0=x0)
        error = torch.sum((imgs - xk) ** 2)
        grad_x0 = torch.autograd.grad(error, [x0])[0]
        ratio = args.ratio ** j
        x0.data -= args.step_size_recons * ratio * grad_x0
        x0 = torch.clamp(x0, -1.0, 1.0)
        print("Step {} time {:.3f} error {:.3f}".format(j, time.time() - start_time, error))

  
    xk = ebm_sample(ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device='cuda', x_0=x0)   
    x0 = x0.detach()
    xk = xk.detach()
    with torch.no_grad():
        z, _ = flow_net(x0, reverse=False)
        z = z.detach()

    # interpolate between z's
    z0 = z[: args.batch_size//2]
    z1 = z[args.batch_size//2 :]
    diff = z1 - z0
    save_list = []
    for i in range(21):
        z_tmp = z0 + diff * float(i) / 20.0
        with torch.no_grad():
            x0, _ = flow_net(z_tmp, reverse=True)
            x0 = 2.0 * torch.sigmoid(x0) - 1.0
        x0 = torch.autograd.Variable(x0.detach().clone(), requires_grad=True)
        xk = ebm_sample(ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device='cuda', x_0=x0)
        save_list.append(torch.unsqueeze(xk.detach(), 1))    
    save_list = torch.cat(save_list, 1)
    num_step = save_list.shape[1]
    save_list = torch.reshape(save_list, (-1, 3, args.image_size, args.image_size))
    img_name = 'interpolate.png'
    torchvision.utils.save_image(torch.clamp(save_list, -1., 1.), img_name, normalize=True, nrow=num_step) 


    #interpolate between random z's
    z_random = torch.randn_like(z)
    z0 = z_random[: args.batch_size//2]
    z1 = z_random[args.batch_size//2 :]
    diff = z1 - z0
    save_list = []
    for i in range(21):
        z_tmp = z0 + diff * float(i) / 20.0
        with torch.no_grad():
            x0, _ = flow_net(z_tmp, reverse=True)
            x0 = 2.0 * torch.sigmoid(x0) - 1.0
        x0 = torch.autograd.Variable(x0.detach().clone(), requires_grad=True)
        xk = ebm_sample(ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device='cuda', x_0=x0)
        save_list.append(torch.unsqueeze(xk.detach(), 1))    
    save_list = torch.cat(save_list, 1)
    num_step = save_list.shape[1]
    save_list = torch.reshape(save_list, (-1, 3, args.image_size, args.image_size))
    img_name = 'interpolate_random.png'
    torchvision.utils.save_image(torch.clamp(save_list, -1., 1.), img_name, normalize=True, nrow=num_step)


def ebm_sample(net, K=10, step_size=0.02, device='cpu', x_0=None):
    x_k = x_0
    for k in range(K):
        net_prime = torch.autograd.grad(net(x_k).sum(), [x_k], retain_graph=True, create_graph=True)[0]
        delta = -0.5 * step_size * step_size * net_prime
        x_k = x_k - delta
    return x_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test OOD')
    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=20, type=int, help='Number of images to show')
    parser.add_argument('--image_size', default=32, type=int, help='Image size')
    parser.add_argument('--mask_size', default=16, type=int, help='The size of masked area')
    parser.add_argument('--num_steps_Langevin_coopNet', default=30, type=int,
                        help='number of Langevin steps in CoopNets')
    parser.add_argument('--step_size', default=0.03, type=float, help='Langevin step size')

    parser.add_argument('--num_steps_recons', default=201, type=int, help='number of steps for doing reconstruction')
    parser.add_argument('--step_size_recons', default=0.03, type=float, help='Langevin step size') 
    parser.add_argument('--ratio', default=0.99, type=float, help='ratio for shrink reconstruction step size')

    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--load_path', default='./celeba.pth.tar', type=str, help='location of pretrained checkpoint')
    
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_dequant_blocks', default=2, type=int, help='Number of blocks in dequantization')
    parser.add_argument('--num_channels', default=96, type=int, help='Number of channels in Flow++')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    
    main(parser.parse_args())

