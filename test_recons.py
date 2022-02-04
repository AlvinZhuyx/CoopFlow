'''
###################################################################
In this code, we test the reconstruction ability of trained
CoopFlow on cifar dataset. For each image, we use normalizing flow
to do an initialization (find an initialization that lies in the 
possible input area of Langevin flow).  Then we update Langevin flow
input to reduce the reconstruction loss.
###################################################################
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
import pickle
import multiprocessing
from sklearn import metrics
from models import EBM_res as EBM
from models import FlowPlusPlus
from matplotlib import pyplot as plt

def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    num_batches = args.num_data // args.batch_size

    ebm_net = EBM(n_c=3, n_f=128)
    flow_net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                       in_shape=(3, 32, 32),
                       mid_channels=args.num_channels,
                       num_blocks=args.num_blocks,
                       num_dequant_blocks=args.num_dequant_blocks,
                       num_components=args.num_components,
                       use_attn=args.use_attn,
                       drop_prob=args.drop_prob)
    print('Resuming from checkpoint at {}...'.format(args.load_path))
    checkpoint = torch.load(args.load_path)
    ebm_net = ebm_net.cuda()
    flow_net = flow_net.cuda()
    ebm_net = torch.nn.DataParallel(ebm_net, [0])
    flow_net = torch.nn.DataParallel(flow_net, [0])
    ebm_net.load_state_dict(checkpoint['ebm_net'])
    flow_net.load_state_dict(checkpoint['flow_net'])
    ebm_net.eval()
    flow_net.eval()

    print("Begin process {}".format(args.data_type))
    if args.data_type == 'cifar10':
        truedata = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
        dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        counter = 0
        for x, _ in dataloader:
            truedata.append(x.detach().clone())
            counter += 1
            if counter > num_batches:
                break
        truedata = torch.cat(truedata, 0).cuda()
        
    else:
        raise NotImplementedError
    # Do reconstruction on real data
    start_time = time.time()

    x0s = []
    xks = []
    obses = []
    errors = []
    print('Begin reconstruction')
    error_curve = np.zeros((args.num_steps_recons, num_batches))
    for i in range(num_batches):
        start = i * args.batch_size
        end = (i+1) * args.batch_size
        obs = truedata[start : end]
        
        z = torch.randn((args.batch_size, 3, args.image_size, args.image_size),\
        dtype=torch.float32, device='cuda', requires_grad=True)
        with torch.no_grad():
            x0, _ = flow_net(z, reverse=True)
            x0 = 2.0 * torch.sigmoid(x0) - 1.0
        x0 = torch.autograd.Variable(x0.detach().clone(), requires_grad=True) 
            
        for j in range(args.num_steps_recons):
            xk = ebm_sample(ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device='cuda', x_0=x0)
            error = torch.sum((obs - xk) ** 2) 
            grad_x0 = torch.autograd.grad(error, [x0])[0]
            ratio = args.ratio ** j
            x0.data -= args.step_size_recons * ratio * grad_x0
            x0 = torch.clamp(x0, -1.0, 1.0)
            if j % 50 == 0:
                print('{} {:.2f} {:.2f}'.format(j, time.time() - start_time, error.item()/len(obs)))
            error_curve[j][i] = error.item()/len(obs)/(args.image_size ** 2) # per-pixel error
        
        xk = ebm_sample(ebm_net, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device='cuda', x_0=x0)
        error = torch.sum((obs - xk) ** 2, dim=[1,2,3]) / (args.image_size ** 2) # per-pixel error
        error = error.detach()
        x0 = x0.detach()
        xk = xk.detach()
        
        x0s.append(x0.clone().detach().cpu())
        xks.append(xk.clone().detach().cpu())
        obses.append(obs.clone().detach().cpu())
        errors.append(error.clone().detach().cpu())


        if i % 1 == 0:
            print("Batch {} time {:.2f} reconstruction loss {:.3f}".format(i, time.time() - start_time, error.mean().item()))

    x0s = torch.cat(x0s, 0).detach()
    xks = torch.cat(xks, 0).detach()
    obses = torch.cat(obses, 0).detach()
    x_save = torch.cat([torch.unsqueeze(x0s[:27], 1), torch.unsqueeze(xks[:27], 1), torch.unsqueeze(obses[:27], 1)], 1)
    x_save = torch.reshape(x_save, (81, 3, args.image_size, args.image_size))
    os.makedirs(args.save_dir, exist_ok=True)
    torchvision.utils.save_image(torch.clamp(x_save, -1., 1.), os.path.join(args.save_dir, 'recons{}_scratch.png'.format(args.data_type)), normalize=True, nrow=9)
    plt.figure()
    error_curve = np.mean(error_curve, axis=1)
    plt.plot(np.arange(args.num_steps_recons), error_curve)
    plt.savefig(os.path.join(args.save_dir, 'error_curve.png'.format(args.data_type)))

    errors = torch.cat(errors, 0).detach().cpu().numpy()

    save_dict = {'x0s': x0s.cpu().numpy(), 'xks': xks.cpu().numpy(), 'errors': errors, 'obs': obses.cpu().numpy(), 'error_curve': error_curve}
    print("Finish {} total time usage {}".format(args.data_type, time.time() - start_time))
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, '{}.pickle'.format(args.data_type)), 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    print("Mean mse loss {:4f}".format(np.mean(errors)))

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

    parser.add_argument('--batch_size', default=40, type=int, help='Batch size') #25
    parser.add_argument('--image_size', default=32, type=int, help='Image size')
    parser.add_argument('--save_interval', default=10, type=int, help='Interval between saving image')
    parser.add_argument('--num_data', default=1000, type=int, help='Number of data for testing') 
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--load_path', default='./ckpt/cifar10.pth.tar', type=str, help='location of pretrained checkpoint')

    parser.add_argument('--num_steps_Langevin_coopNet', default=30, type=int,
                        help='number of Langevin steps in CoopNets')
    parser.add_argument('--step_size', default=0.03, type=float, help='Langevin step size') # trained with 0.03

    parser.add_argument('--num_steps_recons', default=200, type=int, help='number of steps for doing reconstruction')
    parser.add_argument('--step_size_recons', default=0.05, type=float, help='Langevin step size') 
    parser.add_argument('--ratio', default=1.0, type=float, help='ratio for shrink reconstruction step size') #(ori 100 with 0.97)

    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_dequant_blocks', default=2, type=int, help='Number of blocks in dequantization')
    parser.add_argument('--num_channels', default=96, type=int, help='Number of channels in Flow++')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')

    parser.add_argument('--save_dir', type=str, default='./recons_testset')
    parser.add_argument('--data_type', type=str, default='cifar10')
    main(parser.parse_args())

