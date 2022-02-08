"""
CoopFlow(pre) on SVHN.
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
import pytorch_fid_wrapper as pfw
import util
import math
import multiprocessing
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

    # SVHN
    trainset = torchvision.datasets.SVHN('./data/svhn', download=True,
                                             transform=transforms.Compose([
                                             transforms.Resize(args.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    testset = torchvision.datasets.SVHN('./data/svhn', download=True, 
                                             transform=transforms.Compose([
                                             transforms.Resize(args.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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
    # load in the pretrained flow model
    if args.train:
        file_name = "./flow_ckpt/svhn.pth.tar"
        checkpoint = torch.load(file_name)
        flow_net.load_state_dict(checkpoint['net'])
        print("Load pretrained flow model from {}, epoch {}".format(file_name, checkpoint['epoch']))
    best_fid = np.inf
    if args.resume:
        # Load checkpoint.
        file_name = "./ckpt/svhn_pretrain.pth.tar"
        print('Resuming from checkpoint at {}...'.format(file_name))
        checkpoint = torch.load(file_name)
        flow_net.load_state_dict(checkpoint['flow_net'])
        ebm_net.load_state_dict(checkpoint['ebm_net'])
        global gstep
        start_epoch = checkpoint['epoch'] + 1
        gstep = start_epoch * len(trainset)
        best_fid = checkpoint['fid']
        print('load epoch {} best fid {:.4f}'.format(checkpoint['epoch'], best_fid))

    if not args.train:
        assert args.resume
        test(ebm_net, flow_net, trainloader, device, args)
        return

    loss_fn = util.NLLLoss().to(device)
    flow_param_groups = util.get_param_groups(flow_net, args.weight_decay, norm_suffix='weight_g')
    flow_optimizer = optim.Adam(flow_param_groups, lr=args.lr_flow)
    ebm_optimizer = optim.Adam(ebm_net.parameters(), lr=args.lr_ebm, betas=[.5, .5])
    if args.resume:
        flow_optimizer.load_state_dict(checkpoint['flow_optimizer'])
        ebm_optimizer.load_state_dict(checkpoint['ebm_optimizer'])

    warm_up = args.warm_up * args.batch_size
    flow_scheduler = sched.LambdaLR(flow_optimizer, lambda s: min(1., s / warm_up))
    ebm_scheduler = sched.LambdaLR(ebm_optimizer, lambda s: min(1., s/warm_up))

    

    fids = []
    gnorms_flow = []
    gnorms_ebm = []

    #estimate the real mean and real standard deviation of the data (only calculate once)
    start_time = time.time()
    print("Begin calculating real image statistics")
    real_x = []
    length = 0
    for x, _ in testloader:
        if length < 50000:
            real_x.append(x.detach().clone())
            length += len(x)
    real_x = torch.cat(real_x, dim=0)
    real_x = (real_x + 1.0) / 2.0 # normalize to (0, 1)
    print(torch.min(real_x), torch.max(real_x))
    real_m, real_s = pfw.get_stats(real_x, device="cuda:0")
    print("Finish calculating real image statistics {:.3f}".format(time.time() - start_time))
    print('Begin training')
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        gnorms_flow, gnorms_ebm = train(epoch, ebm_net, flow_net, trainloader, device, ebm_optimizer, ebm_scheduler, flow_optimizer, flow_scheduler,
              loss_fn, args.max_grad_norm, args, save_checkpoint=(epoch % 5 == 0), gnorms_flow=gnorms_flow, gnorms_ebm=gnorms_ebm)
        if epoch % 1 == 0:
            fid = test_in_code(ebm_net, flow_net, args, real_m, real_s)
            fids.append(fid)
            if fid < best_fid:
                best_fid = fid
                print('Saving best state_dict'.format(epoch))
                state = {
                    'ebm_net': ebm_net.state_dict(),
                    'flow_net': flow_net.state_dict(),
                    'ebm_optimizer': ebm_optimizer.state_dict(),
                    'flow_optimizer': flow_optimizer.state_dict(),
                    'epoch': epoch,
                    'fid': fid
                }
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(state, '{}/best.pth.tar'.format(args.ckpt_dir))
            print("Epoch {} fid {:.3f} best_fid {:.3f}".format(epoch, fid, best_fid))
            plt.figure()
            plt.plot(np.arange(len(fids)), np.array(fids))
            os.makedirs(args.save_dir, exist_ok=True)
            plt.savefig('{}/fids.png'.format(args.save_dir))
            plt.close()



@torch.enable_grad()
def train(epoch, ebm_net, flow_net, trainloader, device, ebm_optimizer, ebm_scheduler, flow_optimizer, flow_scheduler, loss_fn, max_grad_norm, args, save_checkpoint=True, gnorms_flow=None, gnorms_ebm=None):
    global gstep
    print('\nEpoch: %d' % epoch)
    ebm_net.train()
    flow_net.train()
    flow_loss_meter = util.AverageMeter()
    ebm_loss_meter = util.AverageMeter()
    counter = 0
    start_time = time.time()
    num_iter = math.ceil(float(len(trainloader.dataset)) / args.batch_size)
    x_list = []
    for x, _ in trainloader:
        # train flow model
        x = x.to(device)
        x_flow = sample_flow(flow_net, args.batch_size, device).detach()

        x_ebm = ebm_sample(epoch, ebm_net, m=64, n_ch=3, im_w=32, im_h=32, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device=device, p_0=x_flow\
                           , num_sample=args.batch_size, save_images=False, save_dir=args.save_dir)
        mse_loss = torch.sum(torch.mean((x_ebm.detach() - x_flow) ** 2, dim=0)).detach() # just monitor the image change, not influence optimization
        
        flow_optimizer.zero_grad()
        if (counter % args.n_update_flow == 0) and epoch > 10:  # only update ebm at the first 10 epochs
            z, sldj = flow_net(x_ebm.detach(), reverse=False)
            flow_loss = loss_fn(z, sldj)
            flow_loss_meter.update(flow_loss.item(), x_ebm.size(0))
            flow_loss.backward()
            flow_grad_norm = torch.nn.utils.clip_grad_norm_(flow_net.parameters(), args.grad_norm_flow).item()
        else:
            with torch.no_grad():
                z, sldj = flow_net(x_ebm.detach(), reverse=False)
                flow_loss = loss_fn(z, sldj)
                flow_grad_norm = np.inf

        # train ebm model
        ebm_optimizer.zero_grad()
        en_pos = ebm_net(x.detach()).mean()
        en_neg = ebm_net(x_ebm.detach()).mean()
        ebm_loss = en_neg - en_pos
        ebm_loss_meter.update(ebm_loss.item(), x_ebm.size(0))
        ebm_loss.backward()
        ebm_grad_norm = torch.nn.utils.clip_grad_norm_(ebm_net.parameters(), args.grad_norm_ebm).item()
        if ebm_grad_norm < args.skip_threshold_ebm or epoch == 0:
            ebm_optimizer.step()
            ebm_scheduler.step(gstep)
            if epoch > 10 and counter % args.n_update_flow == 0 and flow_grad_norm < args.skip_threshold_flow: 
                flow_optimizer.step()
                flow_scheduler.step(gstep)
        
        gnorms_ebm.append(ebm_grad_norm)
        if counter % args.n_update_flow == 0:
            gnorms_flow.append(flow_grad_norm) 
        

        if torch.isnan(mse_loss) or torch.isnan(flow_loss) or torch.isnan(en_pos) or torch.isnan(en_neg) or torch.isnan(ebm_loss):
            if not os.path.exists('{}/hist_flow_before_blow.png'.format(args.save_dir)):
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
            print(mse_loss, flow_loss, en_pos, en_neg, ebm_loss)
            #assert False

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

            plt.figure()
            plt.plot(np.arange(len(gnorms_flow))*args.n_update_flow, gnorms_flow)
            plt.savefig('{}/grad_norm_flow.png'.format(args.save_dir))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(gnorms_ebm)), gnorms_ebm)
            plt.savefig('{}/grad_norm_ebm.png'.format(args.save_dir))
            plt.close()
            #print(gnorms_flow, gnorms_ebm)
            np.save('{}/grad_norm_flow.npy'.format(args.save_dir), np.array(gnorms_flow))
            np.save('{}/grad_norm_ebm.npy'.format(args.save_dir), np.array(gnorms_ebm))
            
            torchvision.utils.save_image(torch.clamp(x_flow, -1., 1.), '{}/flow_epoch_{}.png'.format(args.save_dir, epoch),
                                         normalize=True, nrow=int(args.batch_size ** 0.5))
            torchvision.utils.save_image(torch.clamp(x, -1., 1.), '{}/ori_{}.png'.format(args.save_dir, epoch),
                                         normalize=True, nrow=int(args.batch_size ** 0.5))
            torchvision.utils.save_image(torch.clamp(x_ebm, -1., 1.), '{}/ebm_epoch_{}.png'.format(args.save_dir, epoch),
                                         normalize=True, nrow=int(args.batch_size ** 0.5))
            
        if counter % 20 == 0:
            print('Epoch {} iter {}/{} time{:.3f} FLOW: image mse change {:.3f} flow_loss {:.3f} EBM: pos en {:.3f} neg en{:.3f} en diff {:.3f}' \
                .format(epoch, counter, num_iter, time.time() - start_time, mse_loss, flow_loss, en_pos, en_neg, ebm_loss))

        if save_checkpoint and counter == num_iter - 1:
            print('Saving...')
            state = {
                'ebm_net': ebm_net.state_dict(),
                'flow_net': flow_net.state_dict(),
                'ebm_optimizer': ebm_optimizer.state_dict(),
                'flow_optimizer': flow_optimizer.state_dict(),
                'epoch': epoch,
            }
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(state, '{}/{}_{}.pth.tar'.format(args.ckpt_dir, epoch, counter))

        gstep += x.size(0)
        counter += 1
    return gnorms_flow, gnorms_ebm


def ebm_sample(epoch, net, m=64, n_ch=3, im_w=32, im_h=32, K=10, step_size=0.02, device='cpu', p_0=None, num_sample=100, save_images=True, save_dir='./'):

    if p_0 is None:
        sample_p_0 = lambda: torch.FloatTensor(m, n_ch, im_w, im_h).uniform_(-1, 1).to(device)
        print("Should start from flow's output")
        assert False
    else:
        sample_p_0 = lambda:  p_0.clone().to(device)

    x_k = torch.autograd.Variable(sample_p_0(), requires_grad=True)
    for k in range(K):
        net_prime = torch.autograd.grad(net(x_k).sum(), [x_k])[0]
        delta = -0.5 * step_size * step_size * net_prime
        delta = torch.clamp(delta, -0.1, 0.1)
        x_k.data -= delta
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

    x = 2.0 * torch.sigmoid(x) - 1.0

    return x

def test_in_code(ebm_net, flow_net, args, real_m, real_s):
    flow_net.eval()
    ebm_net.eval()
    total_imgs = 50000
    gen_img = []
    start_time = time.time()
    print("Begin testing")
    for i in range(200):
        batch_size = 250
        with torch.no_grad():
            x_flow = sample_flow(flow_net, batch_size, device="cuda:0")
        x_ebm = ebm_sample(0, ebm_net, m=64, n_ch=3, im_w=32, im_h=32, K=args.num_steps_Langevin_coopNet, step_size=0.035, device="cuda:0", p_0=x_flow, num_sample=batch_size, save_images=False)
        x_ebm = (x_ebm + 1.0) * 0.5
        gen_img.append(x_ebm.detach().clone())
        if i % 20 == 0 or i == 199:
            print(i * batch_size, time.time() - start_time)
    gen_img = torch.cat(gen_img, dim=0) 
    fid = pfw.fid(gen_img, real_m = real_m, real_s=real_s, device="cuda:0")
    flow_net.train()
    ebm_net.train()
    return fid


#@torch.no_grad()
def test(ebm_net, flow_net, testloader, device, args):
    flow_net.eval()
    ebm_net.eval()
    ori_dir = './exp_svhn_pre/ori_samples'
    gen_dir = './exp_svhn_pre/gen_samples'
    flow_dir = './exp_svhn_pre/flow_samples'
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    total_imgs = 50000
    counter = 0
    grid_img = []
    grid_img_flow = []
    grid_img_ori = []
    start_time = time.time()
    for x, _ in testloader:
        batch_size = len(x)
        with torch.no_grad():
            x_flow = sample_flow(flow_net,batch_size, device)
        x_ebm = ebm_sample(0, ebm_net, m=64, n_ch=3, im_w=32, im_h=32, K=args.num_steps_Langevin_coopNet, step_size=args.step_size, device=device, p_0=x_flow, num_sample=batch_size, save_images=False)
        for i in range(batch_size):
            torchvision.utils.save_image((x[i] + 1.0) * 0.5, os.path.join(ori_dir, '{}.png'.format(counter)))
            torchvision.utils.save_image((x_ebm[i] + 1.0) * 0.5, os.path.join(gen_dir, '{}.png'.format(counter)))
            torchvision.utils.save_image((x_flow[i] + 1.0) * 0.5, os.path.join(flow_dir, '{}.png'.format(counter)))
            if len(grid_img) < 100:
                grid_img.append((x_ebm[i] + 1.0) * 0.5)
                grid_img_flow.append((x_flow[i] + 1.0) * 0.5)
                grid_img_ori.append((x[i] + 1.0) * 0.5)
            counter += 1
            if counter % 1000 == 0:
                print(counter, time.time() - start_time)
            if counter >= total_imgs:
                images_concat = torchvision.utils.make_grid(grid_img, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                torchvision.utils.save_image(images_concat, 'exp_svhn_pre/ebm.png')
                images_concat = torchvision.utils.make_grid(grid_img_flow, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                torchvision.utils.save_image(images_concat, 'exp_svhn_pre/flow.png')
                images_concat = torchvision.utils.make_grid(grid_img_ori, nrow = int(len(grid_img) ** 0.5), padding=2, pad_value=255, normalize=True)
                torchvision.utils.save_image(images_concat, 'exp_svhn_pre/ori.png')
                return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow++ on CelebA')

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
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='exp_svhn_pre/Coop_samples', help='Directory for saving samples')
    parser.add_argument('--ckpt_dir', type=str, default='exp_svhn_pre/save', help='Directory for saving samples')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--num_steps_Langevin_coopNet', default=30, type=int,
                        help='number of Langevin steps in CoopNets')
    parser.add_argument('--step_size', default=0.035, type=float, help='Langevin step size') # trained with 0.03
    parser.add_argument('--train', default=False, type=bool, help='whether to train the model or to generate image for computing fid')
    parser.add_argument('--n_fid_sample', default=20000, type=int, help='number of samples to estimate fid during training')

    parser.add_argument('--grad_norm_flow', default=1.0, type=float, help='clip gradient norm')
    parser.add_argument('--skip_threshold_flow', default=1e6, type=float, help='skip gradient update')
    parser.add_argument('--grad_norm_ebm', default=1.0, type=float, help='clip gradient norm')
    parser.add_argument('--skip_threshold_ebm', default=5e6, type=float, help='skip gradient update')

    parser.add_argument('--n_update_flow', default=1, type=int, help='number of ebm updates between flow model update')
    
    best_loss = 0
    gstep = 0

    main(parser.parse_args())
