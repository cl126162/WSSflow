import os

import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from flowNetsRAFT import RAFT
from flowNetsRAFT_GMA import RAFT_GMA
from hdf5_dataset import testing_dataset
from transform import target_transform, source_transform
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
from tqdm import tqdm
import random
import h5py
import matplotlib.pyplot as plt
from einops import rearrange


def set_random_seeds(random_seed=0):
    
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class UnsuperviseFlowPLModule(torch.nn.Module):
    def __init__(self, args):
        super(UnsuperviseFlowPLModule, self).__init__()
        self.args = args
        if self.args.arch == 'RAFT_GMA':
            self.model = RAFT_GMA(self.args)
        else:
            self.model = RAFT(self.args)

    def forward(self, batch, batch_idx, mode, epoch):
        self.args.currentEpoch = epoch
        if mode == 'training':
            return self.training_step(batch, batch_idx)
        elif mode == 'validation':
            return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        local_dict = batch[0]
        images = local_dict['target']
        flows = local_dict['flow']

        B, C, H, W = images.size()
        if self.args.use_AlternateCorr:
            pred_flows = self.model(images, flows, args=self.args, flow_init=None)


            if self.args.arch == 'RAFT' or self.args.arch == 'RAFT_GMA':
                all_flow_iters = pred_flows[0]
                predicted_flows = all_flow_iters[-1]
        else:            
            if self.args.offset == H and self.args.offset == W:
                pred_flows = self.model(images, flows, args=self.args, flow_init=None)

                if self.args.arch == 'RAFT' or self.args.arch == 'RAFT_GMA':
                    all_flow_iters = pred_flows[0]
                    predicted_flows = all_flow_iters[-1]

            else:
                # compute number of patches for folding operation
                B, C, H, W = images.size()
                NUM_Yvectors, NUM_Xvectors = int(np.round((H) / self.args.shift - (self.args.offset / self.args.shift - 1))), \
                                            int(np.round((W) / self.args.shift - (self.args.offset / self.args.shift - 1)))
                reconH, reconW = NUM_Yvectors * self.args.shift + (self.args.offset - self.args.shift) - 2 * self.args.cropSize, \
                                NUM_Xvectors * self.args.shift + (self.args.offset - self.args.shift) - 2 * self.args.cropSize
                croppedOffset = self.args.offset - 2 * self.args.cropSize

                # allocate memory of predicted images
                predicted_flows = torch.zeros_like(images)
                folding_mask = torch.ones_like(images)

                # create patches of image and flow
                patches = images.unfold(3, self.args.offset, self.args.shift).unfold(2, self.args.offset, self.args.shift).permute(0, 2, 3, 1, 5, 4)
                patches = patches.reshape((-1, 2, self.args.offset, self.args.offset))
                flow_patches = flows.unfold(3, self.args.offset, self.args.shift).unfold(2, self.args.offset, self.args.shift).permute(0, 2, 3, 1, 5, 4)
                flow_patches = flow_patches.reshape((-1, 2, self.args.offset, self.args.offset))
                splitted_patches = torch.split(patches, self.args.splitSizeTest, dim=0)
                splitted_flow_patches = torch.split(flow_patches, self.args.splitSizeTest, dim=0)

                predicted_flow_patches = predicted_flows.unfold(3, self.args.offset, self.args.shift)\
                                                        .unfold(2, self.args.offset, self.args.shift)\
                                                        .permute(0, 2, 3, 1, 5, 4)
                predicted_flow_patches = predicted_flow_patches.reshape((-1, 2, self.args.offset, self.args.offset))

                splitted_flow_output_patches = []

                for split in range(len(splitted_patches)):
                    pred_flows = self.model(splitted_patches[split], splitted_flow_patches[split], args=self.args, flow_init=None)

                    if self.args.arch == 'RAFT' or self.args.arch == 'RAFT_GMA':
                        all_flow_iters = pred_flows[0]
                        splitted_flow_output_patches.append(all_flow_iters[-1])

                flow_output_patches = torch.cat(splitted_flow_output_patches, dim=0)\
                                        .view(NUM_Yvectors, NUM_Xvectors, 2, self.args.offset, self.args.offset)

                #reconstruction via folding
                flow_output_patches = flow_output_patches[:, :, :, self.args.cropSize:self.args.offset - self.args.cropSize,
                                                        self.args.cropSize:self.args.offset - self.args.cropSize]\
                                        .reshape((B, NUM_Yvectors, NUM_Xvectors, 2, croppedOffset, croppedOffset)).permute(0, 3, 1, 2, 4, 5)
                flow_output_patches = flow_output_patches.contiguous().view(B, C, -1, croppedOffset * croppedOffset)
                flow_output_patches = flow_output_patches.permute(0, 1, 3, 2)
                flow_output_patches = flow_output_patches.contiguous().view(B, C * croppedOffset * croppedOffset, -1)
                predicted_flows_iter = F.fold(flow_output_patches, output_size=(reconH, reconW),
                                            kernel_size=croppedOffset, stride=self.args.shift)

                mask_patches = folding_mask.unfold(3, self.args.offset, self.args.shift).unfold(2, self.args.offset, self.args.shift)
                mask_patches = mask_patches[:, :, :, :, self.args.cropSize:self.args.offset - self.args.cropSize, self.args.cropSize:self.args.offset - self.args.cropSize].contiguous() \
                    .view(B, C, -1, croppedOffset * croppedOffset)
                mask_patches = mask_patches.permute(0, 1, 3, 2)
                mask_patches = mask_patches.contiguous().view(B, C * croppedOffset * croppedOffset, -1)
                folding_mask = F.fold(mask_patches, output_size=(reconH, reconW), kernel_size=croppedOffset,
                                    stride=self.args.shift)

                predicted_flows[:,:,self.args.cropSize:H-self.args.cropSize,self.args.cropSize:W-self.args.cropSize] = predicted_flows_iter / folding_mask

                predicted_flows = predicted_flows[:, :, self.args.cropSize:H-self.args.cropSize,
                                self.args.cropSize:W-self.args.cropSize]
                flows = flows[:, :, self.args.cropSize:H-self.args.cropSize, self.args.cropSize:W-self.args.cropSize]

        test_epe_loss_final = torch.sum((predicted_flows[:, :, :, :] - flows[:, :, :, :]) ** 2,
                                        dim=1).sqrt().view(-1).mean()

        if self.args.return_values:
            print(predicted_flows[0][-1].shape)
            return_dict = {
                'predicted_flows': predicted_flows.detach(),
                'test_epe_loss_final': test_epe_loss_final.detach(),
            }
        else:
            return_dict = {
                'test_epe_loss_final': test_epe_loss_final.detach(),
            }

        return return_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--name', type=str, default='URAFT_Test')
    parser.add_argument('--log_dir', type=str, default='./tensorboard_logs')
    parser.add_argument('--input_path_ckpt', type=str,
                        default='./pretrained_ckpts/pretrained_RAFT-PIV_GMA_model.ckpt',
                        help='path of already trained checkpoints')
    parser.add_argument('--recover', type=eval, default=True,
                        help='Wether to load an existing checkpoint')
    parser.add_argument('--amp', type=eval, default=False)
    parser.add_argument('-a', '--arch', type=str, default='RAFT_GMA', choices=['RAFT', 'RAFT_GMA'], help="""Type of flows to use""")

    parser.add_argument('--validation_file', type=str,
                        default='./data/TCF_minimal_dataset.hdf5',
                        help='HDF5 file for validation')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--splitSizeTest', default=1, type=int)
    parser.add_argument('--numRAFT_Iter', default=12, type=int,
                        help="""Number of iterations in RAFT.""")
    parser.add_argument('--use_AlternateCorr', type=eval, default=False,
                        help="""Whether or not to compute an unsupervised sequence loss.""")
    #test args
    parser.add_argument('--offset', default=128, type=int)
    parser.add_argument('--shift', default=64, type=int)
    parser.add_argument('--cropSize', default=16, type=int)
    parser.add_argument('--return_values', type=eval, default=True,
                        help="""Whether or not to to the single epe value""")
    parser.add_argument('--downsample', type=eval, default=False,
                        help="""Whether or not to down- and upsample image/flow.""")

    parser.add_argument('--num_GMA_heads', default=1, type=int,
                        help="""Number of attention heads for global motion aggregation.""")
    parser.add_argument('--positional_embedding', type=str, default='none',
                        choices=['position_and_content', 'position_only', 'none'],
                        help="""which positional embedding ist used during training.""")

    args = parser.parse_args()

    assert args.shift != 0, 'Shift has to be greater than zero!'

    RANK = int(os.environ.get('RANK'))
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK'))
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE'))
    DEVICE_ORDINAL = 0 #This is managed by CUDA_VISIBLE_DEVICES

    device = torch.device("cuda:{}".format(DEVICE_ORDINAL))

    assert RANK != None
    assert LOCAL_RANK != None
    assert WORLD_SIZE != None

    if WORLD_SIZE > 0:
        args.is_distributed = True
    else:
        args.is_distributed = False

    set_random_seeds()

    args.dir_inference_results = './output/' + args.name + '/'

    # get trial from optuna
    if RANK == 0:        
        if not os.path.exists(args.dir_inference_results):
            os.makedirs(args.dir_inference_results)

    torch.distributed.init_process_group(backend="nccl")
    dist.barrier()

    model = UnsuperviseFlowPLModule(args=args)

    if RANK == 0:
        print('test data file:', args.validation_file, flush=True)

    validation_dataset = testing_dataset(root=args.validation_file,
                                    source_transform=source_transform,
                                    target_transform=target_transform,
                                    )
    validation_data_sampler = DistributedSampler(dataset=validation_dataset, drop_last=False,
                                                 shuffle=False) if args.is_distributed else None
    validation_loader = DataLoader(dataset=validation_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   pin_memory=False,
                                   drop_last=False,
                                   num_workers=0,
                                   sampler=validation_data_sampler)

    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL)

    if args.recover:
        checkpoint = torch.load(args.input_path_ckpt, map_location=device)
        ddp_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        if RANK == 0:
            print('model recovered', args.input_path_ckpt, flush=True)

    dist.barrier()

    ########################################### Testing #################################################
    ddp_model.eval()
    validation_loss_logs = None

    with torch.no_grad():
        if RANK == 0:
            val_pbar = tqdm(enumerate(validation_loader), desc='Epoch %s / %s VALIDATION' % (1, 1),
                        total=validation_loader.__len__())
        else:
            val_pbar = tqdm(enumerate(validation_loader), desc='Epoch %s / %s VALIDATION' % (1, 1),
                                  total=validation_loader.__len__(), disable=True)

        for i, batch in val_pbar:
            target, flow = batch     

            if i == 0:
                # compute number of patches for folding operation
                B1, C1, H1, W1 = target.size()
                if args.use_AlternateCorr:
                    NUM_Yvectors, NUM_Xvectors = 1, 1                    
                    if args.return_values:
                        # store results
                        results = np.zeros((validation_loader.__len__(), 4, H1 , W1))

                else:
                    if args.offset == H1 and args.offset == W1:
                        NUM_Yvectors, NUM_Xvectors = 1, 1
                    else:
                        NUM_Yvectors, NUM_Xvectors = int((H1) / args.shift - (args.offset / args.shift - 1)), \
                                                    int((W1) / args.shift - (args.offset / args.shift - 1))
                    imageH, imageW = NUM_Yvectors * args.shift + (args.offset - args.shift) , \
                                    NUM_Xvectors * args.shift + (args.offset - args.shift)

                    if args.return_values:
                        # store results
                        results = np.zeros((validation_loader.__len__(), 4, imageH - 2 * args.cropSize , imageW - 2 * args.cropSize))

            if args.use_AlternateCorr:
                    gpu_batch = [{'target': target[:,:,:H1,:].to(device),
                                  'flow': flow[:,:,:H1,:].to(device)}]
            else:
                    gpu_batch = [{'target': target[:, :, :imageH, :imageW].to(device),
                                  'flow': flow[:, :, :imageH, :imageW].to(device)}]

            loss_dict = ddp_model(gpu_batch, i, mode='validation',epoch=0)

            if args.return_values:
                B, _, _, _ = flow.size()
                u_plot_pred = torch.squeeze(loss_dict['predicted_flows'][:, 0, :,:]).detach().cpu().numpy()
                v_plot_pred = torch.squeeze(loss_dict['predicted_flows'][:, 1, :,:]).detach().cpu().numpy()
                if args.use_AlternateCorr:
                    u_plot_gt = torch.squeeze(flow[:, 0, :H1, :]).detach().cpu().numpy()
                    v_plot_gt = torch.squeeze(flow[:, 1, :H1, :]).detach().cpu().numpy()
                else:
                    u_plot_gt = torch.squeeze(flow[:, 0, args.cropSize:imageH - args.cropSize, args.cropSize:imageW - args.cropSize]).detach().cpu().numpy()
                    v_plot_gt = torch.squeeze(flow[:, 1, args.cropSize:imageH - args.cropSize, args.cropSize:imageW - args.cropSize]).detach().cpu().numpy()

                # # store results
                results[i * B:i * B + B, 0, :, :] = u_plot_pred
                results[i * B:i * B + B, 1, :, :] = v_plot_pred
                results[i * B:i * B + B, 2, :, :] = u_plot_gt
                results[i * B:i * B + B, 3, :, :] = v_plot_gt

            # # synchronize all losses for logging in the log dict
            # if validation_loss_logs == None:
            #     validation_loss_logs = {}
            #     for key in loss_dict:
            #         if key != 'predicted_flows':
            #             validation_loss_logs[key] = []
            # for key, value in loss_dict.items():
            #     if key != 'predicted_flows':
            #         dist.all_reduce(loss_dict[key], op=torch.distributed.ReduceOp.SUM)
            #         validation_loss_logs[key].append(loss_dict[key].cpu() / WORLD_SIZE)

        # average all validation losses
        # if RANK==0:
        #     print('overall final epe: ',
        #           torch.stack(validation_loss_logs['test_epe_loss_final']).mean().detach().numpy(),
        #           flush=True)

        # if args.return_values:
        #     splittedDatasetName = args.validation_file.split('/')
        #     DatasetSplit = splittedDatasetName[-1].split('.')

        #     savepath_inference_results = args.dir_inference_results + '/' + DatasetSplit[0] + '_Rank_{:03d}'.format(RANK)

        #     np.save(savepath_inference_results, results)
        
    # plot results
    with h5py.File(args.validation_file, 'r') as file:
        delta_x = file.attrs['deltaXinnerUnits'] 
        delta_y = file.attrs['deltaYinnerUnits']
        res_m_iu = file.attrs['resolutionMeterPerInnerUnits'] # m/(inner units)
        scale_vel = file.attrs['scale2PhysicalVelocity']
    
    wall_RAFT = 16 # only for the synthetic TCF case
    ETA = 1.849e-5 # dynamic viscosity
    # -----------------------------------------------------------------------------------------------

    # remove rows inside the wall and scale to true velocity values [m/s]
    # u_RAFT = u_plot_pred[wall_RAFT:,:,:]*scale_vel   
    # u_baseline = u_plot_gt[wall_RAFT:,:,:]*scale_vel

    u_RAFT = rearrange(results[:, 0, wall_RAFT:, :], 'b h w -> h w b') * scale_vel 
    u_baseline = rearrange(results[:, 2, wall_RAFT:, :], 'b h w -> h w b') * scale_vel

    # coordinates in [inner units]
    x_plus_RAFT = np.linspace(0, (np.shape(u_RAFT)[1]-1)*delta_x, num=np.shape(u_RAFT)[1])
    y_plus_RAFT = np.linspace(delta_y/2, (np.shape(u_RAFT)[0]-1)*delta_y + delta_y/2, num=np.shape(u_RAFT)[0])    

    # y coordinates in [m]
    y_RAFT = y_plus_RAFT*res_m_iu

    # -----------------------------------------------------------------------------------------------

    # WSS, average across viscous sublayer
    sublayer = np.argwhere(y_plus_RAFT>5)[0][0] # index for edge of viscous sublayer (at y+=5)
    tau_RAFT = np.mean(ETA*u_RAFT[:sublayer,:,:]/y_RAFT[:sublayer,None,None], axis=0)
    tau_baseline = np.mean(ETA*u_baseline[:sublayer,:,:]/y_RAFT[:sublayer,None,None], axis=0)

    # -----------------------------------------------------------------------------------------------
    # plot WSS
    fsize = 40

    for timestep in range(np.shape(u_RAFT)[-1]):
        plt.figure(figsize=(20,10))
        plt.plot(x_plus_RAFT, tau_baseline[:,timestep], color="royalblue", label='ground truth', marker='o', markersize=7)
        plt.plot(x_plus_RAFT, tau_RAFT[:,timestep], color="darkred", label='RAFT-PIV', marker='s', markersize=7)    
        plt.legend()
        plt.ylabel(r'$\tau_w$ $[Pa]$', size=fsize)
        plt.xlabel(r'$x^+$', size=fsize)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.legend(fontsize=fsize, frameon=False)
        plt.show()
        plt.savefig(args.dir_inference_results + 'WSSalongx_TS' + str(timestep) + '.png', format='png', bbox_inches = "tight")
        plt.close()

    print('done')


if __name__ == '__main__':
    main()
