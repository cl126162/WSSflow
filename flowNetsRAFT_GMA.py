#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:27:59 2020

@author: clagemann
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from submodules_RAFT_extractor import BasicEncoder
from submodules_RAFT_GRU import BasicUpdateBlock_GMA, Attention

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img.float(), grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.half()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2).type(torch.float)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return (corr  / torch.sqrt(torch.tensor(dim).type(torch.float))).type(torch.float)

class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

def sequence_loss(flow_preds, flow_gt):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    # mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # valid = (valid >= 0.5) & (mag < max_flow)
    # valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = 0.8**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    
    return flow_loss, metrics

class RAFT_GMA(nn.Module):
    """
    RAFT-GMA
    """
    def __init__(self, args):
        super(RAFT_GMA,self).__init__()
        
        self.hidden_dim = 128
        self.context_dim = 128
        self.encoder_dim = 256
        self.corr_levels = 4
        self.corr_radius = 4
        self.downsample = args.downsample
        self.num_heads = args.num_GMA_heads
        self.positional_embedding = args.positional_embedding
        
        self.fnet = BasicEncoder(output_dim=self.encoder_dim, norm_fn='instance', dropout=0., downsample=self.downsample)
        self.cnet = BasicEncoder(output_dim=self.hidden_dim+self.context_dim, norm_fn='instance', dropout=0., downsample=self.downsample)
        self.update_block = BasicUpdateBlock_GMA(hidden_dim=self.hidden_dim, corr_levels=self.corr_levels, corr_radius=self.corr_radius, downsample=self.downsample, num_heads=self.num_heads)
        self.att = Attention(positional_embedding=self.positional_embedding, dim=self.context_dim, heads=self.num_heads, max_pos_size=160, dim_head=self.context_dim)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        if self.downsample == True:
            coords0 = coords_grid(N, H//8, W//8).to(img.device)
            coords1 = coords_grid(N, H//8, W//8).to(img.device)
        else:
            coords0 = coords_grid(N, H, W).to(img.device)
            coords1 = coords_grid(N, H, W).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)


    def forward(self,input, flowl0, args,flow_init=None, upsample=True, backward=False):
        if backward:
            img1 = torch.unsqueeze(input[:, 1, :, :], dim=1)
            img2 = torch.unsqueeze(input[:, 0, :, :], dim=1)
        else:
            img1 = torch.unsqueeze(input[:, 0, :, :], dim=1)
            img2 = torch.unsqueeze(input[:, 1, :, :], dim=1)

        with autocast(enabled=args.amp):
            coords0, coords1 = self.initialize_flow(img1)
            fmap1, fmap2 = self.fnet([img1, img2])
            
            
        if args.use_AlternateCorr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)

        with autocast(enabled=args.amp):
            cnet = self.cnet(img1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)
        
        flow_predictions = []
        for itr in range(args.numRAFT_Iter):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            if itr == 0 and flow_init is not None:
                if self.downsample:
                    flow = F.avg_pool2d(F.avg_pool2d(F.avg_pool2d(flow_init, 2, stride=2), 2, stride=2), 2, stride=2)
                else:
                    flow = flow_init
            else:
                flow = coords1 - coords0
            with autocast(enabled=args.amp):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow = coords1 - coords0

            flow_up = self.upsample_flow(coords1 - coords0, up_mask) if args.downsample == True else flow

            flow_predictions.append(flow_up)
        
        loss = sequence_loss(flow_predictions, flowl0)
        return flow_predictions, loss
