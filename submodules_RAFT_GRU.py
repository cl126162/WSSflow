#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 07:23:34 2020

@author: clagemann
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        cor_planes = self.corr_levels * (2*self.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=128, input_dim=128, downsample=False):
        super(BasicUpdateBlock, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.encoder = BasicMotionEncoder(self.corr_levels, self.corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.downsample = downsample

        if self.downsample == True:
            self.mask = nn.Sequential(
                   nn.Conv2d(128, 256, 3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        if self.downsample == True:
            mask = .25 * self.mask(net)
        else:
            mask = None
        return net, mask, delta_flow

class BasicUpdateBlock_GMA(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=128, input_dim=128, downsample=False, num_heads=8):
        super(BasicUpdateBlock_GMA, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.encoder = BasicMotionEncoder(self.corr_levels, self.corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.downsample = downsample
        self.num_heads = num_heads

        if self.downsample == True:
            self.mask = nn.Sequential(
                   nn.Conv2d(128, 256, 3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(dim=128, dim_head=128, heads=self.num_heads)

    def forward(self, net, inp, corr, flow, attention, upsample=True):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp = torch.cat([inp, motion_features, motion_features_global], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        if self.downsample == True:
            mask = .25 * self.mask(net)
        else:
            mask = None
        return net, mask, delta_flow

class Aggregate(nn.Module):
    def __init__(
        self,
        # args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        # self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out

class Attention(nn.Module):
    def __init__(
        self,
        *,
        positional_embedding,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.positional_embedding = positional_embedding
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        if self.positional_embedding != 'none':
            self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        if self.positional_embedding == 'position_only':
            sim = self.pos_emb(q)

        elif self.positional_embedding == 'position_and_content':
            sim_content = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
            sim_pos = self.pos_emb(q)
            sim = sim_content + sim_pos

        else:
            sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        return attn


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score
