#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch
from ..functions import ms_deform_attn_core_pytorch_mot


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, im2col_step=64, do_vis=False):
        super().__init__()
        assert d_model % n_heads == 0

        self.im2col_step = im2col_step

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self.do_vis = True
        self.do_vis = False

        if self.do_vis:
            self.outs={}
            self.out_counter=0

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        grid_init = torch.tensor([-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1], dtype=torch.float32) \
            .view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None, query_attn_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        if not hasattr(self, "do_vis"):
            self.do_vis=True
            self.outs={}
            self.out_counter=0
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if query_attn_mask is not None:
            attention_weights = attention_weights.masked_fill(query_attn_mask[..., None, None, None], float(0))

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / input_spatial_shapes[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        if self.do_vis:
            point = []
            attention_weight = []


            sampling_locations_vis = sampling_locations.detach().cpu()
            attention_weights_vis = attention_weights.detach().cpu()
            bidx=0
            point=(sampling_locations_vis[bidx,:,:,:,:,:])
            attention_weight=(attention_weights_vis[bidx,:,:,:,:])
            self.outs.update({self.out_counter: {'point': point, 'attention_weight': attention_weight,
                                                 'input_flatten': input_flatten.detach().cpu(),
                                                 'input_spatial_shapes': input_spatial_shapes.detach().cpu()}})
            self.out_counter+=1
                
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
