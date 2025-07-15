# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from pose_tracking.utils.kpt_utils import extract_kpts
from pose_tracking.utils.misc import print_cls
from torch import nn
from trackformer.models.detr import DETR, PostProcess, SetCriterion
from trackformer.util import box_ops
from trackformer.util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(DETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 multi_frame_attention=False, multi_frame_encoding=False, merge_frame_features=False, use_pose=False, use_depth=False, use_boxes=True,
                 rot_out_dim=4, t_out_dim=3, dropout=0.0, dropout_heads=0.0, use_kpts=False, use_kpts_as_ref_pt=False, use_kpts_as_img=False,
                 head_num_layers=2, head_hidden_dim=None, r_num_layers_inc=0,
                 factors=None,
                roi_feature_dim=256,
                use_render_token=False,
                use_uncertainty=False,
                use_pose_tokens=False,
                n_layers_f_transformer=1,
                use_nocs=False,
                use_nocs_pred=False,
                use_nocs_pose_pred=False,
                use_spherical_nocs=False,
                ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss, use_pose=use_pose, use_depth=use_depth, use_boxes=use_boxes,
                         rot_out_dim=rot_out_dim, t_out_dim=t_out_dim, dropout=dropout, dropout_heads=dropout_heads,
                         use_kpts=use_kpts, use_kpts_as_ref_pt=use_kpts_as_ref_pt, use_kpts_as_img=use_kpts_as_img,
                         head_num_layers=head_num_layers, head_hidden_dim=head_hidden_dim, r_num_layers_inc=r_num_layers_inc,
                         factors=factors,
                        roi_feature_dim=roi_feature_dim,
                        use_render_token=use_render_token,
                        use_uncertainty=use_uncertainty,
                        use_pose_tokens=use_pose_tokens,
                        n_layers_f_transformer=n_layers_f_transformer,
                        use_nocs=use_nocs,
                        use_nocs_pred=use_nocs_pred,
                        use_nocs_pose_pred=use_nocs_pose_pred,
                        use_spherical_nocs=use_spherical_nocs,
)

        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding
        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for
        # region proposal generation
        num_pred = transformer.decoder.num_layers
        if two_stage:
            num_pred += 1

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if self.use_pose:
                self.rot_embed = _get_clones(self.rot_embed, num_pred)
                self.t_embed = _get_clones(self.t_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if self.use_pose:
                self.rot_embed = nn.ModuleList([self.rot_embed for _ in range(num_pred)])
                self.t_embed = nn.ModuleList([self.t_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if self.merge_frame_features:
            self.merge_features = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = _get_clones(self.merge_features, num_feature_levels)

        if use_kpts_as_img:
            self.input_proj = None

    # def fpn_channels(self):
    #     """ Returns FPN channels. """
    #     num_backbone_outs = len(self.backbone.strides)
    #     return [self.hidden_dim, ] * num_backbone_outs

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None, pose_renderer_fn=None, coformer_kwargs=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        if prev_features is None:
            kpt_extractor_res_prev = None
            self.kpt_extractor_res_prev = None
        else:
            kpt_extractor_res_prev = self.kpt_extractor_res_prev

        if self.use_kpts_as_img:
            src_list=[ 
                torch.randn(samples.tensors.shape[0], self.hidden_dim, 60, 80).to(samples.tensors.device),
                torch.randn(samples.tensors.shape[0], self.hidden_dim, 30, 40).to(samples.tensors.device),
                torch.randn(samples.tensors.shape[0], self.hidden_dim, 15, 20).to(samples.tensors.device),
                torch.randn(samples.tensors.shape[0], self.hidden_dim, 8, 10).to(samples.tensors.device)
            ] * 2
            mask_list = [torch.zeros_like(x).bool().to(samples.tensors.device)[:,0] for x in src_list]
            pos_list = [torch.rand_like(x).to(samples.tensors.device) for x in src_list]
            features_all= None
        else:
            features, pos = self.backbone(samples)

            features_all = features
            # pos_all = pos
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            features = features[-3:]
            # pos = pos[-3:]

            if prev_features is None:
                prev_features = features
                kpt_extractor_res_prev = None
            else:
                prev_features = prev_features[-3:]
                kpt_extractor_res_prev = self.kpt_extractor_res_prev

            # srcs = []
            # masks = []
            src_list = []
            mask_list = []
            pos_list = []
            # for l, (feat, prev_feat) in enumerate(zip(features, prev_features)):

            frame_features = [prev_features, features]
            if not self.multi_frame_attention:
                frame_features = [features]

            for frame, frame_feat in enumerate(frame_features):
                if self.multi_frame_attention and self.multi_frame_encoding:
                    pos_list.extend([p[:, frame] for p in pos[-3:]])
                else:
                    pos_list.extend(pos[-3:])

                # src, mask = feat.decompose()

                # prev_src, _ = prev_feat.decompose()

                for l, feat in enumerate(frame_feat):
                    src, mask = feat.decompose()

                    if self.merge_frame_features:
                        prev_src, _ = prev_features[l].decompose()
                        src_list.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
                    else:
                        src_list.append(self.input_proj[l](src))

                    mask_list.append(mask)

                # if hasattr(self, 'merge_features'):
                #     srcs.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
                # else:
                #     srcs.append(self.input_proj[l](src))

                # masks.append(mask)
                    assert mask is not None

                if self.num_feature_levels > len(frame_feat):
                    _len_srcs = len(frame_feat)
                    for l in range(_len_srcs, self.num_feature_levels):
                        if l == _len_srcs:
                            # src = self.input_proj[l](frame_feat[-1].tensors)
                            # if hasattr(self, 'merge_features'):
                            #     src = self.merge_features[l](torch.cat([self.input_proj[l](features[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                            # else:
                            #     src = self.input_proj[l](features[-1].tensors)

                            if self.merge_frame_features:
                                src = self.merge_features[l](torch.cat([self.input_proj[l](frame_feat[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                            else:
                                src = self.input_proj[l](frame_feat[-1].tensors)
                        else:
                            src = self.input_proj[l](src_list[-1])
                            # src = self.input_proj[l](srcs[-1])
                        # m = samples.mask
                        _, m = frame_feat[0].decompose()
                        mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                        pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                        src_list.append(src)
                        mask_list.append(mask)
                        if self.multi_frame_attention and self.multi_frame_encoding:
                            pos_list.append(pos_l[:, frame])
                        else:
                            pos_list.append(pos_l)

        if self.use_kpts:
            kpt_extractor_res = extract_kpts(samples.tensors, extractor=self.extractor, do_normalize=True, use_zeros_for_pad=self.use_kpts_as_img and not self.use_kpts_as_ref_pt)
            self.kpt_extractor_res_prev = kpt_extractor_res
            # merge kpt_extractor_res with kpt_extractor_res_prev
            kpt_extractor_res_prev = kpt_extractor_res if kpt_extractor_res_prev is None else kpt_extractor_res_prev
            if kpt_extractor_res_prev is not None:
                kpt_extractor_res = {k: torch.cat([kpt_extractor_res[k], kpt_extractor_res_prev[k]], dim=1) if kpt_extractor_res[k] is not None else None for k in kpt_extractor_res}
        else:
            kpt_extractor_res=None

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(src_list, mask_list, pos_list, query_embeds, targets, kpt_extractor_res=kpt_extractor_res,
                             use_kpts_as_ref_pt=self.use_kpts_as_ref_pt, use_kpts_as_img=self.use_kpts_as_img)

        outputs_classes = []
        outputs_coords = []
        outputs_rots = []
        outputs_ts = []
        outputs_depths = []
        outs_fdetr = [] if self.use_uncertainty else None
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.use_uncertainty:
                tmp, last_latent_bbox = self.bbox_embed[lvl](hs[lvl])
            else:
                tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.use_pose:
                if self.use_uncertainty:
                    outputs_rot, last_latent_rot = self.rot_embed[lvl](hs[lvl])
                    outputs_t, last_latent_t = self.t_embed[lvl](hs[lvl])
                else:
                    outputs_rot = self.rot_embed[lvl](hs[lvl])
                    outputs_t = self.t_embed[lvl](hs[lvl])
                if self.do_predict_2d_t:
                    if self.use_uncertainty:
                        outputs_depth, last_latent_depth = self.depth_embed(hs[lvl])
                        last_latent_t = last_latent_bbox
                    else:
                        outputs_depth = self.depth_embed(hs[lvl])
                    outputs_depths.append(outputs_depth)
                    # outputs_t = F.sigmoid(outputs_t)
                    outputs_t = outputs_coord[..., :2]
                outputs_rots.append(outputs_rot)
                outputs_ts.append(outputs_t)

                if self.use_uncertainty:
                    rt_latents=[last_latent_rot, last_latent_t]
                    if self.do_predict_2d_t:
                        rt_latents.append(last_latent_depth)
                    out_fdetr = self.coformer(
                        hs[lvl],
                        rgb=samples.tensors,
                        pred_boxes=outputs_coord,
                        rt_latents=rt_latents,
                        layer_idx=lvl,
                        pose_token=None if self.use_pose_tokens else None,
                        pose_renderer_fn=pose_renderer_fn,
                        out_rt={"t": outputs_t, "rot": outputs_rot},
                        coformer_kwargs=coformer_kwargs,
                    )
                    outs_fdetr.append(out_fdetr)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs[-1]}
        outputs_depth = None
        if self.use_pose:
            outputs_rot = torch.stack(outputs_rots)
            outputs_t = torch.stack(outputs_ts)
            out['rot'] = outputs_rot[-1]
            out['t'] = outputs_t[-1]

            if self.do_predict_2d_t:
                outputs_depth = torch.stack(outputs_depths)
                out["center_depth"] = outputs_depth[-1]
            if self.use_uncertainty:
                for k, v in outs_fdetr[-1].items():
                    out[k] = v
        else:
            outputs_rot = None
            outputs_t = None
            outputs_depth = None

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_rot, outputs_t, outputs_depth, outs_fdetr)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        if self.use_kpts_as_img:
            kpts= kpt_extractor_res['keypoints']
            num_kpts = kpts.shape[1]
            while num_kpts > 0 and (num_kpts % 2 != 0 or num_kpts // 2 % 2 != 0):
                num_kpts -= 1
            height=1
            width=num_kpts
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
        else:
            for src in src_list:
                _, _, height, width = src.shape
                memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                    batch_size, channels, height, width)
                memory_slices.append(memory_slice)
                offset += height * width

        memory = memory_slices
        # memory = memory_slices[-1]
        # features = [NestedTensor(memory_slide) for memory_slide in memory_slices]

        return out, targets, features_all, memory, hs

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class, outputs_coord, outputs_rot=None, outputs_t=None, outputs_depth=None, outs_fdetr=None
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        res = []
        for i in range(len(outputs_class) - 1):
            res_lvl = {"pred_logits": outputs_class[i], "pred_boxes": outputs_coord[i]}
            if outputs_rot is not None:
                res_lvl["rot"] = outputs_rot[i]
            if outputs_t is not None:
                res_lvl["t"] = outputs_t[i]
            if outputs_depth is not None:
                res_lvl["center_depth"] = outputs_depth[i]
            if outs_fdetr is not None:
                for k, v in outs_fdetr[i].items():
                    res_lvl[k] = v
            res.append(res_lvl)
        return res


class DeformablePostProcess(PostProcess):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        ###
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        ###

        scores, labels = prob.max(-1)
        # scores, labels = prob[..., 0:1].max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
