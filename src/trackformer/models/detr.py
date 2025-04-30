# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy

import torch
import torch.nn.functional as F
from pose_tracking.losses import geodesic_loss_mat
from pose_tracking.metrics import calc_r_error, calc_t_error
from pose_tracking.utils.geom import convert_2d_t_to_3d
from pose_tracking.utils.kpt_utils import load_extractor
from pose_tracking.utils.misc import init_params, print_cls
from pose_tracking.utils.pose import convert_rot_vector_to_matrix
from torch import nn

from trackformer.util import box_ops
from trackformer.util.misc import (
    NestedTensor,
    accuracy,
    dice_loss,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    sigmoid_focal_loss,
)


class DETR(nn.Module):
    """This is the DETR module that performs object detection."""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        overflow_boxes=False,
        use_pose=False,
        use_boxes=True,
        rot_out_dim=4,
        t_out_dim=3,
        dropout=0.0,
        dropout_heads=0.0,
        use_depth=False,
        use_kpts=False,
        use_kpts_as_ref_pt=False,
        use_kpts_as_img=False,
        head_num_layers=2,
        head_hidden_dim=None,
        r_num_layers_inc=0,
        uncertainty_coef=0.1,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO, we
                         recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.use_depth = use_depth
        self.use_boxes = use_boxes
        self.use_kpts = use_kpts
        self.use_kpts_as_ref_pt = use_kpts_as_ref_pt
        self.use_kpts_as_img = use_kpts_as_img
        self.use_pose = use_pose
        self.rot_out_dim = rot_out_dim
        self.t_out_dim = t_out_dim
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.head_num_layers = head_num_layers
        self.r_num_layers_inc = r_num_layers_inc
        self.uncertainty_coef = uncertainty_coef

        self.do_predict_2d_t = t_out_dim == 2
        self.head_hidden_dim = head_hidden_dim or transformer.d_model

        self.num_queries = num_queries
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, output_dim=4, num_layers=head_num_layers, dropout=dropout_heads)
        if not use_boxes:
            for p in self.bbox_embed.parameters():
                p.requires_grad = False

        if use_kpts:
            self.extractor = load_extractor(features="superpoint", max_num_keypoints=1024)
            if use_kpts_as_ref_pt:
                for p in self.extractor.parameters():
                    p.requires_grad = False

        if use_pose:
            self.rot_embed = MLP(self.hidden_dim, self.head_hidden_dim, rot_out_dim, num_layers=head_num_layers+r_num_layers_inc, dropout=dropout_heads)
            self.t_embed = MLP(self.hidden_dim, self.head_hidden_dim, t_out_dim, num_layers=head_num_layers, dropout=dropout_heads)
        if self.do_predict_2d_t:
            self.depth_embed = MLP(
                input_dim=self.hidden_dim,
                output_dim=1,
                hidden_dim=self.head_hidden_dim,
                num_layers=head_num_layers,
                dropout=dropout_heads,
            )
            # use bbox 2d for t
            for p in self.t_embed.parameters():
                p.requires_grad = False

        init_params(self, included_names=['rot_embed', 't_embed', 'depth_embed'])

        if use_kpts_as_img:
            self.backbone=None
        else:
            # match interface with deformable DETR
            self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
            # self.input_proj = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
            #     )])

            self.backbone = backbone
        self.aux_loss = aux_loss

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        # src = self.input_proj[-1](src)
        src = self.input_proj(src)
        pos = pos[-1]

        batch_size, _, _, _ = src.shape

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = None
        if targets is not None and 'track_query_hs_embeds' in targets[0]:
            # [BATCH_SIZE, NUM_PROBS, 4]
            track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])

            num_track_queries = track_query_hs_embeds.shape[1]

            track_query_embed = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed.device)
            query_embed = torch.cat([
                track_query_embed,
                query_embed], dim=0)

            tgt = torch.zeros_like(query_embed)
            tgt[:num_track_queries] = track_query_hs_embeds.transpose(0, 1)

            for i, target in enumerate(targets):
                target['track_query_hs_embeds'] = tgt[:, i]

        assert mask is not None
        hs, hs_without_norm, memory = self.transformer(
            src, mask, query_embed, pos, tgt)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs_without_norm[-1]}

        outputs_depth = None
        if self.use_pose:
            outputs_rot = self.rot_embed(hs)
            outputs_t = self.t_embed(hs)
            if self.do_predict_2d_t:
                outputs_depth = self.depth_embed(hs)
                out["center_depth"] = outputs_depth[-1]
                outputs_t[-1] = F.sigmoid(outputs_t[-1])
            out["rot"] = outputs_rot[-1]
            out["t"] = outputs_t[-1]
        else:
            outputs_rot = None
            outputs_t = None
        forward_pose_heads_res = {
            "outputs_rot": outputs_rot,
            "outputs_t": outputs_t,
            "outputs_depth": outputs_depth,
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, **forward_pose_heads_res,
            )

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class, outputs_coord, outputs_rot=None, outputs_t=None, outputs_depth=None
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
            res.append(res_lvl)
        return res

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        focal_loss,
        focal_alpha,
        focal_gamma,
        tracking,
        track_query_false_positive_eos_weight,
        t_out_dim=3,
        use_rel_pose=False,
        use_pose_refinement=False,
        factors=None,
        uncertainty_coef=0.1,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                    available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tracking = tracking
        self.track_query_false_positive_eos_weight = (
            track_query_false_positive_eos_weight
        )
        self.t_out_dim = t_out_dim
        self.factors=factors
        
        self.use_rel_pose = use_rel_pose
        self.use_pose_refinement = use_pose_refinement

        self.use_factors = factors is not None
        self.focal_alpha_confidence = 0.5
        self.mean_delta_t, self.mean_delta_rot = (
                    torch.tensor([0.0287253, 0.0011501, 0.0197429], device=self.device),
                    torch.tensor(
                        [0.01121484, 0.00856275, 0.00846249, 0.00392139, 0.00301261, 0.00935597], device=self.device
                    ),
                )
        self.mean_delta_t, self.mean_delta_rot = 1,1

        if self.use_factors:
            self.losses.append("factors")

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())

    def loss_labels(self, outputs, targets, indices, _, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                  target_classes,
                                  weight=self.empty_weight,
                                  reduction='none')

        if self.tracking and self.track_query_false_positive_eos_weight:
            for i, target in enumerate(targets):
                if 'track_query_boxes' in target:
                    # remove no-object weighting for false track_queries
                    loss_ce[i, target['track_queries_fal_pos_mask']] *= 1 / self.eos_coef
                    # assign false track_queries to some object class for the final weighting
                    target_classes = target_classes.clone()
                    target_classes[i, target['track_queries_fal_pos_mask']] = 0

        # weight = None
        # if self.tracking:
        #     weight = torch.stack([~t['track_queries_placeholder_mask'] for t in targets]).float()
        #     loss_ce *= weight

        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        # query_mask = None
        # if self.tracking:
        #     query_mask = torch.stack([~t['track_queries_placeholder_mask'] for t in targets])[..., None]
        #     query_mask = query_mask.repeat(1, 1, self.num_classes)

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma)
            # , query_mask=query_mask)

        # if self.tracking:
        #     mean_num_queries = torch.tensor([len(m.nonzero()) for m in query_mask]).float().mean()
        #     loss_ce *= mean_num_queries
        # else:
        #     loss_ce *= src_logits.shape[1]
        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        # compute seperate track and object query losses
        # loss_ce = sigmoid_focal_loss(
        #     src_logits, target_classes_onehot, num_boxes,
        #     alpha=self.focal_alpha, gamma=self.focal_gamma, query_mask=query_mask, reduction=False)
        # loss_ce *= src_logits.shape[1]

        # track_query_target_masks = []
        # for t, ind in zip(targets, indices):
        #     track_query_target_mask = torch.zeros_like(ind[1]).bool()

        #     for i in t['track_query_match_ids']:
        #         track_query_target_mask[ind[1].eq(i).nonzero()[0]] = True

        #     track_query_target_masks.append(track_query_target_mask)
        # track_query_target_masks = torch.cat(track_query_target_masks)

        # losses['loss_ce_track_queries'] = loss_ce[idx][track_query_target_masks].mean(1).sum() / num_boxes
        # losses['loss_ce_object_queries'] = loss_ce[idx][~track_query_target_masks].mean(1).sum() / num_boxes

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of
            predicted non-empty boxes. This is not really a loss, it is intended
            for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # compute seperate track and object query losses
        # track_query_target_masks = []
        # for t, ind in zip(targets, indices):
        #     track_query_target_mask = torch.zeros_like(ind[1]).bool()

        #     for i in t['track_query_match_ids']:
        #         track_query_target_mask[ind[1].eq(i).nonzero()[0]] = True

        #     track_query_target_masks.append(track_query_target_mask)
        # track_query_target_masks = torch.cat(track_query_target_masks)

        # losses['loss_bbox_track_queries'] = loss_bbox[track_query_target_masks].sum() / num_boxes
        # losses['loss_bbox_object_queries'] = loss_bbox[~track_query_target_masks].sum() / num_boxes

        # losses['loss_giou_track_queries'] = loss_giou[track_query_target_masks].sum() / num_boxes
        # losses['loss_giou_object_queries'] = loss_giou[~track_query_target_masks].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of
           dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_factors(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        losses = {}
        for f in self.factors:
            src_f_logits = outputs["factors"][f].cpu()[idx].cuda()
            target_fs = (
                torch.cat(
                    [t["factors"][f][i] for t, (_, i) in zip(targets, indices)], dim=0
                )
                .unsqueeze(-1)
                .float()
            )
            target_f_buckets = bucketize_soft_labels(target_f_buckets, num_buckets=10)
            loss_f = F.cross_entropy(
                src_f_logits, target_f_buckets, reduction="none"
            )
            losses[f"loss_factors_{f}"] = loss_f

        losses = {k: v.sum() / num_boxes for k, v in losses.items()}
        return losses

    def loss_rot(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        indices = self.filter_out_idxs_for_rel_pose(targets, indices)

        idx = self._get_src_permutation_idx(indices)

        src_rots = outputs["rot"][idx]
        tgt_key = "rot_rel" if self.use_rel_pose else "rot"
        target_rots = (
            [t[tgt_key][i] for t, (_, i) in zip(targets, indices) if len(t[tgt_key])>0]
        )
        # if self.use_rel_pose:
        #     target_rots = [r[valid_target_idxs[i]] for i, r in enumerate(target_rots)]
        if len(target_rots)==0:
            return {}
        target_rots=torch.cat(target_rots, dim=0)
        losses = {}
        loss_rot = F.mse_loss(src_rots, target_rots, reduction="none")

        losses["loss_rot"] = loss_rot
        losses = {k: v.sum() / num_boxes for k, v in losses.items()}

        if self.use_factors:
            target_rot_mats = convert_rot_vector_to_matrix(target_rots)
            src_rot_mats = convert_rot_vector_to_matrix(src_rots).detach()
            r_err_deg = geodesic_loss_mat(
                src_rot_mats,
                target_rot_mats,
                sym_type="",
                do_return_deg=True,
                do_reduce=False,
            )
            losses["r_err_deg"] = r_err_deg

        return losses

    def filter_out_idxs_for_rel_pose(self, targets, indices):
        indices = copy.deepcopy(indices)
        if self.use_rel_pose:
            # ensure idx[1] contains target idxs of objs present in both frames
            tgt_prev_visib_idxs = [t['prev_target']['visible_obj_idxs'] for t in targets]
            tgt_visib_idxs = [t['visible_obj_idxs'] for t in targets]
            valid_target_idxs = [torch.tensor([i for i in prev_visib_idxs if i in visib_idxs]) for prev_visib_idxs, visib_idxs in zip(tgt_prev_visib_idxs, tgt_visib_idxs)]
            for bidx, indices_b in enumerate(indices):
                idx_mask = torch.isin(indices_b[1], valid_target_idxs[bidx])
                indices[bidx] = (indices_b[0][idx_mask], indices_b[1][idx_mask])
        return indices

    def loss_t(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        indices = self.filter_out_idxs_for_rel_pose(targets, indices)
        idx = self._get_src_permutation_idx(indices)
        src_ts = outputs["t"][idx]
        losses = {}
        if self.t_out_dim == 2:
            src_depths = outputs["center_depth"][idx]
            tgt_key_depth="center_depth"
            if self.use_rel_pose:
                tgt_key_depth += "_rel"
            target_depths = ([t[tgt_key_depth][i] for t, (_, i) in zip(targets, indices) if len(t[tgt_key_depth])>0])
            if len(target_depths)==0:
                return {}
            target_depths=torch.cat(target_depths, dim=0)
            loss_depth = F.mse_loss(src_depths, target_depths, reduction="none")
            losses["loss_depth"] = loss_depth.sum() / num_boxes
            tgt_key = "xy"
        else:
            tgt_key = "t"

        if self.use_rel_pose:
            tgt_key += "_rel"

        target_ts = ([t[tgt_key][i] for t, (_, i) in zip(targets, indices) if len(t[tgt_key])>0])
        if len(target_ts)==0:
            return {}
        target_ts=torch.cat(target_ts, dim=0)
        loss_t = F.mse_loss(src_ts, target_ts, reduction="none")
        losses["loss_t"] = loss_t
        losses = {k: v.sum() / num_boxes for k, v in losses.items()}

        if self.use_factors:
            if self.t_out_dim == 2:
                src_depths = outputs["center_depth"][idx]
                intrinsics = torch.stack(
                    [t["intrinsics"] for t, (_, _) in zip(targets, indices)]
                )
                hw = torch.stack([t["size"] for t, (_, _) in zip(targets, indices)])
                convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                    src_ts, src_depths, intrinsics, hw=hw
                )
                src_ts_3d = convert_2d_t_pred_to_3d_res["t_pred"]
                target_ts_3d = torch.cat(
                    [t["t"] for t, (_, _) in zip(targets, indices)]
                )
                # TODO: predicting 2d and calc uncertainty in 3d seems suboptimal
            else:
                src_ts_3d = src_ts
                target_ts_3d = target_ts
            src_ts_3d = src_ts_3d.detach()

            t_err_cm = calc_t_error(src_ts_3d, target_ts_3d, do_reduce=False) * 1e2
            losses["t_err_cm"] = t_err_cm

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'rot': self.loss_rot,
            't': self.loss_t,
        }
        if self.use_factors:
            loss_map.update({
                'factors': self.loss_factors,
            })
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            loss_value = self.get_loss(loss, outputs, targets, indices, num_boxes)
            losses.update(loss_value)
        if self.use_factors:
            indices = self.filter_out_idxs_for_rel_pose(targets, indices)
            get_uncertainty_loss_res = self.get_uncertainty_loss(outputs, indices=indices, r_err_deg=losses.pop("r_err_deg"), t_err_cm=losses.pop("t_err_cm"))
            losses.update(get_uncertainty_loss_res)
        losses["indices"] = indices

        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if loss == "factors" and "factors" not in aux_outputs:
                        continue
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if f"r_err_deg_{i}" in losses and f"t_err_cm_{i}" in losses:
                    get_uncertainty_loss_res = self.get_uncertainty_loss(aux_outputs, indices=indices, r_err_deg=losses.pop(f"r_err_deg_{i}"), t_err_cm=losses.pop(f"t_err_cm_{i}"))
                    losses.update({f"{k}_{i}":v for k,v in get_uncertainty_loss_res.items()})

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def get_uncertainty_loss(self, outputs, indices, r_err_deg, t_err_cm):
        idx = self._get_src_permutation_idx(indices)
        log_var, var = self.get_uncertainty(outputs, idx)
        r_err_ind = error_to_confidence(r_err_deg, min_err=5.0, max_err=30.0)
        t_err_ind = error_to_confidence(t_err_cm, min_err=3.0, max_err=30.0)
        gt_confidence = r_err_ind * t_err_ind

        # might benefit from temporal weight decay across time (confidence decreases proportial to tracking length)
        var = var.squeeze(-1)
        loss_uncertainty = F.binary_cross_entropy(
                var, gt_confidence.float()
            )
        confidence = var.detach().mean()
        return {
            "loss_uncertainty": loss_uncertainty,
            "confidence": confidence,
        }

    def get_uncertainty(self, outputs, idx):
        log_var = outputs["uncertainty"][idx][:, None].clamp(min=-5, max=0)
        var = torch.exp(log_var)
        return log_var, var


def error_to_confidence(err, min_err=5.0, max_err=30.0):
    """
    Given an error value, returns a confidence value between 0 and 1,
    where err <= min_err => 1.0 and err >= max_err => 0.0.
    Errors in between are linearly mapped.
    """
    # Linear mapping:
    conf = 1.0 - (err - min_err) / (max_err - min_err)
    # Clamp to [0, 1]:
    return torch.clamp(conf, 0.0, 1.0)


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = self.process_boxes(out_bbox, target_sizes)


        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'scores_no_object': s_n_o}
            for s, l, b, s_n_o in zip(scores, labels, boxes, prob[..., -1])]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropout(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x
