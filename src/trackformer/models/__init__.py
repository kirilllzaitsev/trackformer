# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from trackformer.models.backbone import build_backbone
from trackformer.models.deformable_detr import DeformableDETR, DeformablePostProcess
from trackformer.models.deformable_transformer import build_deforamble_transformer
from trackformer.models.detr import DETR, PostProcess, SetCriterion
from trackformer.models.detr_segmentation import (
    DeformableDETRSegm,
    DeformableDETRSegmTracking,
    DETRSegm,
    DETRSegmTracking,
    PostProcessPanoptic,
    PostProcessSegm,
)
from trackformer.models.detr_tracking import DeformableDETRTracking, DETRTracking
from trackformer.models.matcher import build_matcher
from trackformer.models.transformer import build_transformer


def build_model(args, num_classes=None):
    # num_classes includes bg
    if num_classes is None:
        if args.dataset == 'coco':
            num_classes = 91
        elif args.dataset == 'coco_panoptic':
            num_classes = 250
        elif args.dataset in ['coco_person', 'mot', 'mot_crowdhuman', 'crowdhuman', 'mot_coco_person']:
            # num_classes = 91
            num_classes = 20
            # num_classes = 1
        else:
            raise NotImplementedError

    device = torch.device(args.device)
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    assert args.focal_loss, "Have to adjust the postprocessing code for CE otherwise"

    opt_only = getattr(args, "opt_only", None)
    t_out_dim = getattr(args, "t_out_dim", 3)
    dropout = getattr(args, "dropout", 0.0)
    dropout_heads = getattr(args, "dropout_heads", 0.0)
    use_kpts = getattr(args, "use_kpts", False)
    use_kpts_as_ref_pt = getattr(args, "use_kpts_as_ref_pt", False)
    use_kpts_as_img = getattr(args, "use_kpts_as_img", False)
    r_num_layers_inc = getattr(args, "r_num_layers_inc", 0)
    use_boxes = opt_only is not None and all(x in opt_only for x in ['boxes'])
    detr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.focal_loss else num_classes,
        'num_queries': args.num_queries,
        'aux_loss': args.aux_loss,
        'use_pose': opt_only is not None and all(x in opt_only for x in ['rot', 't']),
        'rot_out_dim': getattr(args, "rot_out_dim", 4),
        't_out_dim': t_out_dim,
        'dropout': dropout,
        'dropout_heads': dropout_heads,
        'head_num_layers': getattr(args, "head_num_layers", 2),
        'head_hidden_dim': getattr(args, "head_hidden_dim", None),
        'use_boxes': use_boxes,
        'use_kpts': use_kpts,
        'use_kpts_as_ref_pt': use_kpts_as_ref_pt,
        'use_kpts_as_img': use_kpts_as_img,
        'r_num_layers_inc': r_num_layers_inc,
        'overflow_boxes': args.overflow_boxes}

    tracking_kwargs = {
        'track_query_false_positive_prob': args.track_query_false_positive_prob,
        'track_query_false_negative_prob': args.track_query_false_negative_prob,
        'matcher': matcher,
        'backprop_prev_frame': args.track_backprop_prev_frame,}

    mask_kwargs = {
        'freeze_detr': args.freeze_detr}

    if args.deformable:
        transformer = build_deforamble_transformer(args)

        detr_kwargs['transformer'] = transformer
        detr_kwargs['num_feature_levels'] = args.num_feature_levels
        detr_kwargs['with_box_refine'] = args.with_box_refine
        detr_kwargs['two_stage'] = args.two_stage
        detr_kwargs['multi_frame_attention'] = args.multi_frame_attention
        detr_kwargs['multi_frame_encoding'] = args.multi_frame_encoding
        detr_kwargs['merge_frame_features'] = args.merge_frame_features

        if args.tracking:
            if args.masks:
                model = DeformableDETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs)
            else:
                model = DeformableDETRTracking(tracking_kwargs, detr_kwargs)
        else:
            if args.masks:
                model = DeformableDETRSegm(mask_kwargs, detr_kwargs)
            else:
                model = DeformableDETR(**detr_kwargs)
    else:
        transformer = build_transformer(args)

        detr_kwargs['transformer'] = transformer

        if args.tracking:
            if args.masks:
                model = DETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs)
            else:
                model = DETRTracking(tracking_kwargs, detr_kwargs)
        else:
            if args.masks:
                model = DETRSegm(mask_kwargs, detr_kwargs)
            else:
                model = DETR(**detr_kwargs)

    criterion = build_criterion(args, num_classes, matcher, device)

    if args.focal_loss:
        postprocessors = {'bbox': DeformablePostProcess()}
    else:
        postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


def build_criterion(args, num_classes, matcher, device, use_rel_pose=False):
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_factors_scale': 1,
                   'loss_factors_occlusion': 1,
                   'loss_factors_texture': 1,
                   "loss_rot": getattr(args, "rot_loss_coef", 1),
                   "loss_depth": getattr(args, "depth_loss_coef", 1),
                   "loss_t": getattr(args, "t_loss_coef", 1)}
                   
    losses = [
        "labels",
        "boxes",
        "rot",
        "t",
    ]
    if getattr(args, "opt_only", None) is not None:
        losses = [v for v in losses if v in args.opt_only]

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses += ['cardinality']
    if args.masks:
        losses.append('masks')

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        t_out_dim=args.t_out_dim,
        use_rel_pose=use_rel_pose,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,
        factors=args.factors,
    )
    criterion.to(device)
    return criterion
