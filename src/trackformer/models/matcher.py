# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from trackformer.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        #
        # [batch_size * num_queries, num_classes]
        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class.cpu()[:, tgt_ids.cpu()] - neg_cost_class.cpu()[:, tgt_ids.cpu()]
            cost_class = cost_class.to(out_prob.device)
        else:
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        if self.cost_bbox > 0 or self.cost_giou > 0:
            # [batch_size * num_queries, 4]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox))
        else:
            cost_bbox = 0
            cost_giou = 0

        # Final cost matrix
        try:
            cost_matrix = self.cost_bbox * cost_bbox \
            + self.cost_class * cost_class \
            + self.cost_giou * cost_giou
        except Exception as e:
            print(f"{cost_bbox.shape=} {cost_class.shape=} {cost_giou.shape=}")
            print(f"{cost_bbox=}\n {cost_class=}\n {cost_giou=}\n")
            print(f"{outputs=}\n")
            print(f"{targets=}\n")
            raise ValueError(f"Error in cost matrix calculation\n{e}")
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        for i, target in enumerate(targets):
            if 'track_query_match_ids' not in target:
                continue

            prop_i = 0
            for j in range(cost_matrix.shape[1]):
                # if target['track_queries_fal_pos_mask'][j] or target['track_queries_placeholder_mask'][j]:
                if target['track_queries_fal_pos_mask'][j]:
                    # false positive and palceholder track queries should not
                    # be matched to any target
                    cost_matrix[i, j] = np.inf
                elif target['track_queries_mask'][j]:
                    track_query_id = target['track_query_match_ids'][prop_i]
                    prop_i += 1

                    cost_matrix[i, j] = np.inf
                    cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf
                    cost_matrix[i, j, track_query_id + sum(sizes[:i])] = -1

        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def build_matcher(args):
    if getattr(args, "opt_only", None) is not None:
        if "labels" not in args.opt_only:
            args.set_cost_class = 0
        if "boxes" not in args.opt_only:
            args.set_cost_bbox = 0
            args.set_cost_giou = 0
            
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,)
