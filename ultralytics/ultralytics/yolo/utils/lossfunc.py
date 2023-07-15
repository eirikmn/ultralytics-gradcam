# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.yolo.utils.metrics import OKS_SIGMA
from ultralytics.yolo.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.loss import BboxLoss

from .metrics import bbox_iou
from .tal import bbox2dist

def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            
        return out


def bbox_decode(self, anchor_points, pred_dist, reg_max):
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    proj = torch.arange(reg_max, dtype=torch.float, device=self.device)
    use_dfl = reg_max > 1
    if use_dfl:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
    return dist2bbox(pred_dist, anchor_points, xywh=False)


def v8detectionlosscomputer(model, preds, targets, device):
    
    h = model.args  # hyperparameters
    #m = model.model[-1]  # Detect() module
    
    m = model.model.model.model[-1]  # Detect() module
    bce = nn.BCEWithLogitsLoss(reduction='none')
    hyp = h
    stride = m.stride  # model strides
    nc = m.nc  # number of classes
    no = m.no
    reg_max = m.reg_max
    use_dfl = reg_max > 1

    assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
    bbox_loss = BboxLoss(reg_max - 1, use_dfl=use_dfl).to(device)
    proj = torch.arange(reg_max, dtype=torch.float, device=device)

    
    loss = torch.zeros(3, device=device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    feats = preds[1] if isinstance(preds, list) else preds

    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
        (reg_max * 4, nc), 1)

    
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

    
    # targets
    #targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
    
    #print(batch_size)
    targets = preprocess(model, targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    # ground truth labels and boxes

    
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy #gound truth
    
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) # ground truth


    # pboxes

    pred_bboxes = bbox_decode(model, anchor_points, pred_distri,reg_max)  # xyxy, (b, h*w, 4)

    
    _, target_bboxes, target_scores, fg_mask, _ = assigner(
        pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

    target_scores_sum = max(target_scores.sum(), 1)

    

    # cls loss
    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    loss[1] = bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # bbox loss
    if fg_mask.sum():
        target_bboxes /= stride_tensor
        loss[0], loss[2] = bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask)

    loss[0] *= hyp.box  # box gain #ignore box loss
    #loss[0] = 0
    loss[1] *= hyp.cls  # cls gain
    loss[2] *= hyp.dfl  # dfl gain

    

    return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
