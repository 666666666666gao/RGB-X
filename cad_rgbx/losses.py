"""Equal task means of response + 2 GIoU + 5 L1; no routing auxiliary loss."""
import torch
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap

TASKS = ['lasher', 'depthtrack', 'visevent']


def localization_losses(prediction, batch, cfg):
    boxes = prediction['pred_boxes']
    assert boxes.shape[1] == 1 and torch.isfinite(boxes).all()
    predicted = box_cxcywh_to_xyxy(boxes[:, 0])
    target = box_xywh_to_xyxy(batch['search_anno'][-1]).clamp(0, 1)
    giou, iou = generalized_box_iou(predicted, target)
    l1 = (predicted - target).abs().mean(1)
    heatmap = generate_heatmap(batch['search_anno'], cfg.DATA.SEARCH.SIZE,
                               cfg.MODEL.BACKBONE.STRIDE)[-1].unsqueeze(1)
    task_ids = torch.tensor([TASKS.index(name) for name in batch['dataset']], device=boxes.device)
    losses, metrics, counts = [], [], []
    for task in range(3):
        mask = task_ids == task
        assert mask.any()
        response = FocalLoss()(prediction['score_map'][mask], heatmap[mask])
        box = 2 * (1 - giou[mask]).mean() + 5 * l1[mask].mean()
        losses.append(response + box)
        metrics.append(torch.stack([response.detach(), box.detach(), iou[mask].mean().detach()]))
        counts.append(mask.sum())
    return torch.stack(losses), torch.stack(metrics), torch.stack(counts)
