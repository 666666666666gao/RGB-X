"""Separate localization objectives; preserve balance on the complete mixed batch."""
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap

TASK_NAMES = ['lasher', 'depthtrack', 'visevent']


def components(prediction, batch, cfg):
    boxes = prediction['pred_boxes']
    assert boxes.shape[1] == 1 and torch.isfinite(boxes).all()
    predicted = box_cxcywh_to_xyxy(boxes[:, 0])
    target = box_xywh_to_xyxy(batch['search_anno'][-1]).clamp(0, 1)
    giou, iou = generalized_box_iou(predicted, target)
    l1 = (predicted - target).abs().mean(1)
    heatmap = generate_heatmap(batch['search_anno'], cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.STRIDE)[-1].unsqueeze(1)
    labels = torch.zeros((len(boxes), 6), device=boxes.device)
    task_ids = torch.tensor([TASK_NAMES.index(name) for name in batch['dataset']], device=boxes.device)
    labels[task_ids == 0, :2] = 1
    labels[task_ids == 1, 4:] = 1
    labels[task_ids == 2, 2:4] = 1
    router = sum(F.binary_cross_entropy(probabilities, labels, reduction='none').mean(1)
                 for probabilities in prediction['logits_all'])
    objectives, counts, tasks = [], [], []
    focal = FocalLoss()
    for task in range(3):
        mask = task_ids == task
        assert mask.any()
        box = 2 * (1 - giou[mask]).mean() + 5 * l1[mask].mean()
        response = focal(prediction['score_map'][mask], heatmap[mask])
        objectives.append(torch.stack((response, box)))
        tasks.append(response + box + 0.01 * router[mask].mean() + prediction['loss_prompt_balance'])
        counts.append(mask.sum())
    indices = torch.stack([value.detach().topk(2, dim=1).indices.sort(1).values for value in prediction['logits_all']], dim=1)
    return torch.stack(tasks), torch.stack(objectives), torch.stack(counts), indices, dict(
        total=torch.stack(tasks).mean().detach(), balance=prediction['loss_prompt_balance'].detach(),
        classification=router.mean().detach(), iou=iou.mean().detach())
