"""Three-frame acceptance using predicted boxes for subsequent search crops."""
import torch
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


class ClosedLoopProbe:
    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device
        self.preprocessor = PreprocessorMM()
        size = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.window = hann2d(torch.tensor([size, size]).long(), centered=True).to(device)

    @torch.no_grad()
    def evaluate(self, frames, boxes):
        cfg = self.cfg
        state = boxes[0].tolist()
        template, resize, _ = sample_target(frames[0], state, cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE)
        template = self.preprocessor.process(template)
        template_box = transform_image_to_crop(torch.tensor(state), torch.tensor(state), resize,
                                                torch.tensor([cfg.TEST.TEMPLATE_SIZE] * 2), normalize=True).to(self.device)
        mask = generate_mask_cond(cfg, 1, self.device, template_box.unsqueeze(0))
        values, outside_flags, predicted_boxes, crop_states = [], [], [], []
        for image, ground_truth in zip(frames[1:], boxes[1:]):
            previous = list(state)
            crop_states.append(previous)
            crop, resize, _ = sample_target(image, previous, cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE)
            prediction = self.model(template=template, search=self.preprocessor.process(crop), ce_template_mask=mask)
            target = transform_image_to_crop(ground_truth, torch.tensor(previous), resize,
                                              torch.tensor([cfg.TEST.SEARCH_SIZE] * 2), normalize=True).to(self.device)
            center = target[:2] + target[2:] / 2
            outside = ((center < 0) | (center > 1)).any().float()
            heatmap = generate_heatmap(target.view(1, 1, 4), cfg.TEST.SEARCH_SIZE,
                                       cfg.MODEL.BACKBONE.STRIDE)[-1].unsqueeze(1)
            response_loss = FocalLoss()(prediction['score_map'], heatmap)
            decoded = self.model.box_head.cal_bbox(self.window * prediction['score_map'], prediction['size_map'],
                                                   prediction['offset_map']).view(-1, 4).mean(0)
            cx, cy, width, height = (decoded * cfg.TEST.SEARCH_SIZE / resize).tolist()
            half = cfg.TEST.SEARCH_SIZE / resize / 2
            cx += previous[0] + previous[2] / 2 - half
            cy += previous[1] + previous[3] / 2 - half
            state = clip_box([cx - width / 2, cy - height / 2, width, height], image.shape[0], image.shape[1], margin=10)
            predicted_boxes.append(list(state))
            image_scale = torch.tensor([image.shape[1], image.shape[0]] * 2, device=self.device)
            global_predicted = box_xywh_to_xyxy(torch.tensor(state, device=self.device).view(1, 4)) / image_scale
            global_target = box_xywh_to_xyxy(ground_truth.to(self.device).view(1, 4)) / image_scale
            giou, _ = generalized_box_iou(global_predicted, global_target)
            box_loss = 2 * (1 - giou.mean()) + 5 * (global_predicted - global_target).abs().mean()
            values.append(response_loss + box_loss + outside)
            outside_flags.append(outside.item())
        assert crop_states[1] == predicted_boxes[0]
        score = torch.stack(values).mean()
        assert torch.isfinite(score)
        return score, dict(predicted_boxes=predicted_boxes, search_crop_boxes=crop_states,
                           target_outside_search=outside_flags, uses_gt_only_for_initialization=True)
