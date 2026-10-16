"""Static-template CADTrack inference with the trained 128/256 geometry.

Crop, Hann window, CENTER decoding and margin10 mapping follow the supplied
pruned CADTrack tracker. Export its Hann-response maximum as confidence even
with one template (the supplied tracker otherwise returns None in that case).
"""
import math
import os
import subprocess
import sys
import time

import torch
from cad_rgbx import bootstrap
from cad_rgbx.model import build_random_model
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class CADTrack:
    def __init__(self, params):
        self.params = params
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        checkpoint = torch.load(params.checkpoint, map_location='cpu')
        assert checkpoint['model_type'] == 'CADTrackPruned'
        assert checkpoint['protocol'] == params.protocol and checkpoint['global_step'] == 250000
        self.network = build_random_model(params.cfg)
        self.network.load_state_dict(checkpoint['net'], strict=True)
        del checkpoint
        self.network = self.network.cuda().eval()
        self.mean = torch.tensor(params.cfg.DATA.MEAN * 2, device='cuda').view(1, 6, 1, 1)
        self.std = torch.tensor(params.cfg.DATA.STD * 2, device='cuda').view(1, 6, 1, 1)
        size = params.search_size // params.cfg.MODEL.BACKBONE.STRIDE
        window = 0.5 * (1 - torch.cos(2 * math.pi / (size + 1) * torch.arange(1, size + 1).float()))
        self.output_window = (window.reshape(1, 1, -1, 1) * window.reshape(1, 1, 1, -1)).cuda()
        self.next_temperature_check = 0.0
        self.cooling_seconds = 0.0
        self.maximum_sampled_gpu0_temperature_c = None

    def cool(self):
        if os.environ['CUDA_VISIBLE_DEVICES'] != '0' or time.perf_counter() < self.next_temperature_check:
            return
        protocol = self.params.protocol
        started = time.perf_counter()
        temperature = int(subprocess.check_output(['nvidia-smi', '-i', '0',
            '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip())
        previous = self.maximum_sampled_gpu0_temperature_c
        self.maximum_sampled_gpu0_temperature_c = temperature if previous is None else max(previous, temperature)
        if temperature >= protocol['gpu0_pause_temperature_c']:
            torch.cuda.synchronize()
            print('GPU0_EVAL_COOLING_PAUSE temperature_c=' + str(temperature), file=sys.stderr, flush=True)
            while temperature > protocol['gpu0_resume_temperature_c']:
                time.sleep(protocol['thermal_cooldown_check_seconds'])
                temperature = int(subprocess.check_output(['nvidia-smi', '-i', '0',
                    '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip())
                self.maximum_sampled_gpu0_temperature_c = max(self.maximum_sampled_gpu0_temperature_c, temperature)
            self.cooling_seconds += time.perf_counter() - started
        self.next_temperature_check = time.perf_counter() + protocol['thermal_cooldown_check_seconds']

    def preprocess(self, patch):
        tensor = torch.from_numpy(patch).cuda().float().permute(2, 0, 1).unsqueeze(0)
        return (tensor / 255.0 - self.mean) / self.std

    @torch.no_grad()
    def initialize(self, image, info):
        self.cool()
        patch, _, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                     output_sz=self.params.template_size)
        self.template = [self.preprocess(patch)]
        self.state = list(info['init_bbox'])

    @torch.no_grad()
    def track(self, image):
        self.cool()
        height, width, _ = image.shape
        patch, resize_factor, _ = sample_target(image, self.state, self.params.search_factor,
                                                output_sz=self.params.search_size)
        prediction, = self.network(template=self.template, search=[self.preprocess(patch)])
        response = self.output_window * prediction['score_map']
        boxes, score = self.network.box_head.cal_bbox(response, prediction['size_map'],
                                                     prediction['offset_map'], return_score=True)
        cx, cy, w, h = (boxes.reshape(-1, 4).mean(0) * self.params.search_size / resize_factor).tolist()
        cx_previous = self.state[0] + 0.5 * self.state[2]
        cy_previous = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * self.params.search_size / resize_factor
        box = [cx + cx_previous - half_side - 0.5 * w,
               cy + cy_previous - half_side - 0.5 * h, w, h]
        self.state = clip_box(box, height, width, margin=10)
        return dict(target_bbox=self.state, best_score=float(score[0, 0]))
