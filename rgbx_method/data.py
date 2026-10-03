"""Balanced mixed-task batches and training-only contiguous short clips."""
import random
import numpy as np
import torch
from lib.train.base_functions import names2datasets, update_settings
from lib.train.data import LTRLoader, opencv_loader, processing, sampler
import lib.train.data.transforms as tfm


class MixedPairs(torch.utils.data.Dataset):
    def __init__(self, datasets, processor, samples):
        self.samples = samples
        self.samplers = [sampler.TrackingSampler([dataset], [1], samples, 200, 1,
                                                processing=processor) for dataset in datasets]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return self.samplers[index % 3][index // 3]


class MixedBatchSampler:
    def __init__(self, steps, per_rank, rank, world_size):
        self.steps, self.per_rank, self.rank, self.world_size = steps, per_rank, rank, world_size

    def __len__(self):
        return self.steps

    def __iter__(self):
        for step in range(self.steps):
            start = (step * self.world_size + self.rank) * self.per_rank
            yield list(range(start, start + self.per_rank))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_pair_loaders(cfg, settings, steps, rank, world_size, workers):
    update_settings(settings, cfg)
    settings.num_template, settings.num_search, settings.use_lmdb = 1, 1, False
    datasets = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader)
    assert [dataset.get_name() for dataset in datasets] == ['lasher', 'depthtrack', 'visevent']
    train_transform = tfm.Transform(tfm.ToTensorAndJitter(0.2), tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5))
    check_transform = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    processors = [processing.XTrackProcessing(search_area_factor=settings.search_area_factor, output_sz=settings.output_sz,
                                              center_jitter_factor=settings.center_jitter_factor,
                                              scale_jitter_factor=settings.scale_jitter_factor, mode='sequence',
                                              transform=train_transform, joint_transform=joint, settings=settings),
                  processing.XTrackProcessing(search_area_factor=settings.search_area_factor, output_sz=settings.output_sz,
                                              center_jitter_factor=settings.center_jitter_factor,
                                              scale_jitter_factor=settings.scale_jitter_factor, mode='sequence',
                                              transform=check_transform, settings=settings)]
    loaders = []
    for name, processor, batch in zip(['train', 'training_pool_check'], processors, [8, 3]):
        dataset = MixedPairs(datasets, processor, steps * batch * world_size)
        loader = LTRLoader(name, dataset, batch_sampler=MixedBatchSampler(steps, batch, rank, world_size),
                           num_workers=workers, stack_dim=1, worker_init_fn=seed_worker)
        loaders.append(loader)
    return datasets, loaders


class TrainingClips:
    def __init__(self, dataset, length):
        self.dataset, self.length = dataset, length
        self.eligible = []
        for sequence in range(dataset.get_num_sequences()):
            info = dataset.get_sequence_info(sequence)
            valid = info['visible'].bool() & torch.isfinite(info['bbox']).all(1) & (info['bbox'][:, 2:] > 0).all(1)
            starts = torch.nonzero(valid.unfold(0, length, 1).all(1), as_tuple=False).flatten()
            if len(starts):
                self.eligible.append((sequence, starts.tolist()))
        assert self.eligible

    def sample(self):
        sequence, starts = random.choice(self.eligible)
        start = random.choice(starts)
        frame_ids = list(range(start, start + self.length))
        info = self.dataset.get_sequence_info(sequence)
        frames, annotations, _ = self.dataset.get_frames(sequence, frame_ids, info)
        return frames, annotations['bbox'], dict(dataset=self.dataset.get_name(), sequence_id=sequence, frame_ids=frame_ids)
