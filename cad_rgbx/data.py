"""The established RGB-X readers, with an equal three-task batch on each rank."""
from lib.train.admin.settings import Settings
from lib.train.base_functions import names2datasets, update_settings
from lib.train.data import LTRLoader, opencv_loader, processing
import lib.train.data.transforms as tfm
from rgbx_method.data import MixedPairs, MixedBatchSampler, seed_worker


def build_loader(cfg, protocol, rank, steps, workers):
    settings = Settings()
    settings.local_rank = rank
    update_settings(settings, cfg)
    settings.num_template, settings.num_search, settings.use_lmdb = 1, 1, False
    datasets = names2datasets(protocol['train_datasets'], settings, opencv_loader)
    assert [value.get_name() for value in datasets] == ['lasher', 'depthtrack', 'visevent']
    assert [value.get_num_sequences() for value in datasets] == protocol['expected_train_sequences']
    transform = tfm.Transform(tfm.ToTensorAndJitter(0.2), tfm.RandomHorizontalFlip_Norm(probability=0.5),
                              tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(probability=0.5))
    processor = processing.XTrackProcessing(search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz, center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor, mode='sequence', transform=transform,
        joint_transform=joint, settings=settings)
    per_rank, world = protocol['batch_per_rank'], protocol['world_size']
    assert per_rank % 3 == 0 and world == 2
    dataset = MixedPairs(datasets, processor, steps * per_rank * world)
    return LTRLoader('cad_rgbx_train', dataset,
        batch_sampler=MixedBatchSampler(steps, per_rank, rank, world), num_workers=workers,
        stack_dim=1, worker_init_fn=seed_worker), datasets
