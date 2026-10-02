"""Seeded CUDA kernel witness for every GPU used by this run."""
import json

import torch

assert torch.cuda.device_count() == 3
for index in range(3):
    torch.cuda.set_device(index)
    torch.manual_seed(2026)
    matrix = torch.randn(16, 16, device='cuda')
    output = matrix @ matrix.T
    assert output.numel() == 256 and torch.isfinite(output).all()
    print('CUDA_WITNESS', json.dumps({'index': index, 'gpu': torch.cuda.get_device_name(index),
                                     'shape': list(output.shape), 'sum': float(output.sum()),
                                     'torch': torch.__version__, 'cuda': torch.version.cuda}))
