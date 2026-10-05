"""Explicit evaluation parameters, avoiding the author's hardcoded workspace paths."""
import hashlib
import json
from pathlib import Path


def verified_endpoint(config_path, checkpoint_path, expected_sha256, terminal_audit_path):
    audit_path = Path(terminal_audit_path).resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'PASS'
    if audit['role'] == 'formal_terminal_checkpoint_audit_not_tracking_benchmark':
        assert audit['run_id'] == 'xtrack_b_adamw_3gpu_s2026_20261002'
        assert audit['epoch'] == 65 and audit['seed'] == 2026
    else:
        assert audit['role'] == 'joint_rgbx_checkpoint_audit_not_tracking_benchmark'
        assert audit['endpoint_kind'] in ['author_release', 'full_method']
        assert audit['strict_model_load'] and audit['all_model_tensors_finite']
        if audit['endpoint_kind'] == 'author_release':
            assert audit['checkpoint_sha256'] == '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
        else:
            assert audit['run_id'] in ['xtrack_full_method_v1_s2026', 'xtrack_full_method_v2_distance0_s2026']
            assert audit['epoch'] == 15 and audit['global_step'] == 37500
            assert audit['controller_exit_code'] == 0
            assert audit['frozen_tensors_unchanged'] and audit['replicas_identical']
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    assert checkpoint == Path(audit['checkpoint']).resolve(strict=True)
    if audit['role'] == 'joint_rgbx_checkpoint_audit_not_tracking_benchmark':
        assert config == Path(audit['config']).resolve(strict=True)
    assert expected_sha256 == audit['checkpoint_sha256']
    assert hashlib.sha256(config.read_bytes()).hexdigest() == audit['config_sha256']
    digest = hashlib.sha256()
    with checkpoint.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected_sha256
    return dict(checkpoint=str(checkpoint), config=str(config),
                checkpoint_sha256=expected_sha256, config_sha256=audit['config_sha256'],
                terminal_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())


def parameters(config_path, checkpoint_path, expected_sha256, terminal_audit_path):
    from lib.config.xtrack.config import cfg, update_config_from_file
    from lib.test.utils import TrackerParams

    identity = verified_endpoint(config_path, checkpoint_path, expected_sha256, terminal_audit_path)
    update_config_from_file(identity['config'])
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = identity['checkpoint']
    params.evaluation_identity = identity
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.save_all_boxes = False
    params.debug = 0
    return params
