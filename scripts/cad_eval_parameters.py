"""Bind evaluation to the completed scratch CADTrack B0 endpoint."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def verified_endpoint(config_path, checkpoint_path, expected_sha256, terminal_audit_path):
    config = Path(config_path).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    audit_path = Path(terminal_audit_path).resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    protocol = json.loads(config.read_text())
    run = json.loads((checkpoint.parent / 'run.json').read_text())
    assert audit['status'] == 'PASS' and audit['role'] == 'CAD_CPU_SAVED_STATE_CHECK_NOT_BENCHMARK'
    assert audit['stage'] == 'formal' and audit['strict_model_load'] and audit['all_model_tensors_finite']
    assert audit['replica_state_hash_matches'] and audit['global_step'] == 250000
    assert protocol == run['protocol'] and protocol['run_id'] == 'cad_pruned_rgbx_b0_s2026'
    assert run['status'] == 'TRAINING_COMPLETE_NOT_EVALUATED'
    assert run['completed_epochs'] == 100 and run['global_step'] == 250000
    assert checkpoint == Path(audit['checkpoint']).resolve() == Path(protocol['output']) / 'CADTrack_latest.pth.tar'
    assert checkpoint.stat().st_size == audit['checkpoint_bytes']
    assert expected_sha256 == audit['checkpoint_sha256']
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected_sha256
    return dict(checkpoint=str(checkpoint), config=str(config), model='CADTrackPruned',
                checkpoint_sha256=expected_sha256, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                terminal_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())


def parameters(config_path, checkpoint_path, expected_sha256, terminal_audit_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cad_rgbx import bootstrap
    from cad_rgbx.model import configure

    identity = verified_endpoint(config_path, checkpoint_path, expected_sha256, terminal_audit_path)
    protocol = json.loads(Path(identity['config']).read_text())
    cfg = configure(protocol)
    cfg.TEST.TEMPLATE_SIZE = protocol['template_size']
    cfg.TEST.SEARCH_SIZE = protocol['search_size']
    return SimpleNamespace(cfg=cfg, protocol=protocol, checkpoint=identity['checkpoint'],
        evaluation_identity=identity, template_factor=protocol['template_factor'],
        template_size=protocol['template_size'], search_factor=protocol['search_factor'],
        search_size=protocol['search_size'])
