"""Synthetic CPU identity and completion checks of held C/D execution entry points."""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rgbx_controls.execution import DISTANCE0_STAGES, verify_distance0_completion


def rejected(function):
    # Explicit negative contract checks, not production exception handling.
    try:
        function()
    except AssertionError:
        return True
    return False


def main():
    path = PROJECT / 'scripts/xtrack_eval_parameters_controls_prepared.py'
    spec = importlib.util.spec_from_file_location('prepared_control_parameters', str(path))
    prepared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepared)
    completed = dict(run_id='xtrack_full_method_v2_distance0_s2026', exit_code=0,
                     status='IMPROVEMENT_GOAL_UNMET_REQUIRES_DIAGNOSIS', completed_stages=DISTANCE0_STAGES[:])
    comparison = dict(status=completed['status'], required_count=7, datasets={'LasHeR':{},'VisEvent':{},'DepthTrack':{}})
    uncertainty = dict(status='PAIRED_SEQUENCE_BOOTSTRAP_COMPLETE_CONDITIONAL_ON_SAVED_SCORE_GRIDS', comparison_sha256='synthetic')
    verify_distance0_completion(completed, comparison, uncertainty, 'synthetic')
    unfinished = copy.deepcopy(completed)
    unfinished['completed_stages'].pop()
    assert rejected(lambda: verify_distance0_completion(unfinished, comparison, uncertainty, 'synthetic'))
    failed = copy.deepcopy(completed)
    failed['exit_code'] = 1
    assert rejected(lambda: verify_distance0_completion(failed, comparison, uncertainty, 'synthetic'))
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='rgbx-control-identity-') as temporary:
        root = Path(temporary)
        checkpoint = root / 'synthetic.bin'
        config = root / 'synthetic.yaml'
        checkpoint.write_bytes(b'synthetic identity bytes, NOT a torch/model checkpoint')
        config.write_text('synthetic: true\n')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for run_id, kind in [('xtrack_fixed_mixed_adamw_s2026','adamw'), ('xtrack_task_stats_only_s2026','task_adaptive_only')]:
            audit = dict(status='PASS', role='joint_rgbx_checkpoint_audit_not_tracking_benchmark', endpoint_kind='optimizer_control',
                         run_id=run_id, stage='endpoint', seed=2026, strict_model_load=True, all_model_tensors_finite=True,
                         epoch=15, global_step=37500, controller_exit_code=0, frozen_tensors_unchanged=True, replicas_identical=True,
                         optimizer_check=dict(optimizer_kind=kind,names_shapes_moments_steps_valid=True,always_used_witness_tensors=24),
                         saved_replica_hash_matches_loaded_parameters=True, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                         config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
            audit_path = root / 'synthetic_audit.json'
            audit_path.write_text(json.dumps(audit))
            identity = prepared.verified_endpoint(str(config), str(checkpoint), digest, str(audit_path))
            assert identity['checkpoint_sha256'] == digest
            wrong = copy.deepcopy(audit)
            wrong['optimizer_check']['optimizer_kind'] = 'task_adaptive_only' if kind == 'adamw' else 'adamw'
            audit_path.write_text(json.dumps(wrong))
            assert rejected(lambda: prepared.verified_endpoint(str(config), str(checkpoint), digest, str(audit_path)))
            wrong = copy.deepcopy(audit)
            wrong['stage'] = 'sanity'
            audit_path.write_text(json.dumps(wrong))
            assert rejected(lambda: prepared.verified_endpoint(str(config), str(checkpoint), digest, str(audit_path)))
            wrong = copy.deepcopy(audit)
            wrong['endpoint_kind'] = 'full_method'
            audit_path.write_text(json.dumps(wrong))
            assert rejected(lambda: prepared.verified_endpoint(str(config), str(checkpoint), digest, str(audit_path)))
            audit_path.write_text(json.dumps(audit))
            assert rejected(lambda: prepared.verified_endpoint(str(config), str(checkpoint), 'wrong_sha', str(audit_path)))
            outcomes.append(dict(run_id=run_id, synthetic_endpoint_admitted=True, wrong_optimizer_rejected=True,
                                 sanity_role_rejected=True, full_method_role_cannot_admit_control=True, wrong_digest_rejected=True))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='PASS_CPU_SYNTHETIC_CONTROL_IDENTITY_AND_STAGE_CONTRACTS_NOT_RUNTIME_OR_BENCHMARK',
                          controls=outcomes, incomplete_distance0_stages_rejected=True, failed_distance0_exit_rejected=True,
                          model_forward_calls=0,dataset_reads=0,generated_model_checkpoints=0,
                          live_entry_activated=False, torch_imported=False,
                          scope='Synthetic byte/JSON identity and completion-state contracts using the held prepared template. No real model checkpoint, actual stage completion, GPU/DDP job, source activation or tracking metric acceptance. Temporary fixtures are removed.')))


if __name__ == '__main__':
    main()
