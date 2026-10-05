"""Exact completion and CPU-sanity contracts for the two C/D control jobs."""
import hashlib
import json
from pathlib import Path

RUNS = {'adamw': 'xtrack_fixed_mixed_adamw_s2026', 'task_adaptive_only': 'xtrack_task_stats_only_s2026'}
DISTANCE0_STAGES = ['train', 'method-endpoint-audit', 'method-evaluation', 'complete-comparison', 'paired-sequence-uncertainty']
RESULT_STATUSES = ['ALL_REQUIRED_HEADLINES_IMPROVED', 'IMPROVEMENT_GOAL_UNMET_REQUIRES_DIAGNOSIS']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_distance0_completion(controller, comparison, uncertainty, comparison_sha256):
    assert controller['run_id'] == 'xtrack_full_method_v2_distance0_s2026'
    assert controller['exit_code'] == 0 and controller['status'] == comparison['status']
    assert controller['completed_stages'] == DISTANCE0_STAGES
    assert comparison['status'] in RESULT_STATUSES and comparison['required_count'] == 7
    assert set(comparison['datasets']) == {'LasHeR', 'VisEvent', 'DepthTrack'}
    assert uncertainty['status'] == 'PAIRED_SEQUENCE_BOOTSTRAP_COMPLETE_CONDITIONAL_ON_SAVED_SCORE_GRIDS'
    assert uncertainty['comparison_sha256'] == comparison_sha256


def verify_completed_sanity(protocol, run, audit, execution):
    assert protocol['run_id'] == RUNS[protocol['optimizer']]
    assert run['run_id'] == audit['run_id'] == execution['run_id'] == protocol['run_id']
    assert run['protocol'] == protocol and run['status'] == 'SANITY_COMPLETE_NOT_FORMAL_RESULT'
    assert run['global_step'] == audit['global_step'] == 2 and run['completed_epochs'] == audit['epoch'] == 1
    assert audit['status'] == 'PASS_CONTROL_SANITY_CHECKPOINT_NOT_FORMAL_RESULT' and audit['stage'] == 'sanity'
    assert audit['strict_model_load'] and audit['frozen_tensors_unchanged'] and audit['replicas_identical']
    assert audit['optimizer_check']['names_shapes_moments_steps_valid'] and audit['saved_replica_hash_matches_loaded_parameters']
    assert execution['status'] == 'EXITED' and execution['exit_code'] == 0 and execution['stage'] == 'sanity'
    assert run['protocol_sha256'] == audit['protocol_sha256'] == execution['protocol_sha256']


def launch_inputs(project, kind):
    run_id = RUNS[kind]
    protocol_path = project / 'configs' / (run_id + '.json')
    protocol = json.loads(protocol_path.read_text())
    assert protocol['optimizer'] == kind and protocol['run_id'] == run_id
    assert protocol['output'] == '/home/gaob/rgbx-full-method/' + run_id
    preparation = json.loads((project / 'reports/joint_optimizer_controls_execution_preparation_20261005.json').read_text())
    review = json.loads((project / preparation['source_review_report']).read_text())
    assert review['status'] == 'SOURCE_REVIEW_PASS' and review['blocking_issues'] == []
    assert {name:sha(project / name) for name in preparation['source_sha256']} == preparation['source_sha256']
    # This admission replacement is permitted only after the distance0 stages really finish.
    comparison_path = project / 'reports/full_method_v2_distance0_complete_comparison.json'
    comparison = json.loads(comparison_path.read_text())
    uncertainty = json.loads((project / 'reports/full_method_v2_distance0_paired_sequence_uncertainty.json').read_text())
    controller = json.loads(Path('/home/gaob/rgbx-full-method/xtrack_full_method_v2_distance0_s2026_controller/run.json').read_text())
    verify_distance0_completion(controller, comparison, uncertainty, sha(comparison_path))
    assert not Path('/proc', str(controller['controller_pid'])).exists()
    assert sha(project / 'scripts/xtrack_eval_parameters.py') == preparation['prepared_entry_sha256']
    reference = Path('/home/gaob/rgbx-eval-results/xtrack_author_release_s2026')
    reference_run = json.loads((reference / 'run.json').read_text())
    assert reference_run['status'] == 'ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT'
    assert reference_run['completed_datasets'] == ['LasHeR', 'VisEvent', 'DepthTrack']
    assert reference_run['identity']['checkpoint_sha256'] == '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
    for item in comparison['sources']:
        assert sha(item['path']) == item['sha256']
    if kind == 'task_adaptive_only':
        adamw_root = Path('/home/gaob/rgbx-eval-results/xtrack_fixed_mixed_adamw_s2026')
        adamw_run = json.loads((adamw_root / 'run.json').read_text())
        assert adamw_run['status'] == 'ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT'
        assert adamw_run['completed_datasets'] == ['LasHeR', 'VisEvent', 'DepthTrack']
        prior = json.loads((project / 'reports/xtrack_fixed_mixed_adamw_s2026_author_comparison.json').read_text())
        assert prior['status'] in RESULT_STATUSES and prior['required_count'] == 7
        for item in prior['sources']:
            assert sha(item['path']) == item['sha256']
    return run_id, protocol_path, protocol
