import datetime
import hashlib
import json
import math
import pathlib

root = pathlib.Path('C:/Users/gb/projects/rgbx-risk')
relative_evidence = 'reports/cad_rgbx_evaluation_sanity_evidence_r2_20261016'
evidence = root / relative_evidence
seal_path = root / 'reports/cad_rgbx_evaluation_preparation_r2_20261016.json'
seal = json.loads(seal_path.read_text(encoding='utf-8'))
execution = json.loads((evidence / 'execution_receipt.json').read_text())
collection = json.loads((evidence / 'collected_manifest.json').read_text())
preflight = json.loads((evidence / 'preflight.stdout.txt').read_text())
run_path = evidence / 'remote_run/run.json'
run = json.loads(run_path.read_text())
assert execution['actual_exit_code'] == run['actual_exit_code'] == 1
assert run['status'] == 'FAILED'
assert run['failed_command'] == 'DepthTrack-native-gpu0'
assert len(run['commands']) == 4
assert collection['source_sha256'] == preflight['source_sha256'] == seal['source_sha256']
assert hashlib.sha256(run_path.read_bytes()).hexdigest() == collection['files']['run.json']['sha256']

handoff_name = seal['sole_handoff']
(evidence / 'environment_spec.json').write_bytes((root / 'configs/cad_eval_environment_spec.json').read_bytes())
(evidence / 'preparation_seal.json').write_bytes(seal_path.read_bytes())

ope = []
for dataset in ['LasHeR', 'VisEvent']:
    directory = evidence / 'remote_run' / dataset / 'gpu0'
    receipt_path, = directory.glob('*.json')
    receipt = json.loads(receipt_path.read_text())
    predictions_path = directory / (receipt['sequence'] + '.txt')
    boxes = [[float(value) for value in row.split(',')] for row in predictions_path.read_text().splitlines()]
    scores = [float(value) for value in (directory / (receipt['sequence'] + '_confidence.txt')).read_text().splitlines()]
    assert len(boxes) == len(scores) == receipt['frame_count'] == receipt['sanity_frames'] == 3
    assert all(len(box) == 4 and all(math.isfinite(value) for value in box) and box[2] > 0 and box[3] > 0 for box in boxes)
    assert all(math.isfinite(value) for value in scores)
    assert receipt['checkpoint_sha256'] == run['identity']['checkpoint_sha256']
    assert receipt['config_sha256'] == run['identity']['config_sha256']
    assert receipt['terminal_audit_sha256'] == run['identity']['terminal_audit_sha256']
    assert receipt['predictions_sha256'] == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    assert receipt['model'] == 'CADTrackPruned' and receipt['physical_gpu'] == '0'
    assert receipt['closed_loop_search'] and receipt['peak_allocated_bytes'] > 0
    assert receipt['status'] == 'SANITY_ONLY_NOT_BENCHMARK'
    ope.append({
        'dataset': dataset,
        'sequence': receipt['sequence'],
        'physical_gpu': 0,
        'frame_count': 3,
        'finite_positive_geometry': True,
        'finite_confidences': True,
        'initialization_confidence_protocol_marker': scores[0],
        'actual_prediction_confidences': scores[1:],
        'predicted_boxes_xywh': boxes[1:],
        'closed_loop_search': True,
        'strict_checkpoint_load_completed': True,
        'strict_load_evidence': 'The executed sealed CADTrack constructor loads checkpoint net with strict=True before any predictions; both receipts bind the same verified endpoint.',
        'peak_allocated_bytes': receipt['peak_allocated_bytes'],
        'peak_reserved_bytes': receipt['peak_reserved_bytes'],
        'receipt': str(receipt_path.relative_to(root)).replace('\\', '/'),
    })

failure_log = (evidence / 'remote_run/logs/DepthTrack-native-gpu0.log').read_text()
assert "TypeError: must be real number, not str" in failure_log
assert "math.isfinite(state.properties['confidence'])" in failure_log
real_calls = []
for call in run['commands']:
    failed = call['label'] == run['failed_command']
    real_calls.append(dict(call, actual_exit_code=1 if failed else 0,
        exit_code_evidence='run.json actual_exit_code and failed_command' if failed else 'The sealed synchronous fail-fast subprocess.run wrapper proceeded to the next call, proving returncode 0.',
        log=f"{relative_evidence}/remote_run/logs/{call['label']}.log"))
not_run = [f'{dataset}-native-gpu1' for dataset in ['LasHeR', 'VisEvent', 'DepthTrack']]
config = json.loads((root / 'configs/cad_rgbx_b0_s2026.json').read_text())
assert hashlib.sha256((root / 'configs/cad_rgbx_b0_s2026.json').read_bytes()).hexdigest() == run['identity']['config_sha256']
assert [config['template_size'], config['search_size'], config['template_factor'], config['search_factor']] == [128, 256, 2.0, 4.0]

report = {
    'status': 'FAIL',
    'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'role': 'fresh_run_experiment_agent_follows_doc_validation',
    'review_independence': 'same-family',
    'acceptance_status': 'provisional',
    'requested_model': 'gpt-6-astra',
    'requested_reasoning_effort': 'max',
    'actual_model': 'UNATTESTED',
    'actual_reasoning_effort': 'UNATTESTED',
    'attribution_note': 'Runtime does not expose independently verifiable actual model/effort identity; no cross-family acceptance is claimed.',
    'actual_exit_code': execution['actual_exit_code'],
    'execution': execution,
    'documented_invocation_execution_count': 1,
    'sole_handoff': handoff_name,
    'handoff_scope': 'Only the current top section before the first newline-horizontal-rule separator was read.',
    'environment_ledger': handoff_name + ' current top section',
    'environment_spec': 'configs/cad_eval_environment_spec.json',
    'environment_versions': run['environment_versions'],
    'environment_version_check': 'PASS',
    'sanity_report': collection['root'] + '/run.json',
    'sanity_report_sha256': collection['files']['run.json']['sha256'],
    'sanity_status': run['status'],
    'source_sha256': seal['source_sha256'],
    'source_sha256_verification': {
        'local_before_execution': 'PASS_EXACT_12_OF_12',
        'remote_before_execution': 'PASS_EXACT_12_OF_12',
        'remote_after_execution': 'PASS_EXACT_12_OF_12',
        'local_at_report_write': 'NOT_RECHECKED_AFTER_PARENT_REPORTED_R3_EDIT',
    },
    'source_capture_boundary': 'The remote post-run collection matched all 12 R2 sealed files. After that capture, the parent reported changing check_cad_vot_runtime.py and the handoff for a distinct R3 attempt. This review evaluates only the original captured R2 invocation and evidence; it does not accept or execute R3.',
    'checkpoint_identity': run['identity'],
    'physical_gpus_authorized': [0, 1],
    'physical_gpus_with_actual_native_calls': [0],
    'preflight_gpu_memory': preflight['gpu_query_stdout'],
    'expected_native_calls': 6,
    'actual_native_calls_attempted': 3,
    'actual_native_calls_completed_successfully': 2,
    'real_calls': real_calls,
    'native_calls_not_run': not_run,
    'ope_runtime_checks': ope,
    'depthtrack_runtime': {
        'physical_gpu': 0,
        'status': 'FAIL_VERIFIER_PROPERTY_TYPE_AFTER_FIRST_NATIVE_UPDATE',
        'native_initialization_returned': True,
        'first_native_update_returned': True,
        'three_frame_native_report_written': False,
        'observed_blocker': "scripts/check_cad_vot_runtime.py:31 calls math.isfinite(state.properties['confidence']); the native TraX property is str, causing TypeError: must be real number, not str.",
        'confidence_numeric_value': None,
        'note': 'The failed checker did not serialize the returned prediction. Its later gpu_peak_allocated_bytes comparison was not reached and is not validated by this attempt.',
    },
    'geometry_source_checks': {
        'template_size': 128,
        'search_size': 256,
        'template_factor': 2.0,
        'search_factor': 4.0,
        'single_template': True,
        'hann_window_and_center_decode': True,
        'previous_predicted_box_drives_next_crop': True,
        'mapping_to_original_image': 'search_size / resize_factor then previous center and half-side offset, followed by clip_box(..., margin=10)',
        'confidence': 'The returned box-head score comes from the Hann-weighted score map; initialization marker 1.0 is excluded from actual prediction confidence checks.',
        'evidence': ['scripts/cad_eval_parameters.py', 'cad_rgbx/tracker.py', 'scripts/evaluate_xtrack_ope.py', 'scripts/evaluate_xtrack_vot.py'],
        'scope': 'Static checks bind the sealed evaluation source and hash-matched runtime config; actual three-frame runtime evidence is limited to GPU0 OPE.',
    },
    'divergences': [
        {
            'severity': 'blocking',
            'layer': 'exec_verifier',
            'document_claim': 'The invocation yields SANITY_PASS_NOT_BENCHMARK with six successful native child calls.',
            'reality': 'The invocation exited 1 at DepthTrack-native-gpu0 after two successful OPE calls; GPU1 calls were never reached.',
            'file': 'scripts/check_cad_vot_runtime.py',
            'line': 31,
            'exception': 'TypeError: must be real number, not str',
            'cause': "The checker passes the string-valued TraX confidence property directly to math.isfinite.",
        },
    ],
    'formal_admissible': False,
    'formal_launched': False,
    'training_launched': False,
    'optimizer_steps': 0,
    'benchmark_datasets_completed': 0,
    'weight_files_created': 0,
    'environment_install_or_rebuild': False,
    'source_or_gpu_settings_changed_by_reviewer': False,
    'retry_or_alternate_nn_command_executed': False,
    'earlier_attempt_evidence_modified': False,
    'evidence_directory': relative_evidence + '/',
    'collected_remote_artifact_count': len(collection['files']),
    'actual_stdout_bytes': (evidence / 'actual.stdout.txt').stat().st_size,
    'actual_stderr_bytes': (evidence / 'actual.stderr.txt').stat().st_size,
    'output_note': 'Controller stdout/stderr are empty because child stdout/stderr were redirected to the preserved per-call logs.',
    'server_report_path': '/data/gb/rgbx-risk/reports/cad_rgbx_evaluation_documented_review_r2_20261016.json',
    'server_evidence_path': '/data/gb/rgbx-risk/' + relative_evidence,
}
destination = root / 'reports/cad_rgbx_evaluation_documented_review_r2_20261016.json'
with destination.open('x', encoding='utf-8', newline='\n') as output:
    output.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({
    'status': report['status'],
    'report': str(destination),
    'report_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'sanity_report_sha256': report['sanity_report_sha256'],
    'actual_native_calls_attempted': 3,
    'actual_native_calls_completed_successfully': 2,
    'collected_remote_artifact_count': len(collection['files']),
}, indent=2))
