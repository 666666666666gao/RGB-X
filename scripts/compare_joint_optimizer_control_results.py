"""Compare every required headline and retain all supplemental diagnostics."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def deltas(reference, method, keys):
    result = {}
    for key in keys:
        a, b = reference[key], method[key]
        assert math.isfinite(a) and math.isfinite(b)
        result[key] = dict(reference=a, method=b, delta=b - a, improved=b > a)
    return result


def per_sequence(reference, method, keys):
    left = {row['sequence']: row for row in reference}
    right = {row['sequence']: row for row in method}
    assert set(left) == set(right)
    return [dict(sequence=name, metrics=deltas(left[name], right[name], keys)) for name in left]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--training-updates', required=True)
    args = parser.parse_args()
    left, right = Path(args.reference), Path(args.method)
    assert left.name in ['xtrack_author_release_s2026', 'xtrack_fixed_mixed_adamw_s2026']
    assert right.name in ['xtrack_fixed_mixed_adamw_s2026', 'xtrack_task_stats_only_s2026']
    assert left.name != right.name
    for root in [left, right]:
        receipt = load(root / 'run.json')
        assert receipt['status'] == 'ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT'
        assert receipt['completed_datasets'] == ['LasHeR', 'VisEvent', 'DepthTrack']
    headlines, datasets, sources = {}, {}, []
    all_keys = ['SR_AUC', 'PR_at_20', 'NPR_AUC', 'NP_at_020']
    for dataset, sequences, frames in [('LasHeR', 245, 220703), ('VisEvent', 320, 157421)]:
        paths = [root / dataset / 'scores/metrics.json' for root in [left, right]]
        a, b = map(load, paths)
        for value in [a, b]:
            assert value['status'] == 'PINNED_TOOLKIT_SCORING_COMPLETE'
            assert value['sequence_count'] == sequences and value['frame_pairs'] == frames and value['missing_sequences'] == 0
            assert value['evaluation_type'] == 'real_gt'
        assert a['manifest_sha256'] == b['manifest_sha256'] and a['attributes_sha256'] == b['attributes_sha256']
        assert a['identity']['config_sha256'] == b['identity']['config_sha256']
        assert a['thresholds'] == b['thresholds'] and set(a['attributes']) == set(b['attributes'])
        scores = deltas(a['headline_metrics'], b['headline_metrics'], all_keys)
        headlines[dataset + '_SR'] = scores['SR_AUC']
        headlines[dataset + '_PR'] = scores['PR_at_20']
        attributes = {label: dict(toolkit=deltas(a['attributes'][label]['toolkit_plot_metrics'], b['attributes'][label]['toolkit_plot_metrics'], all_keys),
                                  all_sequences=deltas(a['attributes'][label]['all_sequence_metrics'], b['attributes'][label]['all_sequence_metrics'], all_keys))
                      for label in a['attributes']}
        datasets[dataset] = dict(all_headline_metrics=scores, attributes=attributes,
                                 all_sequence_aggregation=deltas(a['overall']['all_sequence_metrics'], b['overall']['all_sequence_metrics'], all_keys),
                                 per_sequence=per_sequence(a['per_sequence'], b['per_sequence'], all_keys),
                                 reference_fps=a['fps'], method_fps=b['fps'],
                                 reference_gpu_memory=a['gpu_memory'], method_gpu_memory=b['gpu_memory'],
                                 timing_scope=dict(reference=a['timing'], method=b['timing']),
                                 reference_curves=str(paths[0]), method_curves=str(paths[1]))
        sources.extend(paths)
    native_paths = [root / 'DepthTrack/official_metrics.json' for root in [left, right]]
    a, b = map(load, native_paths)
    for value in [a, b]:
        assert value['status'] == 'OFFICIAL_ANALYSES_COMPLETE' and value['evaluation_type'] == 'real_gt'
        assert len(value['sequence_order']) == 50
    assert a['sequence_order'] == b['sequence_order']
    assert a['identity']['dataset_audit_sha256'] == b['identity']['dataset_audit_sha256']
    assert a['identity']['config_sha256'] == b['identity']['config_sha256']
    depth_keys = ['F_score', 'Recall', 'Precision']
    depth_scores = deltas(a['headline_metrics'], b['headline_metrics'], depth_keys)
    headlines.update(DepthTrack_F=depth_scores['F_score'], DepthTrack_Recall=depth_scores['Recall'], DepthTrack_Precision=depth_scores['Precision'])
    diagnostics_paths = [root / 'DepthTrack/attribute_diagnostics.json' for root in [left, right]]
    da, db = map(load, diagnostics_paths)
    assert da['status'] == db['status'] == 'ATTRIBUTE_DIAGNOSTICS_COMPLETE'
    assert da['attributes_sha256'] == db['attributes_sha256']
    coverage = [load(root / 'DepthTrack/coverage.json') for root in [left, right]]
    for value in coverage:
        assert value['status'] == 'PASS' and value['sequences'] == 50 and value['frames'] == 76373
        assert value['missing_sequences'] == value['missing_trajectories'] == 0
    groups = {}
    for kind in ['sequence_groups', 'frame_groups']:
        assert set(da[kind]) == set(db[kind])
        groups[kind] = {label: dict(metrics=deltas(da[kind][label], db[kind][label], depth_keys),
                                   reference_context={key: value for key, value in da[kind][label].items() if key not in depth_keys and not key.endswith('curve')},
                                   method_context={key: value for key, value in db[kind][label].items() if key not in depth_keys and not key.endswith('curve')})
                        for label in da[kind]}
    datasets['DepthTrack'] = dict(all_headline_metrics=depth_scores, **groups,
                                per_sequence=per_sequence(da['per_sequence'], db['per_sequence'], depth_keys),
                                reference_fps=coverage[0]['fps'], method_fps=coverage[1]['fps'],
                                reference_gpu_memory=coverage[0]['gpu_memory'], method_gpu_memory=coverage[1]['gpu_memory'],
                                timing_scope='Native tracker RPC tracking seconds; differs from OPE synchronized image-read/preprocessing/track scope.',
                                per_sequence_threshold_indices=[dict(sequence=x['sequence'], reference_best_threshold_index=x['best_threshold_index'],
                                                                     method_best_threshold_index=y['best_threshold_index'])
                                                                for x, y in zip(da['per_sequence'], db['per_sequence'])],
                                reference_native_curves=str(native_paths[0]), method_native_curves=str(native_paths[1]),
                                threshold_note='Native confidence thresholds are endpoint-specific; curves remain with their own thresholds, not subtracted by array index.')
    sources.extend(native_paths + diagnostics_paths)
    assert len(headlines) == 7
    training_path, updates_path = Path(args.training_run), Path(args.training_updates)
    training = load(training_path)
    controls = {'xtrack_fixed_mixed_adamw_s2026': 'adamw', 'xtrack_task_stats_only_s2026': 'task_adaptive_only'}
    assert training['run_id'] == right.name and training['optimizer_kind'] == controls[right.name]
    assert all(training[key] == 0 for key in ['localization_check_calls', 'correction_calls', 'trajectory_calls', 'check_examples'])
    assert training['status'] == 'TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT' and training['global_step'] == 37500
    telemetry = [json.loads(line) for line in updates_path.read_text().splitlines()]
    boundaries = [row for row in telemetry if row.get('record_type') == 'RESUME_BOUNDARY']
    attempts = [row for row in telemetry if row.get('record_type') != 'RESUME_BOUNDARY']
    committed = {row['global_step']: row for row in attempts}
    assert set(committed) == set(range(1, 37501))
    training_cost = dict(training_receipt=str(training_path), update_telemetry=str(updates_path),
                         committed_updates=37500, committed_training_examples=900000,
                         attempted_updates=len(attempts), attempted_training_examples=24 * len(attempts),
                         attempted_update_seconds=sum(row['step_seconds'] for row in attempts),
                         attempted_pair_check_forward_examples=sum(row['pair_check_forward_examples'] for row in attempts),
                         attempted_trajectory_model_forward_examples=sum(row['trajectory_model_forward_examples'] for row in attempts),
                         max_rank0_allocated_bytes=max(row['peak_allocated_bytes'] for row in attempts),
                         last_attempt_reported_wallclock_seconds=training['wallclock_seconds'],
                         resume_boundaries=boundaries, world_size=3,
                         author_original_training_cost='NOT_MEASURED',
                         cost_note='Update seconds exclude model initialization and checkpoint saving. Receipt wallclock is the last attempt; prior attempts and all retained update costs are reported separately. Peak allocation is rank0 only, not all-rank maximum.')
    improved = all(value['improved'] for value in headlines.values())
    report = dict(status='ALL_REQUIRED_HEADLINES_IMPROVED' if improved else 'IMPROVEMENT_GOAL_UNMET_REQUIRES_DIAGNOSIS',
                  observed_at=datetime.now(timezone.utc).isoformat(), metric_units='fractions_0_to_1',
                  required_headlines=headlines, required_count=7, improved_count=sum(value['improved'] for value in headlines.values()),
                  regressions_or_ties=[key for key, value in headlines.items() if not value['improved']],
                  datasets=datasets, training_cost=training_cost, reference_root=str(left), method_root=str(right),
                  sources=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
                  comparison_setting=('Published joint XTrack-B vs matched repaired C/D optimizer-control fine-tuning; unequal total training budget.' if left.name == 'xtrack_author_release_s2026' else 'Repaired fixed-mixed AdamW vs existing task-statistics-only fine-tuning from the same public initialization and configured primary15-epoch budget; compute costs reported separately.'),
                  optimizer_control_run_id=right.name, optimizer_control_kind=training['optimizer_kind'], canonical_AdaTask_completed=False,
                  annotation_caveat='VisEvent existing author-archive332-frame version; same-protocol paired comparison, published-number equivalence unresolved.',
                  interpretation='One seed does not establish training-seed reliability. Author-reference improvement does not isolate repairs or extra training. C vs D is an existing task-statistics component comparison, not canonical AdaTask or evidence for all three proposed modules.')
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'required_headlines', 'improved_count', 'regressions_or_ties']}))


if __name__ == '__main__':
    main()
