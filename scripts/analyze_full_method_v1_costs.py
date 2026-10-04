"""Analyze stored final-v1 trials without running a model or changing an update."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')
ROOT = Path('/home/gaob/rgbx-full-method/xtrack_full_method_v1_s2026')


def risk(before, after, tolerance):
    if isinstance(before[0], list):
        before = [value for task in before for value in task]
        after = [value for task in after for value in task]
    return sum(max((b - a) / max(abs(a), 1e-8) - tolerance, 0.)
               for a, b in zip(before, after)) / len(before)


def positive(values):
    return any(value > 0 for task in values for value in task)


def main():
    protocol = json.loads((PROJECT / 'configs/full_method_v1.json').read_text())
    training = json.loads((ROOT / 'run.json').read_text())
    assert training['status'] == 'TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT'
    assert training['global_step'] == 37500 and training['completed_epochs'] == 15
    assert training['protocol'] == protocol and protocol['candidate_distance_weight'] == .05
    output = PROJECT / 'reports/full_method_v1_full_run_cost_decomposition.json'
    assert not output.exists()
    epoch_stats = [dict(epoch=i, updates=0, risk_events=0, actual_nonzero_selections=0,
                        events_with_a_local_risk_reducing_candidate=0,
                        fixed_trial_nonzero_without_distance=0,
                        accepted_updates_with_localization_risk=0,
                        trajectory_checks=0, no_local_risk_periodic_checks=0,
                        trajectory_only_deteriorations=0, update_seconds=0.)
                   for i in range(1, 16)]
    candidate_components = [[0, 0] for _ in range(3)]
    accepted_components = [[0, 0] for _ in range(3)]
    events, trajectory_only = [], []
    raw_sha = hashlib.sha256()
    raw_bytes = 0
    pair_forwards = trajectory_forwards = 0
    accepted_risk_eliminated = newly_harmed_components = 0
    with (ROOT / 'updates.jsonl').open('rb') as stream:
        for index, line in enumerate(stream, 1):
            raw_sha.update(line)
            raw_bytes += len(line)
            row = json.loads(line)
            assert row['global_step'] == index and row['attempt_id'] == training['attempt_id']
            stat = epoch_stats[row['epoch'] - 1]
            stat['updates'] += 1
            stat['update_seconds'] += row['step_seconds']
            stat['actual_nonzero_selections'] += row['selected_correction_strength'] != 0
            stat['accepted_updates_with_localization_risk'] += positive(row['accepted_risk'])
            pair_forwards += row['pair_check_forward_examples']
            trajectory_forwards += row['trajectory_model_forward_examples']
            for task in range(3):
                for objective in range(2):
                    candidate_components[task][objective] += row['candidate_risk'][task][objective] > 0
                    accepted_components[task][objective] += row['accepted_risk'][task][objective] > 0
                    newly_harmed_components += row['candidate_risk'][task][objective] == 0 and row['accepted_risk'][task][objective] > 0
            has_risk = positive(row['candidate_risk'])
            assert has_risk == bool(row['active_risk_constraints'])
            if row['trajectory_checked']:
                stat['trajectory_checks'] += 1
                if not has_risk:
                    stat['no_local_risk_periodic_checks'] += 1
                    base = row['candidate_trials'][0]
                    assert base['alpha'] == 0 and len(row['candidate_trials']) == 1
                    value = risk(row['baseline_trajectory'], base['trajectory'], protocol['risk_relative_tolerance'])
                    if value > 0:
                        stat['trajectory_only_deteriorations'] += 1
                        trajectory_only.append(dict(global_step=index, epoch=row['epoch'],
                            baseline_trajectory=row['baseline_trajectory'], candidate_trajectory=base['trajectory'],
                            trajectory_risk_mean=value, candidate_strengths=[0.], selected_strength=row['selected_correction_strength']))
            if not has_risk:
                continue
            stat['risk_events'] += 1
            accepted_risk_eliminated += not positive(row['accepted_risk'])
            trials = []
            for trial in row['candidate_trials']:
                localization = risk(row['before_localization'], trial['localization'], protocol['risk_relative_tolerance'])
                trajectory = risk(row['baseline_trajectory'], trial['trajectory'], protocol['risk_relative_tolerance'])
                cost_without_distance = localization + protocol['trajectory_risk_weight'] * trajectory
                trials.append(dict(alpha=trial['alpha'], localization_risk_mean=localization,
                    trajectory_risk_mean=trajectory, total_recorded_cost=trial['cost'],
                    distance_penalty_recovered=trial['cost'] - cost_without_distance,
                    cost_without_distance=cost_without_distance))
            base = trials[0]
            assert base['alpha'] == 0
            assert min(trials, key=lambda trial: trial['total_recorded_cost'])['alpha'] == row['selected_correction_strength']
            locally_improving = any(trial['localization_risk_mean'] < base['localization_risk_mean'] for trial in trials[1:])
            alternative = min(trials, key=lambda trial: trial['cost_without_distance'])
            stat['events_with_a_local_risk_reducing_candidate'] += locally_improving
            stat['fixed_trial_nonzero_without_distance'] += alternative['alpha'] != 0
            events.append(dict(global_step=index, epoch=row['epoch'], constraints=row['active_risk_constraints'],
                actual_selected_alpha=row['selected_correction_strength'], candidate_risk=row['candidate_risk'], accepted_risk=row['accepted_risk'],
                any_correction_reduces_local_risk=locally_improving,
                fixed_trial_alpha_without_distance=alternative['alpha'], trials=trials))
    assert index == 37500 and all(stat['updates'] == 2500 for stat in epoch_stats)
    totals = {key: sum(stat[key] for stat in epoch_stats) for key in epoch_stats[0] if key != 'epoch'}
    assert totals['risk_events'] == 968 and totals['actual_nonzero_selections'] == 20 and totals['trajectory_checks'] == 1700
    report = dict(status='COMPLETED37500_FIXED_TRIAL_COST_ANALYSIS_NOT_NEW_TRAINING_OR_BENCHMARK',
        observed_at=datetime.now(timezone.utc).isoformat(), run_id=protocol['run_id'],
        input=dict(path=str(ROOT / 'updates.jsonl'), bytes=raw_bytes, sha256=raw_sha.hexdigest(), committed_updates=37500),
        task_order=['RGB-T', 'RGB-D', 'RGB-Event'], objective_order=['response', 'box'],
        protocol_sha256=hashlib.sha256((PROJECT / 'configs/full_method_v1.json').read_bytes()).hexdigest(),
        candidate_distance_weight=protocol['candidate_distance_weight'], epoch_statistics=epoch_stats, totals=totals,
        candidate_risk_component_occurrences=candidate_components, accepted_risk_component_occurrences=accepted_components,
        risky_steps_with_localization_risk_eliminated=accepted_risk_eliminated,
        newly_harmed_localization_components_after_selection=newly_harmed_components,
        max_alpha0_distance_roundoff=max(abs(event['trials'][0]['distance_penalty_recovered']) for event in events),
        trajectory_only_deteriorations=trajectory_only, risk_events=events,
        extra_forward_accounting=dict(primary_training_examples=900000, pair_check_forward_examples=pair_forwards,
            trajectory_model_forward_examples=trajectory_forwards,
            scope='Counts from the original telemetry. Training/check examples are template-search pairs; trajectory counts model forward calls. Not distinct images, unique dataset exposure or FLOPs.'),
        scope='Analyze the immutable completed training log once, without reading live evaluation progress. Recompute risk terms from stored float32 objectives with Python arithmetic, using the same formula as the preserved589-update diagnostic. Distance is recovered as recorded cost minus localization/trajectory terms; alpha0 residual measures arithmetic roundoff.',
        limitation='Distance0 selections are offline counterfactuals over already evaluated trials. Different accepted updates would change all later states. These counts do not establish benchmark gain, independent trajectory control, all-task non-degradation or learning-rate causality. No model forward, gradient, optimizer or training modification.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'observed_at', 'input', 'totals',
        'candidate_risk_component_occurrences', 'accepted_risk_component_occurrences',
        'risky_steps_with_localization_risk_eliminated', 'newly_harmed_localization_components_after_selection',
        'max_alpha0_distance_roundoff', 'extra_forward_accounting']}))


if __name__ == '__main__':
    main()
