"""Describe routing/risk signals in the immutable completed v1 training records."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parents[1]
TASKS = ['RGB-T', 'RGB-D', 'RGB-Event']
OBJECTIVES = ['response', 'box']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    existing_path = PROJECT/'reports/full_method_v1_full_run_cost_decomposition.json'
    existing = json.loads(existing_path.read_text())
    source = Path(existing['input']['path'])
    assert source.stat().st_size == existing['input']['bytes'] == 71091121
    assert sha(source) == existing['input']['sha256'] == '181ea2e5f7b93413a0cc51ac3a82180283a0ca3242d63f27462350499aad4173'
    output = PROJECT/'reports/full_method_v1_complete_route_risk_diagnosis_20261005.json'
    assert not output.exists()
    tasks = [dict(task=name,updates_with_any_localization_risk=0,updates_with_accepted_localization_risk=0,
        candidate_route_change_updates=0,accepted_route_change_updates=0,
        candidate_route_fraction_sum=0.,accepted_route_fraction_sum=0.,
        risk_and_route_change_updates=0,risk_without_route_change_updates=0,
        route_change_without_task_risk_updates=0,neither_task_risk_nor_route_change_updates=0,
        selected_nonzero_steps_with_task_risk=0,selected_nonzero_steps_with_task_route_change=0,
        objectives=[dict(objective=objective,candidate_risk_occurrences=0,accepted_risk_occurrences=0,
            risk_fully_eliminated_occurrences=0,positive_risk_reduced_but_not_eliminated_occurrences=0,
            newly_positive_after_selection=0,normalization_floor_activated=0,normalization_floor_activated_with_risk=0,
            before_values=[],before_values_when_risky=[]) for objective in OBJECTIVES]) for name in TASKS]
    count,risks,nonzero,any_route,accepted_any_route = 0,0,0,0,0
    selected = []
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            assert row['global_step'] == count and row['epoch'] == (count-1)//2500+1
            assert row['global_task_counts'] == [8,8,8]
            assert row['optimizer_statistics_updates'] == row['weight_decay_applications'] == 1
            candidate,accepted = row['candidate_risk'],row['accepted_risk']
            before = row['before_localization']
            candidate_routes,accepted_routes = row['candidate_route_change'],row['accepted_route_change']
            assert len(candidate) == len(accepted) == len(before) == len(candidate_routes) == len(accepted_routes) == 3
            has_risk = any(v>0 for task in candidate for v in task)
            is_nonzero = row['selected_correction_strength'] != 0
            risks += has_risk
            nonzero += is_nonzero
            any_route += any(v>0 for v in candidate_routes)
            accepted_any_route += any(v>0 for v in accepted_routes)
            for m,task in enumerate(tasks):
                cr,ar = candidate_routes[m],accepted_routes[m]
                assert math.isfinite(cr) and math.isfinite(ar) and 0 <= cr <= 1 and 0 <= ar <= 1
                task_risk = any(v>0 for v in candidate[m])
                changed = cr>0
                task['updates_with_any_localization_risk'] += task_risk
                task['updates_with_accepted_localization_risk'] += any(v>0 for v in accepted[m])
                task['candidate_route_change_updates'] += changed
                task['accepted_route_change_updates'] += ar>0
                task['candidate_route_fraction_sum'] += cr
                task['accepted_route_fraction_sum'] += ar
                task['risk_and_route_change_updates'] += task_risk and changed
                task['risk_without_route_change_updates'] += task_risk and not changed
                task['route_change_without_task_risk_updates'] += changed and not task_risk
                task['neither_task_risk_nor_route_change_updates'] += not task_risk and not changed
                task['selected_nonzero_steps_with_task_risk'] += is_nonzero and task_risk
                task['selected_nonzero_steps_with_task_route_change'] += is_nonzero and changed
                assert len(candidate[m]) == len(accepted[m]) == len(before[m]) == 2
                for k,objective in enumerate(task['objectives']):
                    c,a,value = candidate[m][k],accepted[m][k],before[m][k]
                    assert math.isfinite(c) and math.isfinite(a) and math.isfinite(value) and c>=0 and a>=0
                    objective['candidate_risk_occurrences'] += c>0
                    objective['accepted_risk_occurrences'] += a>0
                    objective['risk_fully_eliminated_occurrences'] += c>0 and a==0
                    objective['positive_risk_reduced_but_not_eliminated_occurrences'] += 0<a<c
                    objective['newly_positive_after_selection'] += c==0 and a>0
                    objective['normalization_floor_activated'] += abs(value)<1e-8
                    objective['normalization_floor_activated_with_risk'] += abs(value)<1e-8 and c>0
                    objective['before_values'].append(value)
                    if c>0:
                        objective['before_values_when_risky'].append(value)
            if is_nonzero:
                selected.append({key:row[key] for key in ['global_step','epoch','selected_correction_strength',
                    'candidate_risk','accepted_risk','candidate_route_change','accepted_route_change']})
    assert count == existing['totals']['updates'] == 37500
    assert risks == existing['totals']['risk_events'] == 968
    assert nonzero == existing['totals']['actual_nonzero_selections'] == 20
    assert [[o['candidate_risk_occurrences'] for o in t['objectives']] for t in tasks] == existing['candidate_risk_component_occurrences']
    assert [[o['accepted_risk_occurrences'] for o in t['objectives']] for t in tasks] == existing['accepted_risk_component_occurrences']
    for task in tasks:
        task['mean_candidate_route_change_fraction'] = task.pop('candidate_route_fraction_sum')/count
        task['mean_accepted_route_change_fraction'] = task.pop('accepted_route_fraction_sum')/count
        assert sum(task[key] for key in ['risk_and_route_change_updates','risk_without_route_change_updates',
            'route_change_without_task_risk_updates','neither_task_risk_nor_route_change_updates']) == count
        for objective in task['objectives']:
            values,risky = objective.pop('before_values'),objective.pop('before_values_when_risky')
            objective['before_loss_all_updates'] = dict(minimum=min(values),median=statistics.median(values),maximum=max(values))
            objective['before_loss_risky_updates'] = dict(minimum=min(risky),median=statistics.median(risky),maximum=max(risky))
    report = dict(status='ACTUAL_COMPLETED37500_ROUTING_AND_LOCALIZATION_RISK_DESCRIPTIVE_DIAGNOSIS',
        analyzed_at=datetime.now(timezone.utc).isoformat(),input=existing['input'],existing_analysis_sha256=sha(existing_path),
        script_sha256=sha(Path(__file__)),task_order=TASKS,objective_order=OBJECTIVES,updates=count,
        risk_events=risks,nonzero_selections=nonzero,updates_with_any_candidate_route_change=any_route,
        updates_with_any_accepted_route_change=accepted_any_route,task_results=tasks,all_twenty_nonzero_selection_records=selected,
        route_change_definition='Per task, rank-mean fraction of returned routing Top2 expert sets changed on the paired check batch; sets are sorted before comparison. Three augmented training-pool check pairs per task. Evaluation mode disables noise/dropout.',
        bounds='Top2-set stability does not imply constant gate weights, content, expert contributions or safe predictions. Co-occurrence is not causal attribution. Stored task-mean localization harms are not benchmark scores or per-sample harm labels. Floor activation is an observed arithmetic condition, not evidence that a replacement scale improves accuracy.',
        scope='Read the SHA-bound already completed training telemetry once. No model, optimizer, inference, scorer, current evaluation log/session/process, hardware query or weight action. Current live evaluation remains untouched; no efficacy, significance or full research completion claim.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['status','updates','risk_events','nonzero_selections',
        'updates_with_any_candidate_route_change','updates_with_any_accepted_route_change','task_results']}))


if __name__ == '__main__':
    main()
