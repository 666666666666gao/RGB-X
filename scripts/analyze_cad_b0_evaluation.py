"""Recount the completed scratch B0 scores; compare XTrack as a different setup."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'reports/cad_rgbx_b0_s2026_evaluation'
REFERENCE = PROJECT/'reports/reference_xtrack_author_release_s2026'
INTAKE = PROJECT/'reports/cad_rgbx_evaluation_complete_intake_20261016.json'


def read(path):
    return json.loads(path.read_text())


def summarize(values):
    return dict(improved=sum(value > 0 for value in values), tied=sum(value == 0 for value in values),
                declined=sum(value < 0 for value in values),
                improved_at_least_10pp=sum(value >= 0.1 for value in values),
                declined_at_least_10pp=sum(value <= -0.1 for value in values))


def main():
    intake = read(INTAKE)
    assert intake['status'] == 'ALL_THREE_FULLY_SCORED_ENDPOINT_COLLECTED'
    assert intake['execution']['actual_exit_code'] == 0
    assert intake['full_sequences'] == 615 and intake['full_frames'] == 454497
    for name, digest in intake['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    datasets, comparisons = {}, []
    for dataset, count, frames in [('LasHeR', 245, 220703), ('VisEvent', 320, 157421)]:
        metrics = read(ROOT/dataset/'scores/metrics.json')
        reference = read(REFERENCE/dataset/'scores/metrics.json')
        coverage = read(ROOT/dataset/'coverage.json')
        assert metrics['sequence_count'] == count and metrics['frame_pairs'] == frames
        assert metrics['manifest_sha256'] == reference['manifest_sha256']
        assert len(metrics['per_sequence']) == count and coverage['missing_sequences'] == 0
        assert set(metrics['overall']['zero_curve_sequences'].values()) == {0}
        for key, size in [('success_curve', 21), ('precision_curve', 51), ('normalized_precision_curve', 51)]:
            curves = [row[key] for row in metrics['per_sequence']]
            assert all(len(curve) == size and all(math.isfinite(x) for x in curve) for curve in curves)
            calculated = [mean(column) for column in zip(*curves)]
            assert max(abs(a-b) for a,b in zip(calculated, metrics['overall'][key])) < 1e-12
        overall = metrics['overall']
        calculated = dict(SR_AUC=mean(overall['success_curve']), PR_at_20=overall['precision_curve'][20],
                          NPR_AUC=mean(overall['normalized_precision_curve']), NP_at_020=overall['normalized_precision_curve'][20])
        assert all(abs(value-metrics['headline_metrics'][key]) < 1e-12 for key,value in calculated.items())
        old = {row['sequence']: row for row in reference['per_sequence']}
        paired = []
        for row in metrics['per_sequence']:
            assert row['sequence'] in old
            paired.append(dict(sequence=row['sequence'], **{
                key: row[key]-old[row['sequence']][key] for key in calculated}))
        attributes = {name: dict(sequences=row['sequences'], metrics=row['toolkit_plot_metrics'])
                      for name,row in metrics['attributes'].items()}
        datasets[dataset] = dict(sequences=count,frames=frames,headline_metrics=metrics['headline_metrics'],
            zero_curve_sequences=overall['zero_curve_sequences'],attributes=attributes,
            curves_recomputed_from_all_sequences=True, descriptive_sequence_comparison_to_different_architecture={
                key:summarize([row[key] for row in paired]) for key in calculated},
            largest_SR_decreases=sorted(paired,key=lambda row:row['SR_AUC'])[:5],
            largest_SR_increases=sorted(paired,key=lambda row:row['SR_AUC'])[-5:][::-1],
            timing=dict(fps=metrics['fps'],summed_sequence_seconds=metrics['elapsed_seconds'],scope=metrics['timing']),
            gpu_memory=metrics['gpu_memory'],
            version_note=metrics.get('published_metric_comparability') if dataset=='VisEvent' else 'Frozen LasHeR manifest')
        for key,value in calculated.items():
            ref=reference['headline_metrics'][key]
            comparisons.append(dict(dataset=dataset,metric=key,required_primary=key in ['SR_AUC','PR_at_20'],
                scratch_CAD_B0_percent=value*100,XTrack_author_reference_percent=ref*100,
                difference_percentage_points=(value-ref)*100,relative_change_percent=(value/ref-1)*100,
                improved=value>ref,comparison_type='Different architecture and initialization; descriptive reference only'))
    metrics=read(ROOT/'DepthTrack/official_metrics.json')
    attributes=read(ROOT/'DepthTrack/attribute_diagnostics.json')
    reference=read(REFERENCE/'DepthTrack/official_metrics.json')
    coverage=read(ROOT/'DepthTrack/coverage.json')
    assert coverage['sequences']==coverage['trajectories']==50 and coverage['frames']==76373
    assert metrics['identity']['dataset_audit_sha256']==reference['identity']['dataset_audit_sha256']
    rows=attributes['per_sequence']
    assert len(rows)==50 and [row['sequence'] for row in rows]==attributes['sequence_order']
    precision=[mean(column) for column in zip(*(row['precision_curve'] for row in rows))]
    recall=[mean(column) for column in zip(*(row['recall_curve'] for row in rows))]
    fcurve=[2*p*r/(p+r) if p+r else 0 for p,r in zip(precision,recall)]
    best=max(range(len(fcurve)),key=fcurve.__getitem__)
    calculated=dict(Precision=precision[best],Recall=recall[best],F_score=fcurve[best])
    assert all(abs(value-metrics['headline_metrics'][key])<1e-12 for key,value in calculated.items())
    assert all(math.isfinite(value) for value in precision+recall+fcurve)
    datasets['DepthTrack']=dict(sequences=50,frames=76373,headline_metrics=metrics['headline_metrics'],
        official_F_recomputed_from_macro_PR=True,official_threshold_count=len(fcurve),best_threshold_index=best,
        sequence_groups={name:{key:value for key,value in row.items() if not key.endswith('_curve')}
                         for name,row in attributes['sequence_groups'].items()},
        frame_groups={name:{key:value for key,value in row.items() if not key.endswith('_curve')}
                      for name,row in attributes['frame_groups'].items()},
        attribute_rule_note=attributes['note'], timing=dict(fps=coverage['fps'],summed_rpc_seconds=coverage['rpc_tracking_seconds'],
        scope='Native TraX RPC time; different timing scope from OPE'),gpu_memory=coverage['gpu_memory'])
    for key,value in calculated.items():
        ref=reference['headline_metrics'][key]
        comparisons.append(dict(dataset='DepthTrack',metric=key,required_primary=True,
            scratch_CAD_B0_percent=value*100,XTrack_author_reference_percent=ref*100,
            difference_percentage_points=(value-ref)*100,relative_change_percent=(value/ref-1)*100,
            improved=value>ref,comparison_type='Different architecture and initialization; descriptive reference only'))
    report=dict(status='PASS_FULL_COVERAGE_AND_INDEPENDENT_CURVE_RECOUNT',run_id='cad_pruned_rgbx_b0_s2026',
        checkpoint_sha256=intake['run']['identity']['checkpoint_sha256'],datasets=datasets,comparison=comparisons,
        main_metric_count=7,main_metrics_above_XTrack_reference=sum(row['required_primary'] and row['improved'] for row in comparisons),
        seed=2026,training_seed_count=1,proposed_modules_trained=False,
        evaluation_wallclock_seconds=intake['run']['wallclock_seconds'],
        hypothesis='Random initialization, architecture and recipe differ from XTrack. Their separate effects have not been isolated.',
        next_action='No new training authorized by this evaluation task; retain complete B0 results as the scratch CAD baseline.',
        formal_trajectories_repeated=False,source_intake_sha256=hashlib.sha256(INTAKE.read_bytes()).hexdigest())
    output=PROJECT/'reports/cad_rgbx_b0_complete_analysis_20261016.json'
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    with (PROJECT/'reports/cad_rgbx_b0_all_metrics_20261016.csv').open('w',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(dict(status=report['status'],comparisons=comparisons,wallclock_seconds=report['evaluation_wallclock_seconds'])))


if __name__=='__main__':
    main()
