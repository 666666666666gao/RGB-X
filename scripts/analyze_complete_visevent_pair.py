"""Describe all saved VisEvent sequence/attribute deltas for the fixed v1 pair."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ['SR_AUC', 'PR_at_20', 'NPR_AUC', 'NP_at_020']
CURVES = ['success_curve', 'precision_curve', 'normalized_precision_curve']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    a, b = math.floor(position), math.ceil(position)
    return values[a] + (values[b] - values[a]) * (position - a)


def main():
    left_root = PROJECT/'reports/reference_xtrack_author_release_s2026/VisEvent'
    right_root = PROJECT/'reports/method_xtrack_full_method_v1_s2026/VisEvent'
    left_intake = read(PROJECT/'reports/reference_visevent_complete_intake_20261005.json')
    right_intake = read(PROJECT/'reports/full_method_v1_visevent_complete_intake_20261005.json')
    for root, files in [(left_root,left_intake['files']), (right_root,right_intake['files'])]:
        for name, saved in files.items():
            assert sha(root/name) == saved['sha256']
            assert (root/name).stat().st_size == saved['bytes']
    left_path, right_path = left_root/'scores/metrics.json', right_root/'scores/metrics.json'
    left, right = read(left_path), read(right_path)
    attributes_path = PROJECT/'reports/visevent_test_attributes.json'
    attributes = read(attributes_path)
    assert left['identity']['checkpoint_sha256'] == '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
    assert right['identity']['checkpoint_sha256'] == 'c9a490d4bac93e83e3697abd8fa24eb280cbfcbead226dc4a1629fa2d422fdc8'
    for endpoint in [left,right]:
        assert endpoint['status'] == 'PINNED_TOOLKIT_SCORING_COMPLETE'
        assert endpoint['evaluation_type'] == 'real_gt' and endpoint['dataset'] == 'VisEvent'
        assert endpoint['sequence_count'] == 320 and endpoint['frame_pairs'] == 157421
        assert endpoint['missing_sequences'] == 0 and len(endpoint['per_sequence']) == 320
        assert endpoint['attributes_sha256'] == sha(attributes_path)
    for key in ['manifest_sha256','toolkit_source_sha256','thresholds','attributes_sha256']:
        assert left[key] == right[key]
    rows = []
    a = {row['sequence']:row for row in left['per_sequence']}
    b = {row['sequence']:row for row in right['per_sequence']}
    assert len(a) == len(b) == 320 and set(a) == set(b) == set(attributes['attributes'])
    assert sum(row['frames'] for row in a.values()) == 157421
    for name, old in a.items():
        new = b[name]
        assert old['frames'] == new['frames']
        assert all(sum(old[curve]) > 0 and sum(new[curve]) > 0 for curve in CURVES)
        row = dict(sequence=name,frames=old['frames'])
        for key in KEYS:
            assert math.isfinite(old[key]) and math.isfinite(new[key])
            row.update({key+'_reference':old[key],key+'_method':new[key],key+'_delta':new[key]-old[key]})
        rows.append(row)
    headline, distributions = {}, {}
    for key in KEYS:
        values = [row[key+'_delta'] for row in rows]
        assert abs(statistics.mean(a[n][key] for n in a)-left['headline_metrics'][key]) < 1e-12
        assert abs(statistics.mean(b[n][key] for n in b)-right['headline_metrics'][key]) < 1e-12
        delta = right['headline_metrics'][key]-left['headline_metrics'][key]
        assert abs(statistics.mean(values)-delta) < 1e-12
        headline[key] = dict(reference=left['headline_metrics'][key],method=right['headline_metrics'][key],delta=delta)
        ordered = sorted(rows,key=lambda row:row[key+'_delta'])
        distributions[key] = dict(positive=sum(v>0 for v in values),zero=sum(v==0 for v in values),negative=sum(v<0 for v in values),
            median_delta=statistics.median(values),p10_delta=percentile(values,.1),p25_delta=percentile(values,.25),
            p75_delta=percentile(values,.75),p90_delta=percentile(values,.9),minimum_delta=min(values),maximum_delta=max(values),
            improve_at_least_10_percentage_points=sum(v>=.1 for v in values),
            degrade_at_least_10_percentage_points=sum(v<=-.1 for v in values),
            sum_positive_sequence_deltas=sum(v for v in values if v>0),sum_negative_sequence_deltas=sum(v for v in values if v<0),
            worst_ten_sequences=[dict(sequence=r['sequence'],frames=r['frames'],reference=r[key+'_reference'],method=r[key+'_method'],delta=r[key+'_delta']) for r in ordered[:10]],
            best_ten_sequences=[dict(sequence=r['sequence'],frames=r['frames'],reference=r[key+'_reference'],method=r[key+'_method'],delta=r[key+'_delta']) for r in reversed(ordered[-10:])])
    labels = attributes['labels']
    assert len(labels) == 17 and set(labels) == set(left['attributes']) == set(right['attributes'])
    attribute_rows = []
    for label in labels:
        members = [n for n in a if attributes['attributes'][n][label] == 1]
        old, new = left['attributes'][label], right['attributes'][label]
        assert len(members) == old['sequences'] == new['sequences']
        assert all(v == 0 for endpoint in [old,new] for v in endpoint['zero_curve_sequences'].values())
        row = dict(attribute=label,sequences=len(members))
        for key in KEYS:
            assert abs(statistics.mean(a[n][key] for n in members)-old['toolkit_plot_metrics'][key]) < 1e-12
            assert abs(statistics.mean(b[n][key] for n in members)-new['toolkit_plot_metrics'][key]) < 1e-12
            row.update({key+'_reference':old['toolkit_plot_metrics'][key],key+'_method':new['toolkit_plot_metrics'][key],
                key+'_delta':new['toolkit_plot_metrics'][key]-old['toolkit_plot_metrics'][key]})
        attribute_rows.append(row)
    joint = {}
    for row in rows:
        signs = tuple('positive' if row[k+'_delta']>0 else 'negative' if row[k+'_delta']<0 else 'zero' for k in KEYS[:2])
        name = 'SR_'+signs[0]+'_PR_'+signs[1]
        joint[name] = joint.get(name,0)+1
    report_path = PROJECT/'reports/full_method_v1_visevent_paired_diagnosis_20261005.json'
    sequences_path = PROJECT/'reports/full_method_v1_visevent_paired_per_sequence_20261005.csv'
    attributes_csv = PROJECT/'reports/full_method_v1_visevent_paired_attributes_20261005.csv'
    assert not report_path.exists() and not sequences_path.exists() and not attributes_csv.exists()
    for path, data in [(sequences_path,rows),(attributes_csv,attribute_rows)]:
        with path.open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    report = dict(status='COMPLETE_SAVED_VISEVENT_PAIR_DESCRIPTIVE_DIAGNOSIS_NOT_ALL3_COMPARISON',
        analyzed_at=datetime.now(timezone.utc).isoformat(),sequence_count=320,frame_pairs=157421,attributes=17,
        identities=dict(reference=left['identity'],method=right['identity']),headline_metrics=headline,
        sequence_delta_distributions=distributions,joint_SR_PR_sign_counts=joint,attribute_metrics=attribute_rows,
        metric_units='Fractions0-1; multiply deltas by100 for percentage points. Every sequence has equal weight in the official overall point estimates.',
        source_sha256={str(path.relative_to(PROJECT)).replace('\\','/'):sha(path) for path in [left_path,right_path,attributes_path,
            PROJECT/'reports/reference_visevent_complete_intake_20261005.json',PROJECT/'reports/full_method_v1_visevent_complete_intake_20261005.json']},
        output_sha256={str(path.relative_to(PROJECT)).replace('\\','/'):sha(path) for path in [sequences_path,attributes_csv]},
        script_sha256=sha(Path(__file__)),official_zero_curve_masks_affect_this_pair=False,
        statistical_or_causal_attribution_established=False,full_research_goal_achieved=False,
        interpretation='Complete overall scores can combine substantial paired sequence gains and harms. Attribute memberships overlap and sign counts alone do not determine mean gains. Descriptive saved-score evidence only; no training-seed, causality or significance claim. Retain all320 sequence and17 attribute results, not only examples.',
        scope='CPU arithmetic on complete already scored VisEvent artifacts only; raw hashes, endpoint identities, all sequence names/frames and every attribute mean verified. No live log/session/remote progress query, model/inference/scorer execution, training intervention, weight action or hardware control. Remaining DepthTrack dataset and queued all-seven conditional uncertainty still required. Test results are development diagnostics, never training-pool risk samples.')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=report['status'],headline_metrics=headline,
        distributions={k:{n:v for n,v in d.items() if not isinstance(v,list)} for k,d in distributions.items()},
        joint_SR_PR_sign_counts=joint,report_sha256=sha(report_path),output_sha256=report['output_sha256'])))


if __name__ == '__main__':
    main()
