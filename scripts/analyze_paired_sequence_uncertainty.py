"""Paired sequence bootstrap on completed scores, conditional on saved grids."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

OPE_KEYS = ['SR_AUC', 'PR_at_20', 'NPR_AUC', 'NP_at_020']
CURVES = ['success_curve', 'precision_curve', 'normalized_precision_curve', 'normalized_precision_curve']
DEPTH_KEYS = ['F_score', 'Recall', 'Precision']


def ope_scores(rows, counts):
    result = {}
    for key, curve in zip(OPE_KEYS, CURVES):
        selected = [(counts[row['sequence']], row[key]) for row in rows if sum(row[curve]) > sys.float_info.epsilon]
        denominator = sum(count for count, _ in selected)
        result[key] = sum(count * value for count, value in selected) / denominator
    return result


def depth_scores(rows, counts):
    denominator = sum(counts.values())
    length = len(rows[0]['precision_curve'])
    assert all(len(row['precision_curve']) == len(row['recall_curve']) == length for row in rows)
    precision = [sum(counts[row['sequence']] * row['precision_curve'][i] for row in rows) / denominator for i in range(length)]
    recall = [sum(counts[row['sequence']] * row['recall_curve'][i] for row in rows) / denominator for i in range(length)]
    fscore = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(precision, recall)]
    best = max(range(length), key=lambda index: fscore[index])
    return dict(F_score=fscore[best], Recall=recall[best], Precision=precision[best])


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_bootstrap(left, right, keys, score, draws, seed):
    names = [row['sequence'] for row in left]
    assert len(names) == len(set(names)) and set(names) == {row['sequence'] for row in right}
    assert draws >= 1000
    rng = random.Random(seed)
    observed_left = score(left, Counter(names))
    observed_right = score(right, Counter(names))
    deltas = {key: [] for key in keys}
    for _ in range(draws):
        counts = Counter(rng.choices(names, k=len(names)))
        a, b = score(left, counts), score(right, counts)
        for key in keys:
            difference = b[key] - a[key]
            assert math.isfinite(difference)
            deltas[key].append(difference)
    return {key: dict(reference=observed_left[key], method=observed_right[key], delta=observed_right[key] - observed_left[key],
                      percentile_95_interval=[percentile(deltas[key], .025), percentile(deltas[key], .975)],
                      bootstrap_fraction_delta_positive=sum(value > 0 for value in deltas[key]) / draws)
            for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    comparison_path = Path(args.comparison)
    comparison_raw = comparison_path.read_bytes()
    comparison = json.loads(comparison_raw)
    assert comparison['status'] in ['ALL_REQUIRED_HEADLINES_IMPROVED', 'IMPROVEMENT_GOAL_UNMET_REQUIRES_DIAGNOSIS']
    assert comparison['required_count'] == 7
    source_files = {row['path']: row['sha256'] for row in comparison['sources']}
    sources = []

    def load(path):
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert source_files[str(path)] == digest
        sources.append(dict(path=str(path), sha256=digest))
        return json.loads(raw)

    left_root, right_root = Path(comparison['reference_root']), Path(comparison['method_root'])
    identities = []
    for root in [left_root, right_root]:
        run = json.loads((root / 'run.json').read_text())
        assert run['status'] == 'ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT'
        assert run['completed_datasets'] == ['LasHeR', 'VisEvent', 'DepthTrack']
        identities.append(run['identity'])
    datasets, required = {}, {}
    for dataset, count, frames in [('LasHeR', 245, 220703), ('VisEvent', 320, 157421)]:
        left, right = [load(root / dataset / 'scores/metrics.json') for root in [left_root, right_root]]
        for endpoint, identity in zip([left, right], identities):
            assert endpoint['status'] == 'PINNED_TOOLKIT_SCORING_COMPLETE'
            assert endpoint['identity'] == identity and endpoint['evaluation_type'] == 'real_gt'
            assert endpoint['sequence_count'] == count and endpoint['frame_pairs'] == frames and endpoint['missing_sequences'] == 0
            assert len(endpoint['per_sequence']) == count
            point = ope_scores(endpoint['per_sequence'], Counter(row['sequence'] for row in endpoint['per_sequence']))
            assert all(abs(point[key] - endpoint['headline_metrics'][key]) < 1e-12 for key in OPE_KEYS)
        assert left['manifest_sha256'] == right['manifest_sha256'] and left['attributes_sha256'] == right['attributes_sha256']
        assert left['thresholds'] == right['thresholds']
        result = paired_bootstrap(left['per_sequence'], right['per_sequence'], OPE_KEYS, ope_scores, args.draws, args.seed)
        datasets[dataset] = dict(sequences=count, metrics=result, aggregation='Equal-sequence means with the pinned toolkit curve-specific nonzero masks.')
        required[dataset + '_SR'] = result['SR_AUC']
        required[dataset + '_PR'] = result['PR_at_20']
    left, right = [load(root / 'DepthTrack/attribute_diagnostics.json') for root in [left_root, right_root]]
    native_paths = [root / 'DepthTrack/official_metrics.json' for root in [left_root, right_root]]
    native_left, native_right = [load(path) for path in native_paths]
    assert left['sequence_order'] == right['sequence_order'] and len(left['sequence_order']) == 50
    assert left['attributes_sha256'] == right['attributes_sha256']
    assert native_left['identity']['dataset_audit_sha256'] == native_right['identity']['dataset_audit_sha256']
    for endpoint, native, native_path, identity in zip([left, right], [native_left, native_right], native_paths, identities):
        assert endpoint['status'] == 'ATTRIBUTE_DIAGNOSTICS_COMPLETE' and native['status'] == 'OFFICIAL_ANALYSES_COMPLETE'
        assert endpoint['official_metrics_sha256'] == source_files[str(native_path)]
        assert endpoint['identity'] == native['identity']
        assert native['evaluation_type'] == 'real_gt'
        assert all(native['identity'][key] == value for key, value in identity.items())
        assert len(endpoint['per_sequence']) == 50
        point = depth_scores(endpoint['per_sequence'], Counter(endpoint['sequence_order']))
        assert all(abs(point[key] - native['headline_metrics'][key]) < 1e-12 for key in DEPTH_KEYS)
    result = paired_bootstrap(left['per_sequence'], right['per_sequence'], DEPTH_KEYS, depth_scores, args.draws, args.seed)
    datasets['DepthTrack'] = dict(sequences=50, metrics=result,
        aggregation='Each endpoint separately averages resampled sequence PR curves, maximizes its own F curve, then reads P/R at that maximum.',
        confidence_threshold_scope='Fixed endpoint-specific grids from the complete original evaluations; thresholds are not regenerated from resampled raw confidence and are not matched by index across endpoints.')
    required.update(DepthTrack_F=result['F_score'], DepthTrack_Recall=result['Recall'], DepthTrack_Precision=result['Precision'])
    assert set(required) == set(comparison['required_headlines'])
    for key, value in required.items():
        assert all(abs(value[field] - comparison['required_headlines'][key][field]) < 1e-12 for field in ['reference', 'method', 'delta'])
    report = dict(status='PAIRED_SEQUENCE_BOOTSTRAP_COMPLETE_CONDITIONAL_ON_SAVED_SCORE_GRIDS',
        observed_at=datetime.now(timezone.utc).isoformat(), metric_units='fractions_0_to_1', draws=args.draws, bootstrap_seed=args.seed,
        sampling_unit='Whole sequence; same sampled multiplicities for both endpoints in each dataset.',
        datasets=datasets, required_headlines=required, comparison_sha256=hashlib.sha256(comparison_raw).hexdigest(), sources=sources,
        interpretation='Conditional sequence-sampling variability for one fixed pair of trained endpoints. This is not training-seed variance, a calibrated p-value, proof of module attribution or a multiple-comparison-corrected significance claim. No image, model or optimizer is executed; the original benchmark point estimates remain unchanged.')
    destination = Path(args.output)
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], required_headlines=required)))


if __name__ == '__main__':
    main()
