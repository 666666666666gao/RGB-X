"""Score all 320 paired VisEvent trajectories using the unchanged author's scorer."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from xtrack_eval_parameters import verified_endpoint

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--attributes', required=True)
parser.add_argument('--predictions', required=True)
parser.add_argument('--coverage', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--expected-sha256', required=True)
parser.add_argument('--terminal-audit', required=True)
parser.add_argument('--toolkit', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
identity = verified_endpoint(args.config,args.checkpoint,args.expected_sha256,args.terminal_audit)
manifest_path = Path(args.manifest).resolve(strict=True)
manifest = json.loads(manifest_path.read_text())
coverage = json.loads(Path(args.coverage).read_text())
attributes = json.loads(Path(args.attributes).read_text())
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert manifest['status']=='READY' and manifest['dataset']=='VisEvent'
assert coverage['status']=='PASS' and coverage['role']=='full_trajectory_coverage_not_metric_scoring'
assert coverage['sequences']==320 and coverage['frames']==157421 and coverage['missing_sequences']==0
assert coverage['manifest_sha256']==attributes['manifest_sha256']==manifest_sha
assert coverage['checkpoint_sha256']==identity['checkpoint_sha256']
assert coverage['config_sha256']==identity['config_sha256']
assert coverage['terminal_audit_sha256']==identity['terminal_audit_sha256']
rows = manifest['sequences']
names = [row['name'] for row in rows]
assert set(names)==set(attributes['attributes'])
predictions = Path(args.predictions).resolve(strict=True)
for receipt in coverage['per_sequence']:
    assert hashlib.sha256((predictions/(receipt['sequence']+'.txt')).read_bytes()).hexdigest()==receipt['predictions_sha256']
root = Path(args.output).resolve()
assert not root.exists()
root.mkdir(parents=True)
gt_dir = root/'annos'
(gt_dir/'gt_rect').mkdir(parents=True)
(gt_dir/'absent').mkdir()
for row in rows:
    gt = Path(row['groundtruth'])
    assert hashlib.sha256(gt.read_bytes()).hexdigest()==row['groundtruth_sha256']
    (gt_dir/'gt_rect'/(row['name']+'.txt')).symlink_to(gt)
    absence = Path(row['absence'])
    assert hashlib.sha256(absence.read_bytes()).hexdigest()==row['absence_sha256']
    flags = np.loadtxt(absence,ndmin=1)
    assert flags.shape==(row['frame_count'],) and set(flags)<={0,1}
    (gt_dir/'absent'/(row['name']+'.txt')).symlink_to(absence)
tracker_name = 'xtrack_joint_ep65_s2026'
(root/(tracker_name+'_tracking_result')).symlink_to(predictions,target_is_directory=True)
(root/'sequences.txt').write_text('\n'.join(names)+'\n')
toolkit = Path(args.toolkit).resolve(strict=True)
expected_sources = {
    'calc_seq_err_robust.m':'c851d74d713ace7ee71c12d25fafea9d1b1f9b83ef59b007ccad604054bf10eb',
    'calc_rect_int.m':'7ab693f66c74e066b53e538604f5af60f0d3da33b3f5711d8c96287b56afce62',
    'eval_tracker.m':'3064b18861d7e2cd9793d520dde1e732f362a5daba1eeca470f16aaeb42998ca',
}
source_hashes = {}
for name in expected_sources:
    source_hashes[name] = hashlib.sha256((toolkit/'utils'/name).read_bytes()).hexdigest()
    assert source_hashes[name] == expected_sources[name]
scripts_dir = Path(__file__).parent.resolve()
arguments = [str(toolkit),str(root/'sequences.txt'),str(gt_dir),str(root),tracker_name,str(root/'mat')]
statement = "addpath('{}'); score_visevent_matlab({});".format(
    str(scripts_dir).replace("'","''"), ','.join("'{}'".format(value.replace("'","''")) for value in arguments))
environment = dict(os.environ, OCTAVE_HOME='/home/gaob/conda/envs/rgbx-matlab-scorer')
with (root/'scorer.log').open('w') as log:
    subprocess.run(['/home/gaob/conda/envs/rgbx-matlab-scorer/bin/octave','--no-gui','--quiet','--eval',statement],
                   env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
sr = loadmat(root/'mat/raw/aveSuccessRatePlot_1alg_overlap_OPE.mat')['ave_success_rate_plot'][0]
pr = loadmat(root/'mat/raw/aveSuccessRatePlot_1alg_error_OPE.mat')['ave_success_rate_plot'][0]
np_curve = loadmat(root/'mat/normalized/aveSuccessRatePlot_1alg_error_OPE.mat')['ave_success_rate_plot'][0]
assert sr.shape==(320,21) and pr.shape==np_curve.shape==(320,51)
assert all(np.isfinite(curve).all() for curve in [sr,pr,np_curve])

def aggregate(indices):
    curves = [curve[indices] for curve in [sr,pr,np_curve]]
    all_sequence = [curve.mean(axis=0) for curve in curves]
    nonzero_masks = [curve.sum(axis=1)>np.finfo(float).eps for curve in curves]
    toolkit_plot = [curve[mask].mean(axis=0) for curve,mask in zip(curves,nonzero_masks)]
    def scores(c):
        return dict(SR_AUC=float(c[0].mean()), PR_at_20=float(c[1][20]),
                    NPR_AUC=float(c[2].mean()), NP_at_020=float(c[2][20]))
    return dict(sequences=len(indices),all_sequence_metrics=scores(all_sequence),
                toolkit_plot_metrics=scores(toolkit_plot),
                zero_curve_sequences=dict(zip(['success','precision','normalized_precision'],
                                              [int((~mask).sum()) for mask in nonzero_masks])),
                success_curve=all_sequence[0].tolist(),precision_curve=all_sequence[1].tolist(),
                normalized_precision_curve=all_sequence[2].tolist())

overall = aggregate(list(range(320)))
per_attribute = {}
for label in attributes['labels']:
    indices = [i for i,name in enumerate(names) if attributes['attributes'][name][label]]
    assert indices, label
    per_attribute[label] = aggregate(indices)
per_sequence = [dict(sequence=name,frames=rows[i]['frame_count'],SR_AUC=float(sr[i].mean()),
                     PR_at_20=float(pr[i,20]),NPR_AUC=float(np_curve[i].mean()),NP_at_020=float(np_curve[i,20]),
                     success_curve=sr[i].tolist(),precision_curve=pr[i].tolist(),
                     normalized_precision_curve=np_curve[i].tolist()) for i,name in enumerate(names)]
report = dict(status='PINNED_TOOLKIT_SCORING_COMPLETE',dataset='VisEvent',identity=identity,
              evaluation_type='real_gt',metric_units='fractions_0_to_1',
              headline_metrics=overall['toolkit_plot_metrics'],overall=overall,
              sequence_count=320,frame_pairs=157421,missing_sequences=0,
              per_sequence=per_sequence,attributes=per_attribute,
              thresholds=dict(success=np.linspace(0,1,21).tolist(),precision=list(range(51)),
                              normalized_precision=np.linspace(0,.5,51).tolist()),
              toolkit_repository='https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark',
              toolkit_commit='c30908779d58404ce932abdbdb68a4dfcd07e684',
              toolkit_role='official_VisEvent_MATLAB_scorer',toolkit_source_sha256=source_hashes,
              groundtruth='audited_existing_dataset_groundtruth.txt',
              annotation_version=manifest['annotation_version'],
              repository_annos_zip_equivalence=manifest['repository_annos_zip_equivalence'],
              published_metric_comparability=manifest['published_metric_comparability'],attributes_sha256=hashlib.sha256(Path(args.attributes).read_bytes()).hexdigest(),
              manifest_sha256=manifest_sha,coverage_sha256=hashlib.sha256(Path(args.coverage).read_bytes()).hexdigest(),
              timing=coverage['timing'],elapsed_seconds=coverage['elapsed_seconds'],fps=coverage['fps'],
              gpu_memory=coverage['gpu_memory'])
(root/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'per_sequence.csv').open('w',newline='') as destination:
    writer = csv.DictWriter(destination,fieldnames=['sequence','frames','SR_AUC','PR_at_20','NPR_AUC','NP_at_020'])
    writer.writeheader()
    writer.writerows({key:row[key] for key in writer.fieldnames} for row in per_sequence)
figure, axes = plt.subplots(1,3,figsize=(13,4))
for axis,thresholds,key,title in zip(axes,report['thresholds'].values(),
                                    ['success_curve','precision_curve','normalized_precision_curve'],
                                    ['Success','Precision','Normalized precision']):
    axis.plot(thresholds,overall[key])
    axis.set(title=title,ylabel='All-sequence success fraction',ylim=(0,1))
    axis.grid(alpha=.3)
figure.tight_layout()
figure.savefig(root/'full_curves.png',dpi=180)
plt.close(figure)
print(json.dumps(dict(status=report['status'],dataset='VisEvent',headline_metrics=report['headline_metrics'])))
