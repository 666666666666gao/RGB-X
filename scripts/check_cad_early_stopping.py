"""Deterministic plateau-rule checks; no training or GPU allocation."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx import bootstrap
from cad_rgbx.sot_baseline import plateau_decision


def history(losses):
    return [dict(task_response_box_iou=[[loss/2, loss/2, 0.5]]*3) for loss in losses]


settings = dict(rolling_epochs=3, relative_min_improvement=0.005, patience_epochs=5, minimum_epochs=10)
assert not plateau_decision(history([1.0]*9), settings)['stop']
assert plateau_decision(history([1.0]*10), settings)['stop']
assert not plateau_decision(history([2*0.95**epoch for epoch in range(20)]), settings)['stop']
losses = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]+[1.0]*10
decisions = [plateau_decision(history(losses[:end]), settings) for end in range(1, len(losses)+1)]
first_stop = next(end for end, decision in enumerate(decisions, 1) if decision['stop'])
assert first_stop == 13, first_stop
print('EARLY_STOP_CHECK_PASS constant_stop10 improving_continue plateau_stop13 no_test_metrics')
