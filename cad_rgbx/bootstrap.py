"""Use CADTrack's model/config and the project's established RGB-X data readers."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
CAD_SOURCE = PROJECT / 'third_party/CADTrackPruned'
sys.path.insert(0, str(CAD_SOURCE))
import lib
import lib.utils

lib.__path__.insert(0, str(PROJECT / 'third_party/XTrack/lib'))
import lib.train
lib.__path__.pop(0)
