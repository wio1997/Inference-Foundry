"""Read a versioned experiment mode once when constructing a new cohort."""
import json
from pathlib import Path

def read_cohort_mode(path):
    mode=json.loads(Path(path).read_text())
    if (set(mode)!={'phase','enabled','version'}
            or type(mode['enabled']) is not bool
            or type(mode['version']) is not int or mode['version']<1
            or not isinstance(mode['phase'],str) or not mode['phase']):
        raise RuntimeError('invalid cohort mode')
    return mode
