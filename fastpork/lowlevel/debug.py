"""
Define debugging/timing tools.
"""

#TODO:
"""
Generate a long random DNA sequence s from {0,1,2,3}^n.
Given k and q < k, compute k-mers along s and minimizers in k-mers.
See how often the minimizer changes.
Ideally: Minimizer frequently stays the same for k-q+1 k-mers.
So if we choose q = k-2, we have the same minimizer for 3 successive k-mers.
An open question is how to deal with the symmetry w.r.t. reverse complement.
Since we use rcmode=max mostly, we should examine the effect of min vs. max.
"""

from datetime import datetime as dt
from numba import njit

def _do_nothing(*args):
    pass


def _timestamp(previous=None, msg=None, minutes=False):
    now = dt.now()
    if previous is not None and msg is not None:
        elapsed_sec = (now-previous).total_seconds()
        if minutes:
            elapsed_min = elapsed_sec / 60.0
            print(f"{msg}: {elapsed_min:.2f} min")
        else:
            print(f"{msg}: {elapsed_sec:.1f} sec")
    elif msg is not None:
        print(f"# {now:%Y-%m-%d %H:%M:%S}: {msg}")
    return now


def define_debugfunctions(*, debug=True, compile=True, times=True):
    debugprint = print if debug else njit(_do_nothing) if compile else _do_nothing
    timestamp = _timestamp if times else _do_nothing
    return debugprint, timestamp

