"""
fastcash.values.pairs

Provides a value set for counting k-mers in two differnet counters
with a given number of bits.
Together, the number of bits must be <= 64.
(high bits: fastacounter, low bits: fastqcounter)

Provides public constants:
- NVALUES
and functions:
- get_value_from_name
- update
- is_compatible

Other provided attributes should be considered private, as they may change.
"""

from collections import namedtuple

import numpy as np
from numba import njit, uint64, int64, boolean

ValueInfo = namedtuple("ValueInfo", [
    "name",
    "NVALUES",
    "RCMODE",
    "bits1",
    "bits2",
    "update",
    "is_compatible",
    ])


def initialize(bits1, bits2, rcmode="max"):
    bits1 = int(bits1)
    bits2 = int(bits2)
    nvalues = 2 ** (bits1 + bits2)
    nv1 = uint64(2 ** bits1)
    nv2 = uint64(2 ** bits2)

    @njit(nogil=True, locals=dict(
            old=uint64, new=uint64, upd=uint64, 
            c1=uint64, c2=uint64,
            new1=uint64, new2=uint64))
    def update(old, new):
        """
        update(uint64, uint64) -> uint64
        Update old value (stored) with a new value (from current seq.).
        Return upated value.
        """
        c1 = old & uint64(nv1-1)
        c2 = (old >> bits1) & uint64(nv2-1)
        if new < nv1: # increase c1
            new1 = c1 + new if c1 < uint64(nv1-new) else (nv1 - 1)
            upd = new1 | (c2 << bits1)
        else:  # increase fastacounter
            new >>= bits1
            new2 = c2 + new if c2 < uint64(nv2-new) else (nv2 - 1)
            upd = (new2 << bits1) | c1
        return upd

    @njit( ###__signature__ boolean(uint64, uint64),
        nogil=True, locals=dict(observed=uint64, stored=uint64))
    def is_compatible(observed, stored):
        """
        is_compatible(uint64, uint64) -> bool
        Check wheter an observed value is compatible with a stored value,
        i.e., iff there exists new such that update(observed, new) == stored.
        """
        # TODO
        return True if stored >= observed else False


    return ValueInfo(
        name = "pairs",
        NVALUES = nvalues,
        RCMODE = rcmode,
        bits1 = bits1,
        bits2 = bits2,
        update = update,
        is_compatible = is_compatible,
        )
