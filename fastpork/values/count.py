"""
fastcash.values.count

Provides a value set for counting k-mers up to a given maximum count.

Provides public constants:
- NAME = "count"
- NVALUES
- RCMODE
and functions:
- get_value_from_name
- update
- is_compatible

Other provided attributes should be considered private, as they may change.
"""

from collections import namedtuple
from math import ceil, log2

import numpy as np
from numba import njit, uint64, int64, boolean

ValueInfo = namedtuple("ValueInfo", [
    "NAME",
    "NVALUES",
    "RCMODE",
    "get_value_from_name",
    "update",
    "is_compatible",
    "strsymbols",
    "bytesymbols",
    "unit",
    "true_nvalues",
    "bits",
    ])


def initialize(nvalues, rcmode="max"):
    nvalues = int(nvalues)
    bits = int(ceil(log2(nvalues)))

    def get_value_from_name(name, onetwo=1):
        return 1  # always one count

    @njit(nogil=True, locals=dict(
            old=uint64, new=uint64, updated=uint64))
    def update(old, new):
        """
        update(uint64, uint64) -> uint64
        Update old value (stored) with a new value (from current seq.).
        Return upated value.
        """
        updated = uint64(min(old + new,  nvalues - 1))
        return updated

    @njit(
        nogil=True, locals=dict(observed=uint64, stored=uint64))
    def is_compatible(observed, stored):
        """
        is_compatible(uint64, uint64) -> bool
        Check wheter an observed value is compatible with a stored value,
        i.e., iff there exists new such that update(observed, new) == stored.
        """
        s = stored % nvalues
        o = observed % nvalues
        return (s >= o)


    return ValueInfo(
        NAME = "count",
        NVALUES = nvalues,
        RCMODE = rcmode,
        get_value_from_name = get_value_from_name,
        update = update,
        is_compatible = is_compatible,
        strsymbols = dict(),
        bytesymbols = dict(),
        unit = 1,
        true_nvalues = nvalues,
        bits = bits,
        )
