"""
Module fastcash.hashfunctions

This module provides
several hash functions for different purposes.
"""

from math import ceil, log2
from random import randrange

import numpy as np
from numba import njit, uint64, uint32, int64, boolean

from .mathutils import bitsfor, nextodd, inversemodprime, inversemodpow2



DEFAULT_HASHFUNCS = ("linear62591", "linear42953", "linear48271")

def parse_names(hashfuncs, choices, maxfactor=2**32-1):
    """
    Parse colon-separated string with hash function name(s),
    or string with a special name ("default", "random").
    Return tuple with hash function names.
    """
    if hashfuncs == "default":
        return DEFAULT_HASHFUNCS[:choices]
    elif hashfuncs == "random":
        while True:
            r = [randrange(3, maxfactor, 2) for _ in range(choices)]
            if len(set(r)) == choices: break
        hf = tuple(["linear"+str(x) for x in r])
        return hf
    hf = tuple(hashfuncs.split(":"))
    if len(hf) != choices:
        raise ValueError(f"Error: '{hashfuncs}' does not contain {choices} functions.")
    return hf


def build_get_page_fpr(name, universe, npages, *,
        nfingerprints=-1):
    """
    Build hash function 'name' for keys in {0..'universe'-1} that
    hashes injectively to 'npages' pages and 'nfingerprints' fingerprints.
    
    Return a pair of functions: (get_page_fingerprint, get_key), where
    * get_page_fingerprint(key) returns the pair (page, fingerprint),
    * get_key(page, fpr)        returns the key for given page and fingerprint,
    where page is in {0..npages-1}, fingerprint is in {0..nfingerprints-1}.
    
    Invariants:
    - get_key(*get_page_fingerprint(key)) == key for all keys in {0..universe-1}.
    
    The following hash function 'name's are implemented:
    1. linear{ODD}, e.g. linear123, with a positive odd number.
    ...
    
    Restrictions:
    Currently, universe must be a power of 4 (corresponding to a DNA k-mer).
    """
    if nfingerprints < 0:
        nfingerprints = int(ceil(universe / npages))
    elif nfingerprints == 0:
        nfingerprints = 1
    qbits = bitsfor(universe)
    pagebits = int(ceil(log2(npages)))
    pagemask = uint64(2**pagebits - 1)
    fprbits = int(ceil(log2(nfingerprints)))
    fprmask = uint64(2**fprbits - 1)
    codemask = uint64(2**qbits - 1)
    shift = qbits - pagebits

    if 4**(qbits//2) != universe:
        raise ValueError("hash functions require that universe is a power of 4")
    else:
        q = qbits // 2
     
    # define a default get_key function
    get_key = None  # will raise an error if called from numba as a function.
    if name.startswith("linear"):  # e.g. "linear12345"
        a = int(name[6:])
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)
        @njit(nogil=True, inline='always', locals=dict(
                code=uint64, swap=uint64, f=uint64, p=uint64))
        def get_page_fpr(code):
            swap = ((code << q) ^ (code >> q)) & codemask
            swap = (a * swap) & codemask
            p = swap % npages
            f = swap // npages
            return (p, f)

        @njit(nogil=True, inline='always', locals=dict(
            key=uint64, page=uint64, fpr=uint64))
        def get_key(page, fpr):
            key = fpr * npages + page
            key = (ai * key) & codemask
            key = ((key << q) ^ (key >> q)) & codemask
            return key
    
    elif name.startswith("affine"):  # e.g. "affine12345+66666"
        raise ValueError(f"unknown hash function '{name}'")
    else:
        raise ValueError(f"unknown hash function '{name}'")
    return (get_page_fpr, get_key)


def extend_func_tuple(funcs, n):
    """Extend a tuple of functions to n functions by appending dummies"""
    n0 = len(funcs)
    if n0 < 1 or n0 > 4:
        raise ValueError("Only 1 to 4 hash functions are supported.")
    if n0 == n: return funcs
    if n0 > n:
        raise ValueError(f"Function tuple {funcs} already has {n0}>{n} elements.")
    return funcs + (funcs[0],) * (n - n0)



def get_hashfunctions(hashfuncs, choices, universe, npages, nfingerprints):
    # Define functions get_pf{1,2,3,4}(key) to obtain pages and fingerprints.
    # Define functions get_key{1,2,3,4}(page, fpr) to obtain keys back.
    # Example: hashfuncs = 'linear123:linear457:linear999'
    # Example new: 'linear:123,456,999' or 'affine:123+222,456+222,999+222'
    hashfuncs = parse_names(hashfuncs, choices)  # ('linear123', 'linear457', ...)

    if choices >= 1:
        (get_pf1, get_key1) = build_get_page_fpr(
            hashfuncs[0], universe, npages, nfingerprints=nfingerprints)
    if choices >= 2:
        (get_pf2, get_key2) = build_get_page_fpr(
            hashfuncs[1], universe, npages, nfingerprints=nfingerprints)
    if choices >= 3:
        (get_pf3, get_key3) = build_get_page_fpr(
            hashfuncs[2], universe, npages, nfingerprints=nfingerprints)
    if choices >= 4:
        (get_pf4, get_key4) = build_get_page_fpr(
            hashfuncs[3], universe, npages, nfingerprints=nfingerprints)

    if choices == 1:
        get_pf = (get_pf1,)
        get_key = (get_key1,)
    elif choices == 2:
        get_pf = (get_pf1, get_pf2)
        get_key = (get_key1, get_key2)
    elif choices == 3:
        get_pf = (get_pf1, get_pf2, get_pf3)
        get_key = (get_key1, get_key2, get_key3)
    elif choices == 4:
        get_pf = (get_pf1, get_pf2, get_pf3, get_pf4)
        get_key = (get_key1, get_key2, get_key3, get_key4)
    else:
        raise ValueError("Only 1 to 4 hash functions are supported.")

    return (hashfuncs, get_pf, get_key)
