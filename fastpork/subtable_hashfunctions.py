"""
Module fastcash.subtable_hashfunctions

This module provides multiple hashfunctions
which used for hashing with subtables 
"""

from numba import njit, uint64
from math import log
import numpy as np
from random import randrange

from .mathutils import inversemodpow2, bitsfor

DEFAULT_HASHFUNCS = ("linear94127", "linear62591", "linear42953", "linear48271")

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

def build_get_sub_subkey_from_key(name, universe, subtables):
    qbits = bitsfor(universe)
    codemask = uint64(2**qbits - 1)
    if 4**(qbits//2) != universe:
        raise ValueError("hash functions require that universe is a power of 4")
    else:
        q = qbits // 2

    if name.startswith("linear"):
        a = int(name[6:])
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)
        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, subkey=uint64, s=uint64, f=uint64, p=uint64))
        def get_sub_subkey(code):
            swap = ((code << q) ^ (code >> q)) & codemask
            swap = (a * swap) & codemask
            subkey = swap // subtables
            s = swap % subtables
            return (s, subkey)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, f=uint64, p=uint64))
        def get_key(sub, subkey):
            swap = subkey * subtables + sub
            swap = (ai * swap) & codemask
            key = ((swap << q) ^ (swap >> q)) & codemask
            return key

    elif name.startswith("affine"):
        a, b = name[6:].split("-")
        a = int(a)
        b = int(b)
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, subkey=uint64, s=uint64))
        def get_sub_subkey(code):
            swap = ((code << q) ^ (code >> q)) & codemask
            swap = (a * (swap ^ b)) & codemask
            subkey = swap // subtables
            s = swap % subtables
            return (s, subkey)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, sub=uint64, subkey=uint64))
        def get_key(sub, subkey):
            swap = subkey * subtables + sub
            swap = ((ai * swap) ^ b) & codemask
            key = ((swap << q) ^ (swap >> q)) & codemask
            return key

    return get_sub_subkey, get_key

def build_get_sub_page_fpr_from_key(tablename, name, universe, npages, subtables):
    """

    """
    get_sub_subkey, get_key_from_sub_subkey = build_get_sub_subkey_from_key(tablename, universe, subtables)

    universe = universe//(4**(int(log(subtables,4))))
    qbits = bitsfor(universe)
    codemask = uint64(2**qbits - 1)
    if 4**(qbits//2) != universe:
        raise ValueError("hash functions require that universe is a power of 4")
    else:
        q = qbits // 2


    if name.startswith("linear"):
        a = int(name[6:])
        ai = uint64(inversemodpow2(a, int(4**28)))
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)
        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, subkey=uint64, s=uint64, f=uint64, p=uint64))
        def get_sub_page_fpr(code):
            s, subkey = get_sub_subkey(code)
            swap = ((subkey << q) ^ (subkey >> q)) & codemask
            swap = (a * swap) & codemask
            f = swap // npages
            p = swap % npages
            return (s, p, f)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, fpr=uint64, page=uint64, sub=uint64))
        def get_key(sub, page, fpr):
            subkey = fpr * npages + page
            subkey = (ai * subkey) & codemask
            subkey = ((subkey << q) ^ (subkey >> q)) & codemask
            return get_key_from_sub_subkey(sub, subkey)

    elif name.startswith("affine"):
        a, b = name[6:].split("-")
        a = int(a)
        b = uint64(b)
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, subkey=uint64, s=uint64, f=uint64, p=uint64))
        def get_sub_page_fpr(code):
            s, subkey = get_sub_subkey(code)
            swap = ((subkey << q) ^ (subkey >> q)) & codemask
            swap = (a * (swap ^ b)) & codemask
            f = swap // npages
            p = swap % npages
            return (s, p, f)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, fpr=uint64, page=uint64, sub=uint64))
        def get_key(sub, page, fpr):
            subkey = fpr * npages + page
            subkey = ((ai * subkey) ^ b) & codemask
            subkey = ((subkey << q) ^ (subkey >> q)) & codemask
            return get_key_from_sub_subkey(sub, subkey)
    else:
        raise ValueError(f"unkown hash function {name}")
    return get_sub_page_fpr, get_key


def build_get_page_fpr_from_subkey(name, universe, npages, subtables):
    universe = universe//(4**(int(log(subtables,4))))
    qbits = bitsfor(universe)
    codemask = uint64(2**qbits - 1)
    if 4**(qbits//2) != universe:
        raise ValueError("hash functions require that universe is a power of 4")
    else:
        q = qbits // 2

    if name.startswith("linear"):
        a = int(name[6:])
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)
        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, subkey=uint64, s=uint64, f=uint64, p=uint64))
        def get_page_fpr(subkey):
            swap = ((subkey << q) ^ (subkey >> q)) & codemask
            swap = (a * swap) & codemask
            f = swap // npages
            p = swap % npages
            return (p, f)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, fpr=uint64, page=uint64))
        def get_subkey(page, fpr):
            subkey = fpr * npages + page
            subkey = (ai * subkey) & codemask
            subkey = ((subkey << q) ^ (subkey >> q)) & codemask
            return subkey

    elif name.startswith("affine"):
        a, b = name[6:].split("-")
        a = int(a)
        b = uint64(b)
        ai = uint64(inversemodpow2(a, universe))
        a = uint64(a)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, subkey=uint64, s=uint64, f=uint64, p=uint64))
        def get_page_fpr(subkey):
            swap = ((subkey << q) ^ (subkey >> q)) & codemask
            swap = (a * (swap ^ b)) & codemask
            f = swap // npages
            p = swap % npages
            return (p, f)

        @njit(nogil=True, inline='always', locals=dict(
                    code=uint64, swap=uint64, fpr=uint64, page=uint64, b=uint64))
        def get_subkey(page, fpr):
            subkey = fpr * npages + page
            subkey = ((ai * subkey) ^ b) & codemask
            subkey = ((subkey << q) ^ (subkey >> q)) & codemask
            return subkey

    return get_page_fpr, get_subkey

def build_get_sub_pages_fprs_from_key(names, universe, npages, subtables):
    tablename, name1, name2, name3 = names.split(":")

    get_sub_subkey, get_key_from_sub_subkey = build_get_sub_subkey_from_key(tablename, universe, npages, subtables)
    get_page_fpr1, get_subkey1 = build_get_page_fpr_from_subkey(name1, universe, npages, subtables)
    get_page_fpr2, get_subkey2 = build_get_page_fpr_from_subkey(name2, universe, npages, subtables)
    get_page_fpr3, get_subkey3 = build_get_page_fpr_from_subkey(name3, universe, npages, subtables)
    get_subkeys = (get_subkey1, get_subkey2, get_subkey3)

    @njit(nogil=True, inline='always', locals=dict(
                code=uint64, subkey=uint64, s=uint64, p1=uint64, f1=uint64,
                p2=uint64, f2=uint64, p3=uint64, f3=uint64))
    def get_sub_pages_fprs(code):
        s, subkey = get_sub_subkey(code)
        p1, f1, = get_page_fpr1(subkey)
        p2, f2, = get_page_fpr2(subkey)
        p3, f3, = get_page_fpr3(subkey)

        return s, p1, f1, p2, f2, p3, f3

    @njit(nogil=True, inline='always', locals=dict(
                sub=uint64, page=uint64, fpr=uint64, hf=uint64,
                subkey=uint64, key=uint64))
    def get_key(sub, page, fpr, hf):
        if hf==1:
            subkey = get_subkey1(page, fpr)
        elif hf==2:
            subkey = get_subkey2(page, fpr)
        elif hf==3:
            subkey = get_subkey3(page, fpr)
        key = get_key_from_sub_subkey(sub, subkey)
        return key

    return get_sub_pages_fprs, get_key

def get_hashfunctions(firsthashfunc, hashfuncs, choices, universe, npages, subtables):
    # Define function get_sub_subkey(key) to obtain subtable and reduced code.
    # Define function get_key(sub, subkey) to obtaub jkey back from subtable and reduced code.
    # Define functions get_pf{1,2,3,4}(subkey) to obtain pages and fingerprints from reduced key.
    # Define functions get_subkey{1,2,3,4}(page, fpr) to obtain reduced key back.
    # Example: hashfuncs = 'linear123:linear457:linear999'
    # Example new: 'linear:123,456,999' or 'affine:123+222,456+222,999+222'
    hashfuncs = parse_names(hashfuncs, choices)  # ('linear123', 'linear457', ...)

    if choices >= 1:
        (get_pf1, get_subkey1) = build_get_page_fpr_from_subkey(
            hashfuncs[0], universe, npages, subtables)
        (get_spf1, get_key1) = build_get_sub_page_fpr_from_key(
            firsthashfunc, hashfuncs[0], universe, npages, subtables)
    if choices >= 2:
        (get_pf2, get_subkey2) = build_get_page_fpr_from_subkey(
            hashfuncs[1], universe, npages, subtables)
        (get_spf2, get_key2) = build_get_sub_page_fpr_from_key(
            firsthashfunc, hashfuncs[1], universe, npages, subtables)
    if choices >= 3:
        (get_pf3, get_subkey3) = build_get_page_fpr_from_subkey(
            hashfuncs[2], universe, npages, subtables)
        (get_spf3, get_key3) = build_get_sub_page_fpr_from_key(
            firsthashfunc, hashfuncs[2], universe, npages, subtables)
    if choices >= 4:
        (get_pf4, get_subkey4) = build_get_page_fpr_from_subkey(
            hashfuncs[3], universe, npages, subtables)
        (get_spf4, get_key4) = build_get_sub_page_fpr_from_key(
            firsthashfunc, hashfuncs[3], universe, npages, subtables)

    if choices == 1:
        get_pf = (get_pf1,)
        get_spf = (get_spf1,)
        get_subkey = (get_subkey1,)
        get_key = (get_key1,)
    elif choices == 2:
        get_pf = (get_pf1, get_pf2)
        get_spf = (get_spf1, get_spf2)
        get_subkey = (get_subkey1, get_subkey2)
        get_key = (get_key1, get_key2)
    elif choices == 3:
        get_pf = (get_pf1, get_pf2, get_pf3)
        get_spf = (get_spf1, get_spf2, get_spf3)
        get_subkey = (get_subkey1, get_subkey2, get_subkey3)
        get_key = (get_key1, get_key2, get_key3)
    elif choices == 4:
        get_pf = (get_pf1, get_pf2, get_pf3, get_pf4)
        get_spf = (get_spf1, get_spf2, get_spf3, get_spf4)
        get_subkey = (get_subkey, get_subkey2, get_subkey3, get_subkey4)
        get_key = (get_key1, get_key2, get_key3, get_key4)
    else:
        raise ValueError("Only 1 to 4 hash functions are supported.")

    return ((firsthashfunc,) + hashfuncs, get_pf, get_subkey, get_spf, get_key)
