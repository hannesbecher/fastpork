"""
aggregate_two_species.py:
aggregate solid k-mers in many samples of two different species,
store the resulting hash with pair counts.

Jens Zentgraf & Sven Rahmann, 2022
"""

from argparse import ArgumentParser
from importlib import import_module
import gc
import sys

import numpy as np
from numpy.random import seed as randomseed
from numba import njit, uint64, int64, int32

from fastcash.hashio import load_hash, save_hash
from fastcash.mathutils import bitsfor
from fastcash.values import pairs as values_pairs


def compile_transfer(h1, h2, threshold, increment, walkseed):
    """
    Return a compiled function
    'transfer(ht1, ht2)'
    that transfers solid k-mers from ht1 (count >= threshold)
    into ht2.
    """
    nsubtables = h1.subtables
    if nsubtables == 0:
        raise RuntimeError("Only hashes with subtables are supported")
    npages = h1.npages
    pagesize = h1.pagesize
    is_slot_empty_at = h1.private.is_slot_empty_at
    get_value_at = h1.private.get_value_at
    get_signature_at = h1.private.get_signature_at
    get_subkey_from_page_signature = h1.private.get_subkey_from_page_signature
    get_key_from_subtable_subkey = h1.private.get_key_from_subtable_subkey
    update = h2.update
    
    @njit(nogil=True)
    def set_seed():
        randomseed(walkseed)

    @njit(nogil=True, locals=dict(
        p=uint64, s=int64, sig=uint64,
        subkey=uint64, key=uint64,
        v1=int64, status=int32, info=uint64))
    def transfer(ht1, ht2):
        set_seed()
        done = err = 0
        for t in range(nsubtables):
            print("#     Transfer subtable", t+1, "/", nsubtables)
            for p in range(npages):
                for s in range(pagesize):
                    if is_slot_empty_at(ht1, t, p, s):  continue
                    v1 = get_value_at(ht1, t, p, s)
                    if v1 < threshold: continue
                    sig = get_signature_at(ht1, t, p, s)
                    subkey = get_subkey_from_page_signature(p, sig)
                    key = get_key_from_subtable_subkey(t, subkey)
                    status, info = update(ht2, key, increment)
                    err += (status == 0)
                    done += 1
        return done, err
    return transfer


def check_consistency(L, new):
    if not all(x == new for x in L):
        raise RuntimeError(f"items in {L} differ from {new}")
    L.append(new)


def get_threshold(name, ext):
    assert ext.startswith('.'), f"extension {ext} does not start with a dot (.)"
    ix = name.rfind('.')
    tname = name + ext if ix < 0 else name[:ix] + ext
    with open(tname, "rt") as ft:
        t = int(ft.readline().strip().split()[0])
    return t


def process(hout, names, inc, vset, t_ext, all_ks, all_rc, walkseed):
    nvalues = vset.NVALUES
    update = vset.update
    err = 0
    for name in names:
        print(f"#\n# Loading k-mer counter '{name}'")
        h, values, info = load_hash(name)
        t = get_threshold(name, t_ext)
        k = int(info['k'])
        rcmode = info.get('rcmode', values.RCMODE)
        subtables = h.subtables
        print(f"#     k={k}, rcmode={rcmode}, subtables={subtables}, threshold={t}")
        check_consistency(all_ks, k)
        if not isinstance(rcmode, str):
            raise RuntimeError(f"rcmode {rcmode} is not a string")
        check_consistency(all_rc, rcmode)
        if hout is None:
            # First-time setup
            print("#\n# ==== Allocating output hash!")
            hashtype = info['hashtype'].decode("ASCII")
            hashfuncs = info['hashfuncs'].decode("ASCII")
            print(f"# Building hash table of type '{hashtype}'...")
            print(f"# Hash functions: {hashfuncs}")
            hashmodule = "hash_" + hashtype
            m = import_module("fastcash." + hashmodule)
            hout = m.build_hash(
                h.universe, 4_000_000_000, 9,  # TODO: constants!
                h.pagesize, hashfuncs, nvalues, update,
                aligned=False, nfingerprints=-1,
                init=True, maxwalk=h.maxwalk)
            walkseed = info.get('walkseed', walkseed)
            print("# ====\n#")
            sys.stdout.flush()
        print("# Transferring...")
        sys.stdout.flush()
        transfer = compile_transfer(h, hout, t, inc, walkseed)
        nt, derr = transfer(h.hashtable, hout.hashtable)
        print(f"# Transferred {nt} k-mers with {derr} errors; cleaning up...")
        sys.stdout.flush()
        err += derr
        del h;  gc.collect()
    return hout, err, walkseed


def compile_get_hist2d(hout, gs1, gs2):
    bits = max(bitsfor(gs1+1), bitsfor(gs2+1))
    mask = uint64(2**bits - 1)
    nsubtables = hout.subtables
    npages = hout.npages
    pagesize = hout.pagesize
    is_slot_empty_at = hout.private.is_slot_empty_at
    get_value_at = hout.private.get_value_at

    @njit(locals=dict(val=uint64, v1=uint64, v2=uint64))
    def get_hist2d(ht):
        hist = np.zeros((gs1+1, gs2+1), dtype=np.int64)
        for t in range(nsubtables):
            for p in range(npages):
                for s in range(pagesize):
                    if is_slot_empty_at(ht, t, p, s):  continue
                    val = get_value_at(ht, t, p, s)
                    v1 = val & mask
                    v2 = (val >> bits) & mask
                    if v1 > gs1 or v2 > gs2:
                        print("ASSERT ERROR:", v1, gs1, v2, gs2)
                    assert v1 <= gs1
                    assert v2 <= gs2
                    hist[v1,v2] += 1
        return hist

    return get_hist2d


def print_2d_histogram(hist, title=None):
    if title is not None:
        print(title)
    m, n = hist.shape
    print("  | ", end="")
    for j in range(n):
        print(f"{j:11d}", end=" ")
    print()
    for i in range(m):
        print(i, end=" | ")
        for j in range(n):
            print(f"{hist[i,j]:11d}", end=" ")
        print()


def main(args):
    RCMODE = 'max'
    walkseed = 7
    t_ext = args.threshold_extension
    all_ks = [args.k] if args.k is not None else []
    all_rc = [RCMODE]  # keep track of k-mer sizes and rcmodes
    gs1 = len(args.first)
    gs2 = len(args.second) if args.second is not None else 0
    bits = max(bitsfor(gs1+1), bitsfor(gs2+1))
    valuestr = f'values.pairs {bits} {bits}'
    vset = values_pairs.initialize(bits, bits, rcmode="max")
    print(f"# Group sizes: first={gs1}, second={gs2}")
    print(f"# Bits for counts: first={bits}, second={bits}")
    print(f"# Value set: {vset}")

    print("# Processing first group.")
    hout, err1, walkseed = process(None, args.first, 1, vset, t_ext, all_ks, all_rc, walkseed)
    if args.second is not None:
        print("# Processing second group.")
        increment = 2 ** bits
        _, err2, _ = process(hout, args.second, increment, vset, t_ext, all_ks, all_rc, walkseed)
    else:
        increment = err2 = 0
    err = err1 + err2
    print(f"# Total number of errors {err1} + {err2} = {err}:")

    print("# Collecting statistics...")
    get_hist2d = compile_get_hist2d(hout, gs1, gs2)
    hist = get_hist2d(hout.hashtable)
    print_2d_histogram(hist, title="Number of k-mers in individuals")

    if err == 0:
        print(f"# Saving output hash table '{args.out}'")
        k = all_ks[0]
        save_hash(args.out, hout, valuestr,
            additional=dict(k=k, walkseed=walkseed, rcmode=RCMODE))
    else:
        print(f"# ERROR: {err} k-mers could not be inserted. Exit!")
        sys.exit(1)


def get_argument_parser():
    p = ArgumentParser(description="Aggregate solid k-mers of two species")
    p.add_argument("--first", "-1", nargs='+', required=True, metavar="KMERS_H5",
        help="k-mer counter hashes of first species")
    p.add_argument("--second", "-2", nargs='+', metavar='KMERS_H5',
        help="k-mer counter hashes of second species (optional)")
    p.add_argument("--out", "-o", metavar="OUT_H5", required=True,
        help="output hash with pair counts")
    p.add_argument("-k", type=int,
        help="k-mer size")
    p.add_argument("--threshold-extension", "-T", default=".threshold",
        help="extension of text files with threshold information")
    return p


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
