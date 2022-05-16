"""
pick_kmers: pick unique k-mers from a k-mer selection
Jens Zentgraf & Sven Rahmann, 2022
"""

#import datetime
#import os.path
from argparse import ArgumentParser
from importlib import import_module

import numpy as np
from numba import njit, uint64, int64

from fastcash.hashio import load_hash, save_hash
from fastcash.mathutils import print_histogram, print_histogram_tail


def compile_filter(hs, hg, hf, f_value=2):
    """
    Return a compiled function
    'pick_kmers(ht_selection, ht_genomic, ht_filtered)'
    """
    s_subtables = hs.subtables
    s_npages = hs.npages
    s_pagesize = hs.pagesize
    s_is_slot_empty_at = hs.private.is_slot_empty_at
    s_get_signature_at = hs.private.get_signature_at
    s_get_subkey_from_page_signature = hs.private.get_subkey_from_page_signature
    s_get_key_from_sub_subkey = hs.private.get_key_from_sub_subkey
    g_get_value = hg.get_value
    f_get_subtable_subkey = hf.get_subtable_subkey
    f_overwrite = hf.overwrite

    @njit(nogil=True, locals=dict(
          p=uint64, s=int64, sig=uint64,
          subkey=uint64, key=uint64, gv=uint64))
    def pick_kmers(hts, htg, htf):
        """
        Return histogram statistics of genomic counts of kmers in hts.
        Transfer kmers from hts with genomic count == 1 into htf.

        hts: k-mers from selection (some genes)
        htg: k-mers from genome
        htf: output, k-mers from hts that are unique in htg
        """
        # NOTE: This only works with subtables, not when subtables == 0:
        assert s_subtables != 0
        hist = np.zeros(16, dtype=np.uint64)
        for sub in range(s_subtables):
            for p in range(s_npages):
                for s in range(s_pagesize):
                    if s_is_slot_empty_at(hts, sub, p, s):  continue
                    sig = s_get_signature_at(hts, sub, p, s)
                    subkey = s_get_subkey_from_page_signature(p, sig)
                    key = s_get_key_from_sub_subkey(sub, subkey)
                    # look up value for key in genomic table
                    gv = g_get_value(htg, key)
                    hist[int64(min(int64(gv),15))] += 1
                    if gv != 1: continue
                    # store in new table
                    subtf, subkf = f_get_subtable_subkey(key)
                    f_overwrite(htf, subtf, subkf, f_value)
        return hist
    return pick_kmers


def get_argument_parser():
    p = ArgumentParser(description="Pick k-mers from selection that are unique in genomic index")
    p.add_argument("selection",
        help="index with selected k-mers")
    p.add_argument("genomic",
        help="index with genomic k-mer counts")
    p.add_argument("filtered",
        help="output: index with filtered/picked k-mers")
    return p


def main(args):
    # Load hash tables
    print("# Loading selection:")
    h1, values1, info1 = load_hash(args.selection)
    (valuehist, _, _, _) = h1.get_statistics(h1.hashtable)
    print_histogram(valuehist[0], title="Value statistics of initial selection:", shorttitle="values", fractions="%+")

    print("\n# Loading genomic counts:")
    h2, values2, info2 = load_hash(args.genomic)
    k1 = int(info1['k'])
    k2 = int(info1['k'])
    if k1 != k2:
        raise RuntimeError(f"ERROR: k values differ: {k1} != {k2}")
    rcmode1 = info1.get('rcmode', values1.RCMODE)
    rcmode2 = info2.get('rcmode', values2.RCMODE)
    if rcmode1 != rcmode2:
        raise RuntimeError(f"ERROR: rcmodes differ: {rcmode1} != {rcmode2}")
    ##print(k1, k2, rcmode1, rcmode2, type(rcmode1), type(rcmode2))


    # Create a new hash table for the unique selected genes
    vmodule = import_module("fastcash.values.count")
    values3 = vmodule.initialize(8)
    valueset = f"values.count 8"  # use 3 bits as xengsort
    GRAFT_VALUE = 2

    print("\n# Creating new hash table for filtered selection...")
    hashtype = info1['hashtype'].decode("ASCII")
    hashfuncs = info1['hashfuncs'].decode("ASCII")
    print(f"# Building hash table of type '{hashtype}'...")
    print(f"# Hash functions: {hashfuncs}")
    hashmodule = "hash_" + hashtype
    m = import_module("fastcash." + hashmodule)
    h3 = m.build_hash(h1.universe, h1.n, h1.subtables,
        h1.pagesize, hashfuncs, values3.NVALUES, values3.update,
        aligned=h1.aligned, nfingerprints=h1.nfingerprints,
        init=True, maxwalk=h1.maxwalk, shortcutbits=h1.shortcutbits)

    # filter k-mers
    pick_kmers = compile_filter(h1, h2, h3, f_value=GRAFT_VALUE)
    hist = pick_kmers(h1.hashtable, h2.hashtable, h3.hashtable)
    print_histogram(hist, title="Count statistics of gene k-mers:", shorttitle="counts", fractions="%", nonzerofrac=True)
    save_hash(args.filtered, h3, valueset,
            additional=dict(k=k1, walkseed=info1['walkseed'], rcmode=rcmode1))


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
