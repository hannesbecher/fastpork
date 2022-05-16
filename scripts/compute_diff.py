"""
compute_diff: find k-mers in set difference of two counters
Jens Zentgraf & Sven Rahmann, 2022
"""

from argparse import ArgumentParser
from importlib import import_module

import numpy as np
from numba import njit, uint64, int64
import matplotlib.pyplot as plt

from fastcash.hashio import load_hash, save_hash
from fastcash.mathutils import print_histogram, print_histogram_tail


def plot_histogram(fname, hists, labels):
    fig = plt.figure()
    fig.set_size_inches(24, 12)
    gs = fig.add_gridspec(1, 1, wspace=0)
    ax = gs.subplots()
    for h, label in zip(hists, labels):
        hn = h / h.sum()
        ax.plot(hn[:231], label=label)
    ax.set_title(f"Number of k-mers (y-axis) vs. k-mer count (x-axis)")
    ax.set_xlabel('k-mer count')
    ax.set_ylabel('relative number of k-mers')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(fname, bbox_inches='tight', metadata={
        'Author': 'Sven Rahmann', 'Title': 'K-mer counts'
        })



def compile_filter(h1, h2, a, b):
    """
    Return a compiled function
    'store_diff(ht1, ht2, buffer)'
    """
    nsubtables = h1.subtables
    npages = h1.npages
    pagesize = h1.pagesize
    is_slot_empty_at = h1.private.is_slot_empty_at
    get_signature_at = h1.private.get_signature_at
    get_subkey_from_page_signature = h1.private.get_subkey_from_page_signature
    get_key_from_sub_subkey = h1.private.get_key_from_sub_subkey
    get_value_at_1 = h1.private.get_value_at
    get_value_2 = h2.get_value
    print(f"# Compiling filter with a={a}, b={b}")

    if nsubtables > 1:
        @njit(nogil=True, locals=dict(
            p=uint64, s=int64, sig=uint64, key=uint64, v1=int64, v2=int64, d=int64, minv1=int64))
        def store_diff(ht1, ht2, buffer, hist):
            """
            Store difference of k-mers in (ht1 \\ ht2) in buffer (k-mer codes).
            """
            d = 0; minv1=999_999
            hist[:] = 0
            for t in range(nsubtables):
                for p in range(npages):
                    for s in range(pagesize):
                        if is_slot_empty_at(ht1, t, p, s):  continue
                        sig = get_signature_at(ht1, t, p, s)
                        v1 = get_value_at_1(ht1, t, p, s)
                        if v1 < a: continue
                        subkey = get_subkey_from_page_signature(p, sig)
                        key = get_key_from_sub_subkey(t, subkey)
                        # look up value for key in ht2
                        v2 = get_value_2(ht2, key)
                        hist[v2] += 1
                        if v2 >= b:
                            minv1 = min(minv1, v1)
                            continue
                        buffer[d] = key
                        d += 1
            return d, minv1
    else:
        assert nsubtables == 1
        @njit(nogil=True, locals=dict(
            p=uint64, s=int64, sig=uint64, key=uint64, v1=uint64, v2=uint64, d=int64, minv1=int64))
        def store_diff(ht1, ht2, buffer, hist):
            """
            Store difference of k-mers in (ht1 \\ ht2) in buffer (k-mer codes).
            """
            d = 0; minv1 = 999_999
            hist[:] = 0
            for p in range(npages):
                    for s in range(pagesize):
                        if is_slot_empty_at(ht1, p, s):  continue
                        sig = get_signature_at(ht1, p, s)
                        v1 = get_value_at_1(ht1, p, s)
                        if v1 < a: continue
                        key = get_subkey_from_page_signature(p, sig)
                        # look up value for key in ht2
                        v2 = get_value_2(ht2, key)
                        hist[v2] += 1
                        if v2 >= b:
                            minv1 = min(minv1, v1)
                            continue
                        buffer[d] = key
                        d += 1
            return d, minv1
    return store_diff


def get_argument_parser():
    p = ArgumentParser(description="Compute diff between solid k-mers of two k-mer counters")
    p.add_argument("counterA",
        help="k-mer counter A (HDF5)")
    p.add_argument("counterB",
        help="k-mer counter B (HDF5)")
    p.add_argument("-a", type=int,
        help="threshold for counter A")
    p.add_argument("-b", type=int,
        help="threshold for counter B")
    return p


def get_mincount(hist, cnt):
    cs = np.cumsum(hist[::-1])[::-1]
    return np.amax(np.where(cs >= cnt)[0])


def main(args):
    # Load hash tables with counters
    print("# Loading counter A:")
    h1, values1, info1 = load_hash(args.counterA)
    (vhist1, _, _, _) = h1.get_statistics(h1.hashtable)
    hist = np.sum(vhist1, axis=0)
    print_histogram(hist, title="Statistics of counter A:", shorttitle="A", fractions="%+")

    print()
    print("# Loading counter B:")
    h2, values2, info2 = load_hash(args.counterB)
    (vhist2, _, _, _) = h2.get_statistics(h2.hashtable)
    vhist2 = np.sum(vhist2, axis=0)
    print_histogram(vhist2, title="Statistics of counter B:", shorttitle="B", fractions="%+")

    # Check consistency
    k1 = int(info1['k'])
    k2 = int(info1['k'])
    if k1 != k2:
        raise RuntimeError(f"ERROR: k values differ: {k1} != {k2}")
    rcmode1 = info1.get('rcmode', values1.RCMODE)
    rcmode2 = info2.get('rcmode', values2.RCMODE)
    if rcmode1 != rcmode2:
        raise RuntimeError(f"ERROR: rcmodes differ: {rcmode1} != {rcmode2}")
    if not (isinstance(rcmode1, str) and isinstance(rcmode2, str)):
        raise RuntimeError(f"ERROR: rcmodes are not strings: {rcmode1}, {rcmode2}")

    # Create buffers for diffs
    NAB = np.sum(hist)
    NBA = np.sum(vhist2)
    deltaAB = np.zeros(NAB, dtype=np.uint64)
    deltaBA = np.zeros(NBA, dtype=np.uint64)
    a = args.a if args.a is not None else get_mincount(hist, NBA)
    b = args.b if args.b is not None else 2
    print(f"Using a count threshold of {a} for {args.counterA} ({NAB} entries).")
    print(f"Using a count threshold of {b} for {args.counterB} ({NBA} entries)..")

    # Compile and run filters
    diffAB = compile_filter(h1, h2, a, b)
    nAB, vAB = diffAB(h1.hashtable, h2.hashtable, deltaAB, vhist2)
    assert nAB <= NAB
    print(f"{k1}-mers in {args.counterA} but not in {args.counterB}: {nAB}, min value in both: {vAB}")
    print(f"Elements with counts above {a} in A have counts in B as follows:")
    #print_histogram(vhist2, title="Counts in B:", shorttitle="B", fractions="%+")
    deltaAB[:nAB].sort()
    
    diffBA = compile_filter(h2, h1, b, a)
    hist2 = np.zeros_like(hist)
    nBA, vBA = diffBA(h2.hashtable, h1.hashtable, deltaBA, hist2)
    assert nBA <= NBA
    print(f"{k2}-mers in {args.counterB} but not in {args.counterA}: {nBA}, min value in both: {vBA}")
    print(f"Elements with counts above {b} in B have counts in A as follows:")
    print_histogram(hist2, title="Counts in A:", shorttitle="A", fractions="%+")
    deltaBA[:nBA].sort()

    plot_histogram("blob.png", [hist, hist2], ["all extracted", "k-mers from G"])


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
