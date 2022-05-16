"""
gainloss.py: find gains and losses in Ossabaw
Jens Zentgraf & Sven Rahmann, 2022
"""

from argparse import ArgumentParser
from importlib import import_module

import numpy as np
from numba import njit, uint64, int64
#import matplotlib.pyplot as plt

from fastcash.hashio import load_hash, save_hash
from fastcash.mathutils import print_histogram


def compile_filter(h1, hall, condition, message):
    """
    Return a compiled function
    'diff(TODO)' 
    """
    nsubtables = h1.subtables
    npages = h1.npages
    pagesize = h1.pagesize
    is_slot_empty_at = h1.private.is_slot_empty_at
    get_signature_at = h1.private.get_signature_at
    get_subkey_from_page_signature = h1.private.get_subkey_from_page_signature
    get_key_from_sub_subkey = h1.private.get_key_from_sub_subkey
    get_value_at_1 = h1.private.get_value_at
    gvo, gvm, gvb, gvg = [h.get_value for h in hall]

    @njit(nogil=True, locals=dict(
        key=uint64, vo=int64, vm=int64, vb=int64, vg=int64))
    def _lookup(key, hto, htm, htb, htg):
        vo = gvo(hto, key)
        vm = gvm(htm, key)
        vb = gvb(htb, key)
        vg = gvg(htg, key)
        result = condition(vo, vm, vb, vg)
        if result:
            print(key, " ", vo, vm, vb, vg, " ", message)
        return result

    if nsubtables > 1:
        @njit(nogil=True, locals=dict(
            t=uint64, p=uint64, s=int64, sig=uint64, key=uint64))
        def diff(ht1, allht):
            """
            show k-mers in ht1 \\ ht2 with counts
            """
            (hto, htm, htb, htg) = allht
            d = 0
            for t in range(nsubtables):
                for p in range(npages):
                    for s in range(pagesize):
                        if is_slot_empty_at(ht1, t, p, s):  continue
                        sig = get_signature_at(ht1, t, p, s)
                        subkey = get_subkey_from_page_signature(p, sig)
                        key = get_key_from_sub_subkey(t, subkey)
                        d += _lookup(key, hto, htm, htb, htg)
            return d
    else:
        assert nsubtables == 1
        @njit(nogil=True, locals=dict(
            t=uint64, p=uint64, s=int64, sig=uint64, key=uint64))
        def diff(ht1, allht):
            """
            show k-mers in ht1 \\ ht2 with counts
            """
            (hto, htm, htb, htg) = allht
            d = 0
            for p in range(npages):
                    for s in range(pagesize):
                        if is_slot_empty_at(ht1, p, s):  continue
                        sig = get_signature_at(ht1, p, s)
                        key = get_subkey_from_page_signature(p, sig)
                        d += _lookup(key, hto, htm, htb, htg)
            return d
    return diff



#def get_mincount(hist, cnt):
#    cs = np.cumsum(hist[::-1])[::-1]
#    return np.amax(np.where(cs >= cnt)[0])


def load_table(fname, description, show_histogram=False):
    print(f"# Loading {description} from '{fname}'...")
    h, values, info = load_hash(fname)
    (hist, _, _, _) = h.get_statistics(h.hashtable)
    if len(hist.shape)==2:
        hist = np.sum(hist, axis=0)
    n = np.sum(hist)
    k = int(info['k'])
    print(f"# {description} contains {n} distinct {k}-mers.")
    if show_histogram:
        print_histogram(hist, title=f"Statistics of {description}", shorttitle=description, fractions="%+")
    print()
    return h, values, info, hist, n


def check_consistency(infos, vs):
    ks = [int(info['k']) for info in infos]
    if min(ks) != max(ks):
        raise RuntimeError(f"ERROR: k values differ: {ks}")
    rcs = [info.get('rcmode', value.RCMODE) for info, value in zip(infos, vs)]
    if not all(isinstance(rc, str) for rc in rcs):
        raise RuntimeError(f"ERROR: not all rcmodes are strings: {rcs}")
    if min(rcs) != max(rcs):
        raise RuntimeError(f"ERROR: rcmodes differ: {rcs}")



def get_argument_parser():
    p = ArgumentParser(description="Compute gains and losses of k-mers (ossabaw vs rest)")
    p.add_argument("--ossabaw", "-o", required=True,
        help="Ossabaw extracted k-mer counter (H5)")
    p.add_argument("--minipig", "-m", required=True,
        help="Minipig extracted k-mer counter (H5)")
    p.add_argument("--bait", "--unique", "--genes", "-b", required=True,
        help="Bait k-mer set from genes, uniques (H5)")
    p.add_argument("--genome", "--reference", "-g", required=True,
        help="Sus scrofa full genome k-mer counter (H5)")
    p.add_argument("--show-histograms", "-H", action="store_true",
        help="show count histograms of all counters")
    return p


def main(args):
    # Load hashes
    show_hist = args.show_histograms
    ho, vo, infoo, histo, no = load_table(args.ossabaw, "Ossabaw", show_hist)
    hm, vm, infom, histm, nm = load_table(args.minipig, "Minipig", show_hist)
    hb, vb, infob, histb, nb = load_table(args.bait, "Bait", show_hist)
    hg, vg, infog, histg, ng = load_table(args.genome, "Genome", show_hist)

    # Check consistency
    infos = [infoo, infom, infob, infog]
    vs = [vo, vm, vb, vg]
    check_consistency(infos, vs)

    # Create buffers for diffs
    ns = [no, nm, nb, ng]
    deltao, deltam, deltab, deltag = [np.zeros(n, dtype=np.uint64) for n in ns]

    # 1. Bait k-mers. We expect to find them in Ossabaw.
    # If not, they are a "loss", 
    # unless we also do not find them in the minipig.
    hall = (ho, hm, hb, hg)
    htall = tuple(h.hashtable for h in hall)
    # TODO: lambda functions should be command line parameters!
    bait_loss = njit(lambda vo, vm, vb, vg: vo<100 and vm>=36 and vb>=1 and vg>=1)
    diff_loss = compile_filter(hb, hall, bait_loss, "bait_loss")
    n_loss = diff_loss(hb.hashtable, htall)
    ossabaw_gain = njit(lambda vo, vm, vb, vg: vo>=300 and vm<=3 and vb==0 and vg==0)
    diff_gain = compile_filter(ho, hall, ossabaw_gain, "ossabaw_gain")
    n_gain = diff_gain(ho.hashtable, htall)


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
