"""
call_variants: call variants from candidates produced by gainloss.py
Jens Zentgraf & Sven Rahmann, 2022
"""

"""
VCF Format:
#CHROM POS ID REF ALT QUAL FILTER INFO [FORMAT] Sample1 Sample2 ...

Process gene sequence FASTA, translate coordinates to CHROM,POS.
Leave ID empty.
Determine REF and ALT
QUAL = high value (50), FILTER = PASS
INFO ?
FORMAT? leave out
samples? leave out
"""

from argparse import ArgumentParser
from importlib import import_module

import numpy as np
from numba import njit, uint64, int64, uint32

from fastcash.io.fastaio import all_fasta_seqs
from fastcash.kmers import compile_kmer_iterator
from fastcash.hashio import load_hash



def hash_candidates(fname):
    losses = dict(); gains = dict()
    with open(fname, "rt") as fc:
        for line in fc:
            line = line.strip()
            if not len(line): continue
            fields = line.split()
            kmer, co, cm, cb, cg = map(int, fields[:5])
            typ = fields[5]
            if typ == "bait_loss":
                assert cb == 2 and cg == 1
                losses[kmer] = (co, cm)
            elif typ == "ossabaw_gain":
                assert cb == 0 and cg == 0
                gains[kmer] = (co, cm)
            else:
                raise RuntimeError(f"unrecognized type {typ} in file {fname}")
    return losses, gains


@njit(nogil=True, locals=dict(
    elem1=uint64, elem2=uint64, h=uint64, mask=uint64, result=uint32))
def _hamming_dist(elem1, elem2):
    mask = uint64(6148914691236517205)  # Mask for ....01010101 for uint64
    h = elem1 ^ elem2
    h = (h | (h >> 1)) & mask
    h = h & uint64(h - 1)
    if h == uint64(0): return uint32(1)
    h = h & uint64(h - 1)
    if h == uint64(0): return uint32(2)
    h = h & uint64(h - 1)
    if h == uint64(0): return uint32(65)
    return uint32(65)


@njit(nogil=True, locals=dict(kmer=uint64, other=uint64, p=int64))
def _find_partners(kmers, others):
    partners = np.zeros(kmers.size, dtype=int64)
    for i, kmer in enumerate(kmers):
        p = -1
        n = 0
        for other in others:
            if _hamming_dist(kmer, other) <= 64:
                n += 1
                p = int64(other)
        if n <= 1:
            partners[i] = p
        else:
            partners[i] = -2
    return partners


def find_partners(losses, gains):
    print("# finding partners for losses...")
    n = len(losses)
    aloss = np.array(list(losses.keys()), dtype=np.uint64)
    again = np.array(list(gains.keys()), dtype=np.uint64)
    apartners = _find_partners(aloss, again)
    partners = {int(l): int(g) for l, g in zip(aloss, apartners)}
    n0 = np.sum(apartners==-1)
    n2 = np.sum(apartners==-2)
    n1 = n - n0 - n2
    print(f"# {n} losses have gain partners: none: {n0}, many: {n2}, unique: {n1} ({n1/n:.1%})")
    print()
    return partners


def examine_losses(fasta, losses, gains, k, genomecounts, losspartners):
    value_from_name = lambda header, direction: 1
    both = False; skipvalue = -1
    _, kmers = compile_kmer_iterator(k, rcmode="max")
    get_value = genomecounts.get_value
    ht = genomecounts.hashtable
    arr_gains = np.array(list(gains.keys()), dtype=np.uint64)
    Tlost = 0
    for (hd, sq, _, _) in all_fasta_seqs(fasta, value_from_name, both, skipvalue):
        lost = unique = partners = 0
        for kmer in kmers(sq, 0, sq.size):
            assert kmer not in gains
            v = get_value(ht, kmer)
            assert v >= 1
            if v == 1:
                unique += 1
                if kmer in losses:
                    lost += 1
                    partners += (losspartners[kmer] >= 0)
            else:
                assert kmer not in losses
        total = sq.size - k + 1
        print(f'\n{hd.decode("ASCII")}: length {sq.size}')
        print(f"unique k-mers: {unique} of {total} ({unique/total:.1%})")
        print(f"lost k-mers: {lost} of {unique} uniques ({lost/unique:.1%}, or {lost/total:.1%} of all)")
        print(f"with gain partners (HD2): {partners} of {lost} ({partners/max(lost,1):.1%})")
        Tlost += lost
    print(f"\nTotal lost: {Tlost} / {len(losses)}")


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



def main(args):
    k = args.k
    losses, gains = hash_candidates(args.candidates)
    losspartners = find_partners(losses, gains)
    gc, values, info, hist, n = load_table(args.genomecounts, "genomic k-mer counts")    
    examine_losses([args.genes], losses, gains, k, gc, losspartners)


def get_argument_parser():
    p = ArgumentParser(description="Call variants from candidates (from gainloss.py)")
    p.add_argument("-k", type=int, required=True,
        help="k-mer size k")
    p.add_argument("candidates",
        help="text file with candidates (produced by gainloss.py)")
    p.add_argument("genes",
        help="FASTA file with gene sequences, to examine losses")
    p.add_argument("genomecounts",
        help="H5 file with genomic k-mer counts")
    return p


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)

