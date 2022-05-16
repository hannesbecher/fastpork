"""
genomewide_analysis.py: 

Scan the reference genome;
see which k-mers are present in Ossabaw resp. minipigs
(and which of these k-mers are unique).

Construct bit vectors across each chromosome:

uint8[:nkmers]:
| kmer-paircount (4+4) | 
uint8[:n]
| #kmers covering position in minipig (5) |
uint8[:n]:
| #kmers covering position in ossabaw (5) |
uint8[:n|:nkmers]
| k-mer unique? (1) | nucleotide (3) |

Save the vectors in a zarr directory, grouped by chromosome.
Sven Rahmann, 2022
"""

UNIQ = 8


from argparse import ArgumentParser
from sys import stdout
from collections import Counter
import re

import numpy as np
from numba import njit, uint64, int64, uint8, uint32

from fastcash.mathutils import bitsfor
from fastcash.dnaencode import dna_to_2bits
from fastcash.io.fastaio import fasta_reads
from fastcash.hashio import load_hash, save_hash
from fastcash.kmers import compile_positional_kmer_processor

from zarrutils import save_to_zgroup


def get_chromosome(name):
    m = re.search(r'chromosome (\S+), Sscrofa11.1,', name)
    if m is not None:
        return m.group(1)
    if 'Sus scrofa mitochondrion, complete genome' in name:
        return 'MT'
    return None


def check_consistency(info1, info2):
    k1 = int(info1['k'])
    k2 = int(info1['k'])
    if k1 != k2:
        raise RuntimeError(f"ERROR: k values differ: {k1} != {k2}")
    rcmode1 = info1.get('rcmode')
    rcmode2 = info2.get('rcmode')
    if rcmode1 != rcmode2:
        raise RuntimeError(f"ERROR: rcmodes values differ: {rcmode1} != {rcmode2}")
    return k1, rcmode1


def compile_get_hist2d(hout, gs1, gs2):
    bits1, bits2 = bitsfor(gs1+1), bitsfor(gs2+1)
    mask1 = uint64(2**bits1 - 1)
    mask2 = uint64(2**bits2 - 1)
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
                    v1 = val & mask1
                    v2 = (val >> bits1) & mask2
                    assert v1 <= gs1
                    assert v2 <= gs2
                    hist[v1,v2] += 1
        total = hist.sum()
        m = gs1
        while np.sum(hist[m,:])==0: m -= 1
        n = gs2
        while np.sum(hist[:,n])==0: n -= 1
        return hist[:m+1, :n+1]

    return get_hist2d


def print_2d_histogram(hist, title=None, percent=False):
    if title is not None:
        print(title)
    m, n = hist.shape
    h = hist if not percent else np.round(100*hist.astype(np.float64)/hist.sum(), 2)
    fmt = '10d' if not percent else '10.2f'
    print("|   | ", end="")
    for j in range(n):
        print(f"{j:10d} | ", end="")
    print()
    for i in range(m):
        print(f"| {i} | ", end="")
        for j in range(n):
            print(f"{h[i,j]:{fmt}} | ", end="")
        print()


def compile_analyze_sequence(
        ha, hg, k, rcmode, bits_ossabaw, bits_minipig,
        T_ossabaw, T_minipig):
    """
    uint8[:nkmers]:
        | kmer-paircount (4+4) | 
    uint8[:n]:
        | #kmers covering position in minipig (5) |
    uint8[:n]:
        | #kmers covering position in ossabaw (5) |
    uint8[:n|:nkmers]:
        | k-mer unique? (1) | nucleotide (3) |
    """
    assert bits_ossabaw + bits_minipig <= 8, "too many paircount bits"
    ha_get_value = ha.get_value
    hg_get_value = hg.get_value
    mask_ossabaw = uint64((2 ** bits_ossabaw) - 1)
    mask_minipig = uint64((2 ** bits_minipig) - 1)

    @njit(locals=dict(
        va=uint64, voss=uint64, vmin=uint64, vg=uint64))
    def process_one_kmer(hta, code, position, 
            htg, res_pc, res_cov_m, res_cov_o, res_seq):
        # uses the interface of positional_kmer_processor!

        # compute k-mer result
        va = ha_get_value(hta, code)
        res_pc[position] = uint8(va)
        voss = va & mask_ossabaw
        vmin = uint64(va >> bits_ossabaw) & mask_minipig
        vg = hg_get_value(htg, code)
        if vg == 1:
            res_seq[position] |= uint8(UNIQ)

        # compute sequence result
        if voss >= T_ossabaw:
            res_cov_o[position:position+k] += uint8(1)
        if vmin >= T_minipig:
            res_cov_m[position:position+k] += uint8(1)
        return False  # False == no failure

    print(f"# Compiling positonal k-mer processor with {k=} and {rcmode=}; {bits_ossabaw=}, {bits_minipig=}, {T_ossabaw=}, {T_minipig=}...")
    _, process_kmers = compile_positional_kmer_processor(
        k, process_one_kmer, rcmode=rcmode)

    @njit(locals=dict())
    def analyze_sequence(seq, hta, htg, 
            res_pc, res_cov_m, res_cov_o, res_seq):
        res_pc[:] = 0
        res_cov_m[:] = 0
        res_cov_o[:] = 0
        res_seq[:] = seq[:] & 7
        process_kmers(hta, seq, 0, len(seq), htg,
            res_pc, res_cov_m, res_cov_o, res_seq)

    return analyze_sequence



def get_argument_parser():
    p = ArgumentParser(description="Compare Sus scrofa genome against Minipig/Ossabaw k-mer sets")
    p.add_argument("--aggregated", "-a", metavar="AGGREGATED_H5", required=True,
        help="hash table with aggregated pair counts (Ossabaw/minipig)")
    p.add_argument("--genomic", "-g", metavar="GENOMIC_H5", required=True,
        help="hash table with genomic k-mer counts (Sus scrofa)")
    p.add_argument("--fasta", "-f", metavar="GENOMIC_FASTA", required=True,
        help="FASTA file with Sus scrofa genome")
    p.add_argument("--out", "-o", metavar="OUTPUT_ZARR", required=True,
        help="output: ZARR path with chromosome-wise results")

    p.add_argument("--TOssabaw", type=int, default=3,
        help="required number of Ossabaw individual genomes containing a k-mer [3]")
    p.add_argument("--TMinipig", type=int, default=3,
        help="required number of Minipig individual genomes containing a k-mer [3]")
    return p


def main(args):
    T_ossabaw = args.TOssabaw
    T_minipig = args.TMinipig

    # Load hash tables
    print("# Loading aggregated ossabaw/minipig counts:")
    stdout.flush()
    ha, valuesa, infoa = load_hash(args.aggregated)
    print("# Collecting statistics...")
    stdout.flush()
    bits_ossabaw = int(valuesa.bits1)
    bits_minipig = int(valuesa.bits2)
    ubossabaw = 2 ** bits_ossabaw - 1
    ubminipig = 2 ** bits_minipig - 1
    get_hist2d = compile_get_hist2d(ha, ubossabaw, ubminipig)
    hist = get_hist2d(ha.hashtable)
    print_2d_histogram(hist, title="Number of k-mers in Ossabaw/minipig individual pigs")
    print_2d_histogram(hist, title="Percent of k-mers in Ossabaw/minipig individual pigs", percent=True)

    print("# Loading sus scrofa genomic k-mer counts:")
    stdout.flush()    
    hg, valuesg, infog = load_hash(args.genomic)
    k, rcmode = check_consistency(infoa, infog)

    print("# Processing sus scrofa genomic FASTA:")
    analyze_sequence = compile_analyze_sequence(
        ha, hg, k, rcmode, bits_ossabaw, bits_minipig,
        T_ossabaw, T_minipig)

    # initialize result arrays
    result_seq = np.zeros(1, dtype=np.uint8)
    result_cov_oss = np.zeros(1, dtype=np.uint8)
    result_cov_min = np.zeros(1, dtype=np.uint8)
    result_paircount = np.zeros(1, dtype=np.uint8)

    outfile = args.out
    for (header, seq) in fasta_reads(args.fasta):
        name = header.decode('ASCII')
        n = len(seq)
        ch = get_chromosome(name)
        if ch is None:
            if 'unplaced' in name or 'unlocalized' in name: continue
            raise RuntimeError(f"Unrecognized chromosomze in '{name}'")
        print(f"# Processing chromosome {ch} of length {n} ('{name}')")
        stdout.flush()
        seq = dna_to_2bits(seq)
        nk = n - k + 1
        if n > result_seq.size:
            print(f"# (Re-)allocating sequence status buffer to capture {n} bytes")
            result_seq = np.zeros(n, dtype=np.uint8)
            result_cov_oss = np.zeros(n, dtype=np.uint8)
            result_cov_min = np.zeros(n, dtype=np.uint8)
            result_paircount = np.zeros(nk, dtype=np.uint8)
        r_seq, r_co, r_cm, r_pc = result_seq[:n], result_cov_oss[:n], result_cov_min[:n], result_paircount[:nk]
        analyze_sequence(seq, ha.hashtable, hg.hashtable,
            r_pc, r_cm, r_co, r_seq)
        ru = ((r_seq & UNIQ) == UNIQ).sum()
        print(f"#    Unique: {ru} / {nk} = {ru/nk:.1%}")
        print(f"#    Ossabaw: ", Counter(r_co))
        print(f"#    Minipig: ", Counter(r_cm))
        stdout.flush()
        save_to_zgroup(outfile, f'/{ch}',
            seq=r_seq,
            coverage_ossabaw=r_co,
            coverage_minipig=r_cm,
            paircount=r_pc,
        )


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
