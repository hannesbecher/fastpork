"""
fastpork/count.py:
count k-mers in FASTQ file(s) using either
- a standard hashtable (serial)
- a parallel hash table with subtables

Hint: unzip gzipped fastqs with '<(pigz -cd -p2 sample.fq.gz)''
"""

import sys
from math import ceil
import datetime
from os import cpu_count
from importlib import import_module
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numba import njit, uint8, uint32, uint64, int64, boolean, prange

from .mathutils import print_histogram, print_histogram_tail
from .srhash import get_npages
from .hashio import save_hash
from .parameters import get_valueset_parameters, parse_parameters
from .builders import build_from_fastq, build_from_fasta
from .builders_subtables import parallel_build_from_fastq, parallel_build_from_fasta


DEFAULT_HASHTYPE = "3c_fbcbvb"

# build index #########################################

def build_new_index(args):
    # obtain the parameters
    P = get_valueset_parameters(args.valueset, k=args.kmersize, rcmode=args.rcmode)
    (values, valuestr, rcmode, k, parameters) = P
    if not isinstance(k, int):
        print(f"Error: k-mer size k not given; k={k}")
        sys.exit(1)
    print(f"# Imported value set '{valuestr}'.")
    print(f"# Dataset parameters: {parameters}")
    parameters = parse_parameters(parameters, args)
    print(f"# Updated parameters: {parameters}")
    (nobjects, hashtype, aligned, hashfuncs, pagesize, nfingerprints, fill) = parameters

    # create the hash table
    subtables = args.subtables
    if hashtype == "default":
        hashtype = DEFAULT_HASHTYPE
    if subtables:
        if not hashtype.startswith("s"):
            hashtype = "s" + hashtype  # make it a subtable hash type
    else:
        if hashtype.startswith("s"):
            hashtype = hashtype[1:]  # make it a non-subtable hash type

    hashmodule = import_module(".hash_" + hashtype, __package__)
    print(f"# Use hash type {hashtype}")
    build_hash = hashmodule.build_hash
    universe = int(4**k)
    nvalues = values.NVALUES
    update_value = values.update
    n = get_npages(nobjects, pagesize, fill) * pagesize
    print(f"# Allocating hash table for {n} objects, functions '{hashfuncs}'...")
    h = build_hash(universe, n, subtables, pagesize,
        hashfuncs, nvalues, update_value,
        aligned=aligned, nfingerprints=nfingerprints,
        maxwalk=args.maxwalk, shortcutbits=args.shortcutbits)
    print(f"# Memory for hash table: {h.mem_bytes/(2**20):.3f} MB")
    print(f"# Info:  rcmode={rcmode}, walkseed={args.walkseed}")
    return (h, values, valuestr, k, rcmode)


# main #########################################

def main(args):
    starttime = datetime.datetime.now()
    print(f"# {starttime:%Y-%m-%d %H:%M:%S}: fastpork count: will create '{args.out}'.")
    # create new hash table
    print(f"# building a new index (not loading an existing one)...")
    (h, values, valuestr, k, rcmode) = build_new_index(args)
    assert type(rcmode) == str

    source = f"FASTQ(s): {args.fastq};  FASTA(s): {args.fasta}"
    print(f"# {starttime:%Y-%m-%d %H:%M:%S}: Using source {source}.")
    bufsize = int(ceil(args.chunksize * 2**20))
    chunkreads = args.chunkreads if args.chunkreads is not None \
        else bufsize // 200
    print(f"# FASTQ chunksize: {args.chunksize} MB;  max reads per chunk: {chunkreads}")
    
    if args.fastq is not None:
        if h.subtables:
            total, failed, walkstats \
                = parallel_build_from_fastq(args.fastq, k, h, (1,1),
                        subsample=args.subsample, maxfailures=args.maxfailures,
                        rcmode=rcmode, fqbufsize=bufsize, chunkreads=chunkreads)
        else:
            total, failed, walkstats \
                = build_from_fastq(args.fastq, k, h, (1,1),
                        subsample=args.subsample, maxfailures=args.maxfailures,
                        rcmode=rcmode, bufsize=bufsize, chunkreads=chunkreads)
    elif args.fasta is not None:
        if h.subtables:
            total, failed, walkstats \
                 = parallel_build_from_fasta(args.fasta, k, h, lambda x: 1, maxfailures=args.maxfailures,
                       rcmode=rcmode, fabufsize=bufsize, chunkreads=chunkreads)
        else:
            total, failed, walkstats \
                 = build_from_fasta(args.fasta, k, h, lambda name,i: 1, maxfailures=args.maxfailures,
                       rcmode=rcmode)
    else:
        raise RuntimeError("need --fastq or --fasta argument")
    endtime = datetime.datetime.now()
    elapsed = (endtime - starttime).total_seconds()
    print(f"time sec: {elapsed:.1f}")
    print(f"time min: {elapsed/60:.3f}")
    print()
    
    # calculate shortcut bits
    if args.shortcutbits > 0:
        startshort = datetime.datetime.now()
        print(f'# {startshort:%Y-%m-%d %H:%M:%S}: Begin calculating shortcut bits ({args.shortcutbits})...')
        h.compute_shortcut_bits(h.hashtable)
        now = datetime.datetime.now()
        elapsed = (now-startshort).total_seconds()
        print(f'# {now:%Y-%m-%d %H:%M:%S}: Done calculating shortcut bits after {elapsed:.2f} sec.')

    outname = args.out
    now = datetime.datetime.now()
    if failed == 0:
        print(f"# {now:%Y-%m-%d %H:%M:%S}: SUCCESS, processed {total} k-mers.")
        print(f"# writing output file '{outname}'...")
        save_hash(outname, h, valuestr,
            additional=dict(k=k, subtables=args.subtables, walkseed=args.walkseed, rcmode=rcmode, fillrate=args.fill, valueset=valuestr))
    else:
        print(f"# {now:%Y-%m-%d %H:%M:%S}: FAILED for {failed}/{total} processed k-mers.")
        print(f"# output file '{outname}'' will NOT be written.")

    show_statistics = not args.nostatistics
    if show_statistics:
        now = datetime.datetime.now()
        print(f"# {now:%Y-%m-%d %H:%M:%S}: collecting statistics...")
        print()
        valuehist, fillhist, choicehist, shortcuthist = h.get_statistics(h.hashtable)
        print_histogram_tail(walkstats, [1,2,10], title=f"Extreme walk lengths:", shorttitle="walk", average=True)
        if args.substatistics:
            for i in range(args.subtables):
                print(f"# Statistics for subtable {i}")
                print_histogram(valuehist[i], title="Value statistics:", shorttitle="values", fractions="%")
                print_histogram(fillhist[i], title="Page fill statistics:", shorttitle="fill", fractions="%", average=True, nonzerofrac=True)
                print_histogram(choicehist[i], title="Choice statistics:", shorttitle="choice", fractions="%+", average="+")
                if args.shortcutbits > 0:
                    print_histogram(shortcuthist, title="shortcut bits statistics:", shorttitle="shortcutbits", fractions="%")
        print("# Combined statsitcs for all subtables")
        print_histogram(np.sum(valuehist, axis=0), title="Value statistics:", shorttitle="values", fractions="%")
        print_histogram(np.sum(fillhist, axis=0), title="Page fill statistics:", shorttitle="fill", fractions="%", average=True, nonzerofrac=True)
        print_histogram(np.sum(choicehist, axis=0), title="Choice statistics:", shorttitle="choice", fractions="%+", average="+")
    endtime = datetime.datetime.now()
    elapsed = (endtime - starttime).total_seconds()
    print(f"time sec: {elapsed:.1f}")
    print(f"time min: {elapsed/60:.3f}")
    print()
    print(f"{endtime:%Y-%m-%d %H:%M:%S}: Done.")
    if failed:
        print("FAILED! see above messages!")
        sys.exit(1)
