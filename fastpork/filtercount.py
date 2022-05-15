"""
fastpork/filtercount.py:
count k-mers in FASTQ file(s) with a pre-filter (or two/three)

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

from ..mathutils import print_histogram, print_histogram_tail
from ..srhash import get_npages
from ..hashio import save_hash
from ..parameters import get_valueset_parameters, parse_parameters
from ..builders import build_from_fastq, build_from_fasta, build_from_fastq_filter
from ..builders_subtables import parallel_build_from_fastq, parallel_build_from_fasta
from ..builders_subtables_filter import parallel_build_from_fastq_filter
from ..filter_b3c import build_filter
from ..lowlevel.debug import define_debugfunctions
debugprint, timestamp = define_debugfunctions(debug=True, times=True)


DEFAULT_HASHTYPE = "3c_fbcbvb"

# build index #########################################

def build_new_index(args):
    # obtain the parameters
    P = get_valueset_parameters(args.valueset, k=args.kmersize, rcmode=args.rcmode)
    (values, valuestr, rcmode, k, parameters) = P
    if not isinstance(k, int):
        print(f"Error: k-mer size k not given; k={k}")
        sys.exit(1)
    debugprint(f"# Imported value set '{valuestr}'.")
    debugprint(f"# Dataset parameters: {parameters}")
    parameters = parse_parameters(parameters, args)
    debugprint(f"# Updated parameters: {parameters}")
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

    hashmodule = import_module("..hash_" + hashtype, __package__)
    debugprint(f"# Use hash type {hashtype}")
    build_hash = hashmodule.build_hash
    universe = int(4**k)
    nvalues = values.NVALUES
    update_value = values.update
    n = get_npages(nobjects, pagesize, fill) * pagesize
    debugprint(f"# Allocating hash table for {n} objects, functions '{hashfuncs}'...")
    h = build_hash(universe, n, subtables, pagesize,
        hashfuncs, nvalues, update_value,
        aligned=aligned, nfingerprints=nfingerprints,
        maxwalk=args.maxwalk, shortcutbits=args.shortcutbits)
    debugprint(f"# Memory for hash table: {h.mem_bytes/(2**20):.3f} MB")
    debugprint(f"# Info:  rcmode={rcmode}, walkseed={args.walkseed}")

    if args.filtersize1 is not None and args.filtersize1 <= 0.0:
        args.filter = args.filtersize1 = None
    if args.filtersize2 is not None and args.filtersize2 <= 0.0:
        args.filtersize2 = None
    if args.filtersize3 is not None and args.filtersize <= 0.0:
        args.filtersize3 = None
    if args.filter is not None:
        f1 = build_filter(k, universe, args.filtersize1, max(subtables, 1), hashfuncs)
        if args.filtersize2 is not None:
            f2 = build_filter(k, universe, args.filtersize2, max(subtables, 1), hashfuncs)
        else:
            f2 = None
        if args.filtersize3 is not None:
            f3 = build_filter(k, universe, args.filtersize3, max(subtables, 1), hashfuncs)
        else:
            f3 = None
    else:
        f1 = f2 = f3 = None
    return (h, f1, f2, f3, values, valuestr, k, rcmode)


# main #########################################

def main(args):
    starttime = timestamp(msg=f"fastpork filtercount: will create '{args.out}'.")
    # create new hash table
    debugprint(f"# building a new index (not loading an existing one)...")
    (h, fltr1, fltr2, fltr3, values, valuestr, k, rcmode) \
        = build_new_index(args)
    assert type(rcmode) == str

    source = f"FASTQ(s): {args.fastq};  FASTA(s): {args.fasta}"
    timestamp(msg=f"Using source {source}.")
    bufsize = int(ceil(args.chunksize * 2**20))
    chunkreads = args.chunkreads if args.chunkreads is not None \
        else bufsize // 200
    debugprint(f"# FASTQ chunksize: {args.chunksize} MB;  max reads per chunk: {chunkreads}")
    maxfailures = args.maxfailures

    if args.fastq is not None:
        if h.subtables:
            total, failed, walkstats \
                = parallel_build_from_fastq_filter(
                    args.filter, k, h, (0,0),
                    fltr1=fltr1, fltr2=fltr2, fltr3=fltr3, 
                    subsample=args.subsample, rcmode=rcmode, 
                    fqbufsize=bufsize, chunkreads=chunkreads,
                    maxfailures=maxfailures)

            timestamp(starttime, msg="pass 1 time sec")
            timestamp(starttime, msg="pass 1 time min", minutes=True)
            fill_level_1 = fltr1.get_fill_level(fltr1.filter_array) if fltr1 is not None else 0.0
            fill_level_2 = fltr2.get_fill_level(fltr2.filter_array) if fltr2 is not None else 0.0
            fill_level_3 = fltr3.get_fill_level(fltr3.filter_array) if fltr3 is not None else 0.0
            timestamp(msg=f"filter fill levels: {fill_level_1:.4f}, {fill_level_2:.4f}, {fill_level_3:.4f}")
            debugprint()

            starttime2 = timestamp()
            if failed <= maxfailures:
                total2, failed2, walkstats2 \
                    = parallel_build_from_fastq_filter(
                        args.fastq, k, h, (1,1),
                        update=h.private.update_existing_ssk,
                        subsample=args.subsample, rcmode=rcmode, 
                        fqbufsize=bufsize, chunkreads=chunkreads,
                        maxfailures=total)

        else:  # not parallel (no subtables)
            total, failed, walkstats \
                = build_from_fastq_filter(
                    args.filter, k, h, (0,0), 
                    fltr1=fltr1, fltr2=fltr2, fltr3=fltr3,
                    subsample=args.subsample, rcmode=rcmode,
                    bufsize=bufsize, chunkreads=chunkreads,
                    maxfailures=maxfailures)

            timestamp(starttime, msg="pass 1 time sec")
            timestamp(starttime, msg="pass 1 time min", minutes=True)
            fill_level_1 = fltr1.get_fill_level(fltr1.filter_array) if fltr1 is not None else 0.0
            fill_level_2 = fltr2.get_fill_level(fltr2.filter_array) if fltr2 is not None else 0.0
            fill_level_3 = fltr3.get_fill_level(fltr3.filter_array) if fltr3 is not None else 0.0
            timestamp(msg=f"filter fill levels: {fill_level_1:.4f}, {fill_level_2:.4f}, {fill_level_3:.4f}")
            debugprint()

            starttime2 = timestamp()
            if failed <= maxfailures:
                total2, failed2, walkstats2 \
                    = build_from_fastq(args.fastq, k, h, (1,1),
                            update=h.update_existing, subsample=args.subsample,
                            rcmode=rcmode, bufsize=bufsize, chunkreads=chunkreads,
                            maxfailures=total)

    elif args.fasta is not None:
        if args.filter or args.filtersize1 or args.filtersize2 or args.filtersize3:
            print("# filters are only supported for fastq files")

        starttime2 = timestamp()
        if h.subtables:
            total, failed, walkstats \
                 = parallel_build_from_fasta(args.fasta, k, h, lambda x: 1,
                       rcmode=rcmode, fabufsize=bufsize, chunkreads=chunkreads)
        else:
            total, failed, walkstats \
                 = build_from_fasta(args.fasta, k, h, lambda name,i: 1,
                       rcmode=rcmode)
    else:
        raise RuntimeError("need --fastq or --fasta argument")

    # Output time
    timestamp(starttime2, msg="pass 2 time sec")
    timestamp(starttime2, msg="pass 2 time min", minutes=True)
    debugprint()
    
    # calculate shortcut bits
    if args.shortcutbits > 0:
        startshort = timestamp(msg=f'Begin calculating shortcut bits ({args.shortcutbits})...')
        h.compute_shortcut_bits(h.hashtable)
        timestamp(msg=f'Done calculating shortcut bits.')
        timestamp(startshort, msg='shortcutbits time sec')

    outname = args.out
    now = datetime.datetime.now()
    if failed == 0:
        timestamp(msg=f"SUCCESS, processed {total} k-mers.")
        debugprint(f"# writing output file '{outname}'...")
        save_hash(outname, h, valuestr,
            additional=dict(k=k, subtables=args.subtables, walkseed=args.walkseed, rcmode=rcmode, fillrate=args.fill, valueset=valuestr))
    else:
        timestamp(msg=f"FAILED for {failed}/{total} processed k-mers.")
        debugprint(f"# output file '{outname}' will NOT be written.")

    show_statistics = not args.nostatistics
    if show_statistics:
        timestamp(msg=f"collecting statistics...\n")
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

    timestamp(msg="Done.")
    timestamp(starttime, msg="TOTAL time sec")
    timestamp(starttime, msg="TOTAL time min", minutes=True)
    if failed:
        print("FAILED! see above messages!")
        sys.exit(1)
