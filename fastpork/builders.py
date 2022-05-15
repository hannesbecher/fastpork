"""
fastcash/builders.py

Utilities to build a hash table from a data source,
e.g. FASTA or FASTQ

build_from_fasta():
    Fill a hash table with the k-mers and values from FASTA files
verify_from_fasta():
    Check that a hash table is correctly filled with all k-mers from FASTA files
build_from_fastq():
    Fill a hash table with the k-mers and values from FASTQ files
"""

import numpy as np
from numpy.random import seed as randomseed
from numba import njit, uint8, void, int64, uint64, int32, boolean

from .io.fastaio import fasta_reads, all_fasta_seqs
from .io.fastqio import fastq_chunks
from .dnaencode import (
    generate_revcomp_and_canonical_code,
    quick_dna_to_2bits,
    )
from .kmers import compile_kmer_iterator, compile_kmer_processor


# build from FASTA #####################################

def build_from_fasta(
    fastas,  # list of FASTA files
    shp,  # k-mer size or shape
    h,  # hash data structure, pre-allocated, to be filled (h.update)
    value_from_name,  # function mapping FASTA entry names to numeric values
    func = None,
    *func_params,
    rcmode="min",  # from 'f', 'r', 'both', 'min', 'max'
    skipvalue=-1,  # value that skips a FASTA entry
    walkseed=7,
    maxfailures=0,
    ):
    """
    Build (fill) pre-allocated (and correctly sized) hash table 'h'
    with 'k'-mers from FASTA files 'fastas'.

    Each entry from each FASTA file is processed sequentially.
    The name of the entry (after '>' up to the first space or '_') is converted
    into a numeric value using function value_from_name(name, 1) 
    and/or value_from_name(name, 2), depending on 'rcmode'.
    Each k-mer (and/or reverse complementary k-mer) of the entry is inserted into 'h'
    with the computed value.
    If the k-mer is already present, its value is updated accordint to h's value update policy.

    rcmode has the following effect:
    rcmode=='f': insert k-mers as they appear in the file using value_from_name(name, 1)
    rcmode=='r': insert reverse complementary k-mers using value_from_name(name, 2)
    rcmode=='both': insert both k-mers using two different values
    rcmode=='min': insert the smaller of the two k-mers using value_from_name(name, 1)
    rcmode=='max': insert the larger of the two k-mers using value_from_name(name, 1)

    Return (total, failed, walkstats), where:
      total is the total number of valid k-mers read, 
      failed is the number of k-mers unsuccessfully processed,
      walkstats is an array indexed 0 to h.maxwalk+slack, counting walk lengths
    """
    print(f"# Building from FASTA, using rcmode={rcmode}, maxfailures={maxfailures}")
    update = h.update
    
    @njit(nogil=True, locals=dict(
        st=int32, result=uint64, vx=uint64))
    def store_in_table(ht, code, state, walkstats):
        # state: [vsum, vx, total, failures]
        breakout = False
        vx = state[0] - state[1]  # vsum - vx
        state[2] += 1  # total
        st, result = update(ht, code, vx)  # result is (choice:int32, steps/value:uint64)
        if st & 128:  # existing
            walkstats[uint64(st&127)] += 1
        else:
            walkstats[result] += 1
        if st == 0:
            state[3] += 1  # failures
            if maxfailures >= 0 and state[3] > maxfailures:
                breakout = True
        state[1] = state[0] - state[1]  # vx = vsum - vx
        return breakout  # flag: finally FAILED if nonzero

    if func == None:
        func = store_in_table
    
    k, process_kmers = compile_kmer_processor(shp, func, rcmode)

    # because the supplied function 'store_in_table' has TWO extra parameters (state, walkstats),
    # the generated function 'process_kmers' also gets the same TWO extra parameters.

    assert 4**k == h.universe, f"Error: k={k}; 4**k={4**k}; but universe={h.universe}"
    
    revcomp, ccode = generate_revcomp_and_canonical_code(k, rcmode)
    codemask = uint64(4**(k-1) - 1)

    @njit( ###__signature__ void(),
        nogil=True)
    def set_seed():
        randomseed(walkseed)

    @njit(nogil=True,
        locals=dict(total=int64, fail=int64, v1=uint64, v2=uint64, vsum=uint64))
    def add_kmers(ht, seq, start, end, v1, v2, state, walkstats, *flter):
        vsum = v1 + v2
        state[0:3] = (vsum, v1, uint64(0))  # status[3]==fail is kept
        process_kmers(ht, seq, start, end, state, walkstats, *flter)
        total, fail = state[2], state[3]
        return (total, fail)  # walkstats has been modified, too!
    
    set_seed()
    ht = h.hashtable
    both = (rcmode=="both")
    total = fail = 0
    walkstats = np.zeros(h.maxwalk+5, dtype=np.uint64)
    state = np.zeros(4, dtype=np.uint64)
    for (_, sq, v1, v2) in all_fasta_seqs(
            fastas, value_from_name, both, skipvalue):
        (dtotal, fail) = add_kmers(ht, sq, 0, len(sq), v1, v2, state, walkstats)
        total += dtotal
        if maxfailures >= 0 and fail > maxfailures: 
            break
    # done; hashtable h is now filled; return statistics
    return (total, fail, walkstats)


# verify from FASTA #####################################

def verify_from_fasta(
    fastas,  # list of FASTA files
    shp,     # k-mer shape or size
    h,       # populated hash data structure to be checked (h.get_value)
    value_from_name,      # function mapping FASTA entry names to numeric values
    value_is_compatible,  # function checking whether observed value 
    *,                    # is compatible with stored value (observed, stored)
    rcmode="min",  # from {'f', 'r', 'both', 'min', 'max'}
    skipvalue=-1,   # value that skips a FASTA entry
    ):
    """
    Verify that all k-mers from FASTA files 'fastas'
    are correctly represented in hash table 'h' (present and compatible value).    with 'k'-mers from FASTA files 'fastas'.
    
    For name-to-value conversion, see build_from_fasta().
    For rcmode options {'f', 'r', 'both', 'min', 'max'}, see build_from_fasta().

    Return (ok, kmer, fasta_value, stored_value) om failure;
    return (ok, nsequences, -1, -1) on success, where
    - ok is the number of successfully verified k-mers before failure
    - kmer is the kmer encoding
    - fasta_value, stored_value are the incompatible values >= 0,
    - nsequences is the number of successfully processed sequences.
    """
    print(f"# Verifying from FASTA, using rcmode={rcmode}")
    k, kmers = generate_kmer_iterator(shp, rcmode)
    assert 4**k == h.universe, f"Error: k={k}; 4**k={4**k}; but universe={h.universe}"
    get_value = h.get_value

    @njit( ###__signature__ (uint64[:], uint8[:], int64, int64, int64, int64),
        nogil=True, locals=dict(
            code=uint64, v=int64, v1=int64, v2=int64, vx=int64, vsum=int64))
    def check_kmers(ht, seq, start, end, v1, v2):
        ok = 0
        vsum = v1 + v2
        vx = v1
        for code in kmers(seq, start, end):
            v = get_value(ht, code)
            if not value_is_compatible(vx, v): # (observed, stored)
                return (ok, code, v, vx)  # (stored, observed)
            ok += 1
            vx = vsum - vx
        return (ok, 0, -1, -1)

    ht = h.hashtable
    both = (rcmode == "both")
    ok = 0
    for (i,(_, sq, v1, v2)) in enumerate(all_fasta_seqs(
            fastas, value_from_name, both, skipvalue, progress=True)):
        ##print(i, len(sq), v1, v2, ht.shape)
        (dok, key, value, expected) = check_kmers(ht, sq, 0, len(sq), v1, v2)
        ok += int(dok)
        if value != -1:
            return (ok, int(key), int(value), int(expected))
    nsequences = i+1
    return (ok, nsequences, -1, -1)


# FASTQ #####################################

def build_from_fastq_filter(
    fastqs,  # list of FASTQ files
    shp,  # k-mer size or shape
    h,  # hash data structure, pre-allocated, to be filled
    values, # pair of values for indexing
    *,
    fltr1=None,
    fltr2=None,
    fltr3=None,
    update=None,
    subsample=1,
    rcmode="min",  # from 'f', 'r', 'both', 'min', 'max'
    walkseed=7,
    maxfailures=0,
    bufsize=2**23,
    chunkreads=2**23//200,
    ):

    if fltr1 is None:
        get_and_set_value_in_subfilter1 = njit(nogil=True)(lambda ft, sf, k: True)
        f1 = None
    else:
        get_and_set_value_in_subfilter1 = fltr1.private.get_and_set_value_in_subfilter
        f1 = fltr1.filter_array
    if fltr2 is None:
        get_and_set_value_in_subfilter2 = njit(nogil=True)(lambda ft, sf, k: True)
        f2 = None
    else:
        get_and_set_value_in_subfilter2 = fltr2.private.get_and_set_value_in_subfilter
        f2 = fltr2.filter_array
    if fltr3 is None:
        get_and_set_value_in_subfilter3 = njit(nogil=True)(lambda ft, sf, k: True)
        f3 = None
    else:
        get_and_set_value_in_subfilter3 = fltr3.private.get_and_set_value_in_subfilter
        f3 = fltr3.filter_array

    if update is None:
        update = h.update
    @njit(nogil=True, locals=dict(code=uint64, vx=uint64, result=int64))
    def store_in_table(ht, code, state, walkstats, ft1, ft2, ft3):
        # note that v1, v2 are known constants
        breakout = False
        state[2] += 1  # total
        if get_and_set_value_in_subfilter1(ft1, 0, code):
            if get_and_set_value_in_subfilter2(ft2, 0, code):
                if get_and_set_value_in_subfilter3(ft3, 0, code):
                    vx = state[0] - state[1]  # vsum - vx
                    st, result = update(ht, code, vx)
                    if st & 128:  # existing
                        walkstats[uint64(st&127)] += 1
                    else:
                        walkstats[result] += 1
                    if st == 0:
                        state[3] += 1  # failures
                        if maxfailures >= 0 and state[3] > maxfailures:
                            breakout = True
                    state[1] = state[0] - state[1]  # vx = vsum - vx
        return breakout

    return build_from_fastq(
        fastqs,  # list of FASTQ files
        shp,  # k-mer size or shape
        h,  # hash data structure, pre-allocated, to be filled
        values, # pair of values for indexing
        func=store_in_table,
        f1=f1, f2=f2, f3=f3,
        subsample=subsample,
        rcmode=rcmode,  # from 'f', 'r', 'both', 'min', 'max'
        walkseed=walkseed,
        maxfailures=maxfailures,
        bufsize=bufsize,
        chunkreads=chunkreads,
        )


def build_from_fastq(
    fastqs,  # list of FASTQ files
    shp,  # k-mer size or shape
    h,  # hash data structure, pre-allocated, to be filled
    values, # pair of values for indexing
    *,
    func=None,
    f1=None,
    f2=None,
    f3=None,
    update=None,
    subsample=1,
    rcmode="min",  # from 'f', 'r', 'both', 'min', 'max'
    walkseed=7,
    maxfailures=0,
    bufsize=2**23,
    chunkreads=2**23//200,
    ):
    """
    Build (fill) pre-allocated (and correctly sized) hash table 'h'
    with 'k'-mers from FASTQ files 'fastqs'.

    Each entry from each FASTQ file is processed sequentially.
    Each k-mer (and/or reverse complementary k-mer) of the entry 
    is inserted into 'h' with one of the given value.
    If the k-mer is already present, its value is updated,
    according to h's value update policy.

    rcmode has the following effect:
    rcmode=='f': insert k-mers as they appear in the file using value1.
    rcmode=='r': insert reverse complementary k-mers using value2.
    rcmode=='both': insert both k-mers using value1 and value2, respectively.
    rcmode=='min': insert the smaller of the two k-mers using value1.
    rcmode=='max': insert the larger of the two k-mers using value1.

    Return (total, failed, walkstats), where:
      total is the total number of valid k-mers read, 
      failed is the number of k-mers unsuccessfully processed,
      walkstats is an array indexed 0 to h.maxwalk+slack, counting walk lengths
    """
    print(f"# Building from FASTQ, using rcmode={rcmode}, values={values}")
    print(f"# shp={shp}, rcmode={rcmode}, maxfailures={maxfailures}, subsample={subsample}")
    ##print(f"## DEBUG values: {v1}, {v2}")  # DEBUG
    v1, v2 = values
    if rcmode == "r":
        v1 = v2
    elif rcmode != "both":
        v2 = v1
    vsum = v1 + v2

    @njit(nogil=True)
    def set_seed():
        randomseed(walkseed)

    if update is None:
        update = h.update

    @njit(nogil=True, locals=dict(code=uint64, vx=uint64, result=int64))
    def store_in_table(ht, code, state, walkstats, *_):
        # note that v1, v2 are known constants
        breakout = False
        vx = state[0] - state[1]  # vsum - vx
        state[2] += 1  # total
        st, result = update(ht, code, vx)
        if st & 128:  # existing
            walkstats[uint64(st&127)] += 1
        else:
            walkstats[result] += 1
        if st == 0:
            state[3] += 1  # failures
            if maxfailures >= 0 and state[3] > maxfailures:
                breakout = True
        state[1] = state[0] - state[1]  # vx = vsum - vx
        return breakout

    if func is None:
        func = store_in_table

    k, process_kmers = compile_kmer_processor(shp, func, rcmode)
    assert 4**k == h.universe, f"Error: k={k}; 4**k={4**k}; but universe={h.universe}"

    @njit(nogil=True,
        locals=dict(total=int64, fail=int64, v1=uint64, v2=uint64, vsum=uint64))
    def add_kmers_chunkwise(buf, linemarks, ht, v1, v2, state, walkstats, f1, f2, f3):
        vsum = v1 + v2
        n = linemarks.shape[0]
        total = 0
        for i in range(n):
            state[0:3] = (vsum, v1, uint64(0))
            sq = buf[linemarks[i,0]:linemarks[i,1]]
            length = len(sq)
            quick_dna_to_2bits(sq)
            process_kmers(ht, sq, 0, length, state, walkstats, f1, f2, f3)
            total += state[2]
            fail = state[3]
            if maxfailures >= 0 and fail > maxfailures: 
                break
        fail = state[3]
        return (total, fail)
   
    set_seed()
    ht = h.hashtable
    total = fail = 0
    walkstats = np.zeros(h.maxwalk+5, dtype=np.uint64)
    state = np.zeros(4, dtype=np.uint64)
    for chunk in fastq_chunks(fastqs, 
            bufsize=bufsize, maxreads=chunkreads, subsample=subsample):
        dtotal, fail = add_kmers_chunkwise(chunk[0], chunk[1], ht, v1, v2, state, walkstats, f1, f2, f3)
        total += dtotal
        if maxfailures >= 0 and fail > maxfailures: 
            break
    # done; hashtable h is now filled; return statistics
    return (total, fail, walkstats)
