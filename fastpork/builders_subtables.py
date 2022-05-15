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

from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.random import seed as randomseed
from numba import njit, uint8, void, int64, uint64, int32, boolean, objmode

from .io.fastaio import fasta_reads
from .io.fastqio import fastq_chunks
from .kmers import compile_kmer_processor
from .dnaencode import (
    generate_revcomp_and_canonical_code,
    quick_dna_to_2bits,
    )
from .lowlevel.pause import compile_pause
from .lowlevel.volatile import generate_volatile_load, generate_volatile_store
from .lowlevel.cmpxchg import generate_compare_xchange
vload = generate_volatile_load("uint64")
vstore = generate_volatile_store("uint64")
cmpxchg = generate_compare_xchange("uint64")

################################################################
## Control constants ###########################################

( # Status constants
    READY_TO_WRITE,
    WRITING,
    READY_TO_READ,
    READING,
    FINISHED,
) = (0, 1, 2, 3, 9)
(  # Bit sizes
    LENGTHBITS,
    STATEBITS,
    CONSUMERBITS,
) = (32, 16, 16)
(  # Bit maskes
    LENGTHMASK,
    STATEMASK,
    CONSUMERMASK,
    TESTMASK,
) = (2**LENGTHBITS-1, 2**STATEBITS-1, 2**CONSUMERBITS-1, 2**(STATEBITS+CONSUMERBITS)-1)
(  # Shifts
    LENGTHSHIFT,
    STATESHIFT,
    CONSUMERSHIFT,
) = (STATEBITS+CONSUMERBITS, CONSUMERBITS, 0)


########################################################################
## Parallel build from FASTQ with subtables ############################


def compile_kmer_dispatcher_inserter(h, shp, rcmode="max", nbuffers_per_subtable=10, nprocessors = 1, niobuffers=4, bufsize=2**17, iobufsize=2**23, maxreads=(2**23)//200, value=1):
    nconsumers = h.subtables
    nbuffers = nbuffers_per_subtable * nconsumers
    get_subtable_subkey = h.private.get_subtable_subkey_from_key
    update = h.private.update_ssk
    cpu_pause = compile_pause()
    SLEEPTIME = 1e-5
    assert nprocessors*nconsumers < nbuffers

    def initialize(stupid=False):
        """
        Initialize arrays 'control' and 'buffers' for the given parameters.

        control array:
        control[b]: uint64 = (length:32, state:16, consumer:16)
          * length is the available amount of data in buffer b.
          * state reports the status of buffer b:
            0 = empty, producer can start writing into buffer j
            1 = writing, producer is writing into buffer j
            2 = full, producer has written 'length' items into buffer j;
                ready for reading by consumer i
            3 = reading, consumer is reading from buffer j
            9 = empty, done, producer will not write more data here
          * consumer is the consumer associated with this buffer
        buffers array:
        buffers[b]: uint64[:BUFSIZE] -- space for writing/reading data
        """
        if not stupid:
            if nbuffers < 2*nconsumers:
                raise ValueError(f"use at least nbuffers={2*nconsumers} instead of {nbuffers} for {nconsumers} consumers!")
            if bufsize < 2**16:
                raise ValueError(f"use a buffer size of at least 2**16 = 65536")
        if nbuffers < nconsumers:
            raise ValueError(f"use at least nbuffers={nconsumers} instead of {nbuffers} for {nconsumers} consumers!")
        if bufsize > 2**30:
            raise ValueError(f"use a buffer size of at most 2**30 = 1 G")
        buffers = np.zeros((nbuffers, bufsize), dtype=np.uint64)
        iobuffers = np.zeros((niobuffers, iobufsize), dtype=np.uint8)
        iolinemarks = np.empty((niobuffers, maxreads, 4), dtype=np.int32)
        control = np.zeros(nbuffers, dtype=np.uint64)
        iocontrol = np.zeros(niobuffers, dtype=np.uint64)
        buffer_for = np.zeros((nconsumers, nprocessors, 2), dtype=np.int32)
        for c in range(nconsumers):
            for p in range(nprocessors):
                # control[c] = (WRITING << STATESHIFT) | c
                control[p*nconsumers + c] = (WRITING << STATESHIFT) | c
                # buffer_for[c, 0, 0] = c
                buffer_for[c, p, 0] = p*nconsumers + c

        controls = (control, iocontrol, buffer_for, buffers, iobuffers, iolinemarks)
        return controls

    @njit(nogil=True)
    def finalize(control, iocontrol, buffer_for, buffers, iobuffers, iolinemarks):
        """notify k-mer processor about of end of io"""
        # Mark io buffers for reading
        print("end of production, marking io buffers READY_TO_READ")
        for b in range(niobuffers):
            state = uint64((vload(iocontrol, b) >> STATESHIFT) & STATEMASK)
            if state != WRITING:
                print("state of iobuffer", b, "is", state)
                continue
            vstore(iocontrol,b, READY_TO_READ << STATESHIFT)
        # Now wait that every buffer has been read and is clean
        # We use a copy of control to indicate to the compiler
        # that it can't just "optimize away" the idle waiting,
        # because otherwise we have an infinite loop, where
        # control[b] is never actually checked again.
        # Damn compiler optimizations.
        print("end of production, waiting for iobuffers to clear")
        for b in range(niobuffers):
            while (vload(iocontrol, b) >> STATESHIFT) & STATEMASK != READY_TO_WRITE:
                state = (vload(iocontrol, b) >> STATESHIFT) & STATEMASK
                cpu_pause()
                # with objmode:
                #     sleep(SLEEPTIME)
            vstore(iocontrol, b, FINISHED << STATESHIFT)

        """notify consumers about end of production"""
        # Mark current writing buffers for reading
        print("end of k-mer processing, marking buffers READY_TO_READ")
        for c in range(nconsumers):
            for p in range(nprocessors):
                b = buffer_for[c, p, 0]
                state = uint64((vload(control, b) >> STATESHIFT) & STATEMASK)
                if state != WRITING:
                    print("# surprise: state of consumer",c, "buffer", b, "is", state)
                    continue
                length = buffer_for[c,p,1]
                vstore(control, b, (length << LENGTHSHIFT) | (READY_TO_READ << STATESHIFT) | c)
            ##print(b, control[b], "=", length, READY_TO_READ, c)
        # Now wait that every buffer has been read and is clean
        # We use a copy of control to indicate to the compiler
        # that it can't just "optimize away" the idle waiting,
        # because otherwise we have an infinite loop, where
        # control[b] is never actually checked again.
        # Damn compiler optimizations.
        print("# end of k-mer code generation, waiting for buffers to clear")
        for b in range(nbuffers):
            while (vload(control, b) >> STATESHIFT) & STATEMASK != READY_TO_WRITE:
                state = (vload(control, b) >> STATESHIFT) & STATEMASK
                cpu_pause()
                #with objmode: sleep(10.0*SLEEPTIME)
            vstore(control, b, FINISHED << STATESHIFT)
        return state  # irrelevant

    @njit(nogil=True, locals=dict(data=uint64, state=uint64))
    def send_to_consumer(c, data, control, iocontrol, buffer_for, buffers, iobuffers, iolinemarks, p):
        b, k = buffer_for[c, p, 0], buffer_for[c, p, 1]
        # see if the buffer is full
        if k >= bufsize:
            # find new buffer to write into for consumer
            ##print("consumer", c, "buffer", b, "READY_TO_READ")
            found = False
            while not found:
                for newb in range(nbuffers):
                    comp = vload(control, newb)
                    state = (comp >> STATESHIFT) & STATEMASK
                    if state == READY_TO_WRITE:
                        new_state = (WRITING << STATESHIFT) | c
                        if cmpxchg(control, newb, comp, new_state):
                        # vstore(control, newb, (WRITING << STATESHIFT) | c)  # length 0
                            b = buffer_for[c, p, 0] = newb
                            k = buffer_for[c, p, 1] = 0
                            ##print("producer writing buffer", b, "for consumer", c)
                            found = True
                            break
                else:  # no break
                    cpu_pause()
                    #with objmode: sleep(SLEEPTIME)
                    #print("producer: no buffer for consumer", c)
        ##print("write", data, "into buffer", c, b, k)
        buffer_for[c, p, 1] += 1
        buffers[b, k] = data
        if k + 1 >= bufsize:
            vstore(control, b, (bufsize << LENGTHSHIFT) | (READY_TO_READ << STATESHIFT) | c)

    @njit(nogil=True, locals=dict(b=int32, wait=int64, length=int64))
    def find_readable_buffer(c, control, lastbuffer):
        wait = 0
        for_me = (READY_TO_READ << STATESHIFT) | c
        while True:
            # try to find a readable buffer
            for delta in range(1, nbuffers+1):
                b = (lastbuffer + delta) % nbuffers
                if vload(control, b) & TESTMASK == for_me:
                    length = vload(control, b) >> LENGTHSHIFT
                    vstore(control, b, (READING << STATESHIFT) | c)
                    return (b, length)
            # no readable buffer, so perhaps we are done?
            done = True
            for b in range(nbuffers):
                state = (vload(control, b) >> STATESHIFT) & STATEMASK
                if state != FINISHED:
                    done = False
                    break
            if done: return (-1, wait)  # negative buffer number: done
            cpu_pause()
            #with objmode: sleep(SLEEPTIME)
            #print("no readable buffer for", c)
            wait += 1

    @njit(nogil=True, locals=dict(b=int32, mytotal=int64, failed=int64))
    def inserter(ht, c, state, walkstats, control, iocontrol, buffer_for, buffers, iobuffers, iolinemarks):
        """
        Insert k-mers in appropriate buffers into hash table.
        For now: Count total number of insertions.
        state = [total, fail, ...] (up to 16 numbers)
        """
        print("start consumer", c)
        failed = 0
        b = -1  # current buffer that this consumer (c) is reading
        while True:
            mytotal = 0
            b, length = find_readable_buffer(c, control, b)
            if b < 0: break  # done
            ##print("consumer", c, "read buffer", b, length)
            # consume the buffer
            if not failed:
                for subkey in buffers[b, :length]:
                    status, result = update(ht, c, subkey, value)
                    # status == 0: FAILED
                    # status & 127: choice
                    # status & 128: subkey existed?
                    mytotal += 1
                    if status & 128 == 0:
                        walkstats[result] += 1
                    if status == 0:
                        print("FAILURE for consumer", c, result)
                        failed += 1
                        state[1] += 1  # communicate failure to outside
                        break
            # clear buffer for re-writing
            state[0] += mytotal
            vstore(control, b, READY_TO_WRITE << STATESHIFT)
        return failed

    @njit(nogil=True, locals=dict(st=uint64, sk=uint64))
    def dispatch_kmer(ht, code, *controls):
        # function called by k-mer processor
        st, sk = get_subtable_subkey(code)
        send_to_consumer(st, sk, *controls)

    k, kmer_processor = compile_kmer_processor(shp, dispatch_kmer, rcmode)

    @njit(nogil=True)
    def process_kmers(ht, p, control, iocontrol, buffer_for, buffers, iobuffers, iolinemarks):
        print("starting kmer processor ", p)
        lastbuffer = -1
        while True:
            buffound = False
            # try to find a readable buffer
            for delta in range(1, niobuffers+1):
                b = (lastbuffer + delta) % niobuffers
                comp = uint64(vload(iocontrol, b))
                if uint64(vload(iocontrol, b) >> STATESHIFT) & STATEMASK == READY_TO_READ:
                    n = uint64(vload(iocontrol,b) >> LENGTHSHIFT) & LENGTHMASK
                    new = uint64(READING << STATESHIFT)
                    res = cmpxchg(iocontrol, b, comp, new)
                    if res:
                        # print("processor", p, "found new buffer")
                        buffound = True
                        for i in range(n):
                            sq = iobuffers[b][iolinemarks[b][i,0]:iolinemarks[b][i,1]]
                            quick_dna_to_2bits(sq)
                            controls = (control, iocontrol, buffer_for, buffers, iobuffers, iolinemarks, p)
                            kmer_processor(ht, sq, 0, len(sq), *controls)
                        lastbuffer = b
                        vstore(iocontrol, b, uint64(READY_TO_WRITE << STATESHIFT))
                        break

            if buffound:
                continue

            # no readable buffer, so perhaps we are done?
            done = True
            for i in range(niobuffers):
                state = (vload(iocontrol, i) >> STATESHIFT) & STATEMASK
                if state != FINISHED:
                    done = False
                    break
            if done: 
                print("kmer processor done")
                return
            cpu_pause()

    return k, initialize, finalize, inserter, process_kmers


def parallel_build_from_fastq(
    fastqs,  # list of FASTQ files
    shp,  # k-mer size or shape
    h,  # hash data structure, pre-allocated, to be filled, subtables >= 2
    values, # pair of values for indexing
    *,
    subsample=1,
    rcmode="min",  # from 'f', 'r', 'both', 'min', 'max'
    walkseed=7,
    maxfailures=0,
    fqbufsize=2**23,
    chunkreads=(2**23)//200,
    ):
    """
    Build (fill) pre-allocated (and correctly sized) hash table 'h'
    with 'k'-mers from FASTQ files 'fastqs'.

    Each entry from each FASTQ file is processed sequentially.
    Each k-mer (and/or reverse complementary k-mer) of the entry 
    is passed to an insert thread,
    which inserts it into 'h' with one of the given values.
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
    @njit(nogil=True)
    def set_seed():
        randomseed(walkseed)

    print(f"# Building in PARALLEL from FASTQ, using {h.subtables} subtables.")
    print(f"# shp={shp}, rcmode={rcmode}, values={values}")
    print(f"# maxwalk={h.maxwalk}, walkseed={walkseed}, maxfailures={maxfailures}.")
    
    nconsumers = h.subtables
    nprocessors = 1
    niobuffers = 2*nprocessors #TODO parameter
    nbuffers_per_subtable = 10
    k, initialize, finalize, inserter, process_kmers \
        = compile_kmer_dispatcher_inserter(h, shp, 
            rcmode=rcmode, nbuffers_per_subtable=nbuffers_per_subtable, nprocessors=nprocessors, 
            niobuffers=niobuffers, iobufsize=fqbufsize,
            maxreads=chunkreads)
    #k, process_kmers = compile_kmer_processor(shp, dispatch_kmer, rcmode)
    assert 4**k == h.universe, f"k={k}; 4**k={4**k}; but universe={h.universe}"
    update = h.update
    v1, v2 = values
    ##print(f"## DEBUG values: {v1}, {v2}")  # DEBUG
    if rcmode == "r":
        v1 = v2
    elif rcmode != "both":
        v2 = v1
    vsum = v1 + v2

    set_seed()
    ht = h.hashtable
    walkstats = np.zeros((nconsumers, h.maxwalk+16), dtype=np.int64)
    state = np.zeros((nconsumers, 16), dtype=np.int64)
    # state: per consumer: (total, failed, ...)
    controls = initialize()

    ##inserter(ht, 0, state[0,:], walkstats[0], *controls)
    result = 0
    lastiobuf = 0
    with ThreadPoolExecutor(max_workers=nprocessors+nconsumers) as executor:
        futures = [
           executor.submit(inserter, ht, c, state[c,:], walkstats[c], *controls)
           for c in range(nconsumers) ]
        futures += [executor.submit(process_kmers, ht, i, *controls) for i in range(nprocessors)]
        i=-1
        thread_index = 0
        for fqbuf, linemarks in fastq_chunks(fastqs, 
                bufsize=fqbufsize, maxreads=chunkreads, subsample=subsample):
            while True:
                i += 1
                iobuffer = (lastiobuf + i) % niobuffers
                bufstate = int(controls[1][iobuffer])
                if (bufstate >> STATESHIFT) & STATEMASK != READY_TO_WRITE:
                    assert not futures[thread_index].done(), futures[thread_index].exception()
                    thread_index = (thread_index + 1) % (nconsumers+nprocessors)
                    continue
                lastiobuf = iobuffer
                controls[1][iobuffer] = uint64(WRITING << STATESHIFT)
                controls[4][iobuffer] = fqbuf
                length = uint64(len(linemarks))
                controls[5][iobuffer][:length] = linemarks
                controls[1][iobuffer] = (uint64(READY_TO_READ) << uint64(STATESHIFT)) | (length << uint64(LENGTHSHIFT))
                break  # out of while True
            if sum(state[:,1]) > maxfailures:
                break  # out of for fqbuf, linemarks...

        finalize(*controls)
        for fut in as_completed(futures):
            pass
        # done with all futures

    # done; hashtable h is now filled; return statistics
    full_state = np.sum(state, axis=0)
    total, failed = full_state[0:2]
    full_walkstats = np.sum(walkstats, axis=0)
    return (total, failed, full_walkstats)

def parallel_build_from_fasta(
    fastas,  # list of FASTA files
    shp,  # k-mer size or shape
    h,  # hash data structure, pre-allocated, to be filled, subtables >= 2
    value_from_name, # function
    *,
    rcmode="min",  # from 'f', 'r', 'both', 'min', 'max'
    skipvalue=-1,  # value that skips a FASTA entry
    walkseed=7,
    maxfailures=0,
    fabufsize=2**23,
    chunkreads=(2**23)//200,
    ):
    """
    Build (fill) pre-allocated (and correctly sized) hash table 'h'
    with 'k'-mers from FASTA files 'fastas'.

    Each entry from each FASTA file is processed sequentially.
    Each k-mer (and/or reverse complementary k-mer) of the entry
    is passed to an insert thread,
    which inserts it into 'h' with one of the given values.
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
    @njit(nogil=True)
    def set_seed():
        randomseed(walkseed)

    print(f"# Building in PARALLEL from FASTA, using {h.subtables} subtables.")
    print(f"# shp={shp}, rcmode={rcmode}")
    print(f"# maxwalk={h.maxwalk}, walkseed={walkseed}, maxfailures={maxfailures}.")

    nconsumers = h.subtables
    nprocessors = 1
    niobuffers = 2*nprocessors #TODO parameter
    nbuffers_per_subtable = 10
    k, initialize, finalize, inserter, process_kmers \
        = compile_kmer_dispatcher_inserter(h, shp, 
            rcmode=rcmode, nbuffers_per_subtable=nbuffers_per_subtable, nprocessors=nprocessors,
            niobuffers=niobuffers, iobufsize=fabufsize,
            maxreads=chunkreads,
            value=value_from_name(None))
    #TODO J: The above only works for fixed values (xengsort)
    #k, process_kmers = compile_kmer_processor(shp, dispatch_kmer, rcmode)
    assert 4**k == h.universe, f"k={k}; 4**k={4**k}; but universe={h.universe}"
    update = h.update
    both = (rcmode=="both")

    set_seed()
    ht = h.hashtable
    walkstats = np.zeros((nconsumers, h.maxwalk+16), dtype=np.int64)
    state = np.zeros((nconsumers, 16), dtype=np.int64)
    # state: per consumer: (total, failed, ...)
    controls = initialize()

    # inserter(ht, 0, state[0,:], walkstats[0], *controls)
    # process_kmers(ht, 0, *controls)
    result = 0
    lastiobuf = 0

    with ThreadPoolExecutor(max_workers=nprocessors+nconsumers) as executor:
        futures = [
           executor.submit(inserter, ht, c, state[c,:], walkstats[c], *controls)
           for c in range(nconsumers) ]
        futures += [executor.submit(process_kmers, ht, i, *controls) for i in range(nprocessors)]
        i=-1
        thread_index = 0
        for seq in fasta_reads(fastas, sequences_only=True):
            pos = 0
            while pos < len(seq):
                end = min(pos+fabufsize, len(seq))
                while True:
                    i += 1
                    iobuffer = (lastiobuf + i) % niobuffers
                    bufstate = int(controls[1][iobuffer])
                    if (bufstate >> STATESHIFT) & STATEMASK != READY_TO_WRITE:
                        assert not futures[thread_index].done(), futures[thread_index].exception()
                        thread_index = (thread_index + 1) % (nconsumers+nprocessors)
                        continue
                    lastiobuf = iobuffer
                    controls[1][iobuffer] = uint64(WRITING << STATESHIFT)
                    controls[4][iobuffer][:end-pos] = seq[pos:end]
                    length = uint64(1)
                    controls[5][iobuffer][:length] = [0,end-pos,0,0]
                    controls[1][iobuffer] = (uint64(READY_TO_READ) << uint64(STATESHIFT)) | (length << uint64(LENGTHSHIFT))
                    break
                pos += fabufsize-k+1

                if sum(state[:,1]) > maxfailures:
                    break

        finalize(*controls)
        for fut in as_completed(futures):
            pass
        # done with all futures

    # done; hashtable h is now filled; return statistics
    full_state = np.sum(state, axis=0)
    total, failed = full_state[0:2]
    full_walkstats = np.sum(walkstats, axis=0)
    return (total, failed, full_walkstats)
