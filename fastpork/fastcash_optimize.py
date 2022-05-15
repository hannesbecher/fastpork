import datetime

import numpy as np
from numba import njit, jit, uint64, int64, uint32, uint8, boolean
from math import log, ceil
from importlib import import_module

from .lowlevel.bitarray import bitarray
from .lowlevel.intbitarray import intbitarray, IntBitArray
from .hashio import save_hash
from .h5utils import load_from_h5group, save_to_h5group
from .hashfunctions import get_hashfunctions, build_get_page_fpr
from .subtable_hashfunctions import parse_names as st_parse_names
from .subtable_hashfunctions import build_get_sub_subkey_from_key, build_get_page_fpr_from_subkey
from .srhash import get_npages
from .parameters import get_valueset_parameters
from .mathutils import print_histogram, print_histogram_tail
from concurrent.futures import ThreadPoolExecutor, wait

def show_memory(*arrays, names=None):
    total = 0
    if names is None:
        names = [f"array{i}" for i in range(1, len(arrays)+1)]
    for a, name in zip(arrays, names):
        if a is None:
            continue
        if isinstance(a, np.ndarray):
            size = a.size
            b = a.nbytes
            dtype = str(a.dtype)
        elif isinstance(a, IntBitArray):
            size = a.size
            b = a.capacity_bytes
            dtype = f'bits{a.width}'
        elif isinstance(a, StructBitArray):
            size = a.size
            b = a.capacity_bytes
            w = str(tuple(a.widths)).replace(" ", "")
            dtype = f'bits{a.width}{w}'
        else:
            raise RuntimeError(f"unknown array type '{type(a)}'")
        print(f"{name}: {size} x {dtype} = {b/1E6:.3f} MBytes = {b/1E9:.3f} GBytes")
        total += b
    print(f"TOTAL: {total/1E6:.3f} MBytes = {total/1E9:.3f} GBytes")

def compile_calc_subkeys(keys, subkeys, subtables, nkeys, get_st_sk):
    get_key = keys.get
    set_sk = subkeys.set
    set_st = subtables.set

    @njit(nogil=True, locals=dict(st=uint64, sk=uint64))
    def calc_subkeys(akey, ask, ast, st_size):
        for i in range(nkeys):
            st, sk = get_st_sk(get_key(akey, i))
            set_sk(ask, i, sk)
            set_st(ast, i, st)
            st_size[st] += 1

    return calc_subkeys

def compile_split_bitarrays(subkeys, subtables, nkeys):
    get_key = subkeys.get
    set_key = subkeys.set
    get_st = subtables.get

    @njit(nogil=True, locals=dict())
    def split_bitarrays(ask, ast, st_keys):
        pos = np.zeros(len(st_keys), dtype=np.int64)
        for i in range(nkeys):
            key = get_key(ask, i)
            st = get_st(ast, i)
            set_key(st_keys[st], pos[st], key)
            pos[st] += 1
    return split_bitarrays

def compile_merge_choices(choices, subtables, nkeys):
    get_choice = choices.get
    set_choice = choices.set
    get_st = subtables.get

    def merge_choices(achoices, subtable_choices, asubtables):
        pos = np.zeros(nkeys, dtype=np.int64)
        for i in range(nkeys):
            st = get_st(asubtables, i)
            c = get_choice(subtable_choices[st], pos[st])
            pos[st] += 1
            set_choice(achoices, i, c)

    return merge_choices

def check_choices(get_c, ac, get_k, ak, get_pfs, nkeys, npages, pagesize):
    fill = np.zeros(npages, dtype=np.uint8)
    for i in range(nkeys):
        sk = get_k(ak,i)
        c = get_c(ac,i)-1
        p, _ = get_pfs[c](sk)
        fill[p] += 1
    assert (fill <= pagesize).all()



def optimize(nkmers, bucketsize, bucketcount, hfs, kmers):
    pFF = intbitarray(bucketcount, np.ceil(np.log2(bucketsize+1)))
    pF = pFF.array
    getbucketfill = pFF.get
    setbucketfill = pFF.set

    ukmerr = bitarray(nkmers*2)
    ukmer = ukmerr.array
    get_bits_at = ukmerr.get  # (array, startbit, nbits=1)
    set_bits_at = ukmerr.set
    hf1 = hfs[0]
    hf2 = hfs[1]
    hf3 = hfs[2]


    @njit()
    def calcStats(get_bits_at, ukmer, pF):
        val = 0
        choiceStat = np.zeros(3, dtype=np.uint64)
        for i in range(nkmers):
            choice = int(get_bits_at(ukmer, i*2, 2))
            val += choice
            choiceStat[choice-1] +=1
        bucketFillStat = np.zeros(bucketsize+1, dtype = np.uint64)
        for i in range(bucketcount):
            bucketFillStat[getbucketfill(pF, i)] += 1

        print("costs:", val, val/nkmers)
        print("choice Stat:")
        print("1: ", choiceStat[0])
        print("2: ", choiceStat[1])
        print("3: ", choiceStat[2])
        print("bucket fill:")
        for i, j in enumerate(bucketFillStat):
            print(i, ": ", j)


    @njit
    def checkL(Li, Lv, Lh, ukmer, get, set, nkmers, bucketcount):
        for i in range(0, Li[-1]):
            if Lv[i] != np.iinfo(uint64).max:
                choice = int(get(ukmer, Lv[i]*2, 2))
                hf = int(get(Lh, i*2, 2))
                #''assert(choice != hf)


    @njit(nogil=True, locals=dict(kmer=uint64, i=int64,
        choice=uint8, h1=uint64, h2=uint64, h3=uint64,
        ))
    def getL(nkmers, #number of k-mers
             kmers, # compressed array of k-mer codes
             get_int_at, # function to get a code out of kmers
             bucketcount, # number of buckets
             hf,
             ukmer, # hf that is used used to assign a k-mer
             val,
             ind # Array of all indices at which a bucket starts
            ):

        set = sethashat
        get = gethashat

        # For each bucket get the number of elements that can be assigned to the bucket but currently are not
        # If for one elementem multiple hash function point to the same bucket only count one
        mFill = np.zeros(bucketcount, dtype=np.uint64)
        for i in range(nkmers):
            choice = int(get_bits_at(ukmer, i*2, 2))
            if choice == 0:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                h3 = hf3(int(get_int_at(kmers, i)))[0]
                mFill[h1] += 1
                if h2 != h1:
                    mFill[h2] += 1
                if h3 != h1 and h3 != h2:
                    mFill[h3] += 1
            elif choice == 1:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                h3 = hf3(int(get_int_at(kmers, i)))[0]
                if h2 != h1:
                    mFill[h2] += 1
                if h3 != h2 and h3 != h1:
                    mFill[h3] += 1
            elif choice == 2:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                h3 = hf3(int(get_int_at(kmers, i)))[0]
                mFill[h1] += 1
                if h3 != h1 and h3 != h2:
                    mFill[h3] += 1
            elif choice == 3:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                mFill[h1] += 1
                if h2 != h1:
                    mFill[h2] += 1
            else:
                assert False, "choice must be in [0...3] "

        # store all indices where a new bucket beginns
        for i in range(1, bucketcount):
            ind_set(ind, i, ind_get(ind, (i-1))+mFill[i-1])
        ind_set(ind, bucketcount, ind_get(ind, bucketcount-1)+mFill[bucketcount-1])


        pFill = np.zeros(bucketcount, dtype=np.uint32)
        for i in range(nkmers):
            if int(get_bits_at(ukmer, i*2, 2)) == 0:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                h3 = hf3(int(get_int_at(kmers, i)))[0]

                ti_set(val, (ind_get(ind, h1)+pFill[h1]),i)
                set(hf, (ind_get(ind, h1)+pFill[h1]), 1)
                pFill[h1] += 1

                if h2 != h1:
                    ti_set(val, (ind_get(ind, h2)+pFill[h2]), i)
                    set(hf, (ind_get(ind, h2)+pFill[h2]), 2)
                    pFill[h2] += 1
                if h3 != h1 and h3 != h2:
                    ti_set(val, ind_get(ind, h3)+pFill[h3], i)
                    set(hf, (ind_get(ind, h3)+pFill[h3]), 3)
                    pFill[h3] += 1

            elif int(get_bits_at(ukmer, i*2, 2)) == 1:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                h3 = hf3(int(get_int_at(kmers, i)))[0]

                if h2 != h1:
                    ti_set(val, (ind_get(ind, h2)+pFill[h2]), i)
                    set(hf, (ind_get(ind, h2)+pFill[h2]), 2)
                    pFill[h2] += 1

                if h3 != h2 and h3 != h1:
                    ti_set(val, (ind_get(ind, h3)+pFill[h3]), i)
                    set(hf, (ind_get(ind, h3)+pFill[h3]), 3)
                    pFill[h3] += 1
            elif int(get_bits_at(ukmer, i*2, 2)) == 2:
                h1 = hf1(int(get_int_at(kmers, i)))[0]
                h2 = hf2(int(get_int_at(kmers, i)))[0]
                h3 = hf3(int(get_int_at(kmers, i)))[0]

                ti_set(val, (ind_get(ind, h1)+pFill[h1]), i)
                set(hf, (ind_get(ind, h1)+pFill[h1]), 1)
                pFill[h1] += 1

                if h3 != h1 and h3 != h2:
                    ti_set(val, (ind_get(ind, h3)+pFill[h3]), i)
                    set(hf, (ind_get(ind, h3)+pFill[h3]), 3)
                    pFill[h3] += 1

            elif int(get_bits_at(ukmer, i*2, 2)) == 3:
                print("choice cannot be 3")
                assert()

        mFill = None
        pFill = None

        return ind, val, hf

    @njit(nogil=True, locals=dict(bucket=uint64, bucketFill=uint8))
    def _init(pF, bucketcount, bucketsize, set_bits_at, get_bits_at, ukmer, kmers, get_int_at, nkmers):
        # Initialization passes:
        # Insert as many elements as possible only using the first and second hash functions without moving elements.

        # Insert as many elements as possible using only the first hash function
        for i in range(nkmers):
            bucket = hf1(int(get_int_at(kmers, i)))[0]
            bucketFill = getbucketfill(pF, bucket)
            if bucketFill != bucketsize:
                setbucketfill(pF, bucket, bucketFill+1)
                set_bits_at(ukmer, i*2, 1, 2)

        count = 0
        # Insert as many of the remaining elments as possible only using the second hashfunction
        for i in range(nkmers):
            if int(get_bits_at(ukmer, i*2, 2)) == 0:
                bucket = hf2(int(get_int_at(kmers, i)))[0]
                bucketFill = getbucketfill(pF, bucket)
                if bucketFill != bucketsize:
                    setbucketfill(pF, bucket, bucketFill+1)
                    set_bits_at(ukmer, i*2, 2, 2)
                else:
                    count += 1

        print("Number of not inserted k-mers after initialization:")
        print(count)
        return count

    @njit(nogil=True, locals=dict(node=uint64, bucket_choice=uint64, pbucket=uint64, bucketFill=uint64, choice=uint64))
    def alternatePaths(prev_kmer, prev_bucket, set_bits_at, get_bits_at, ukmer, pF, bucketcost, visitedKmer, kmers, get_int_at, nkmers, Li, Lh, Lv):
        count = 0
        """
        Iterate over all k-mers; if a k-mer is not assigned insert it following the calculated way
        """

        for skmer in range(nkmers):
            if int(get_bits_at(ukmer, skmer*2, 2)) != 0:
                continue

            visited = False
            node = skmer
            while node != nkmers+1:
                if int(get_bits_at(visitedKmer, node)):
                    visited = True
                    break

                bucket_choice = int(get_bits_at(prev_bucket, node*2, 2))
                if bucket_choice == 1:
                    pbucket = hf1(int(get_int_at(kmers, node)))[0]
                elif bucket_choice == 2:
                    pbucket = hf2(int(get_int_at(kmers, node)))[0]
                elif bucket_choice == 3:
                    pbucket = hf3(int(get_int_at(kmers, node)))[0]

                if prev_kmer_get(prev_kmer, pbucket) == nkmers+1:
                    if getbucketfill(pF, pbucket) == bucketsize:
                        visited = True
                        break
                node = prev_kmer_get(prev_kmer, pbucket)

            if visited:
                continue

            count += 1
            node = skmer
            while node != np.iinfo(np.uint32).max:
                set_bits_at(visitedKmer, node, 1)
                if int(get_bits_at(prev_bucket, node*2, 2)) == 1:
                    set_bits_at(ukmer, node*2, 1, 2)
                elif int(get_bits_at(prev_bucket, node*2, 2)) == 2:
                    set_bits_at(ukmer, node*2, 2, 2)
                elif int(get_bits_at(prev_bucket, node*2, 2)) == 3:
                    set_bits_at(ukmer, node*2, 3, 2)
                else:
                    assert()

                bucket_choice = int(get_bits_at(prev_bucket, node*2, 2))
                if bucket_choice == 1:
                    pbucket = hf1(int(get_int_at(kmers, node)))[0]
                elif bucket_choice == 2:
                    pbucket = hf2(int(get_int_at(kmers, node)))[0]
                elif bucket_choice == 3:
                    pbucket = hf3(int(get_int_at(kmers, node)))[0]

                bucketFill = getbucketfill(pF, pbucket)
                if bucketFill != bucketsize:
                    for i in range(ind_get(Li, pbucket), ind_get(Li, pbucket+1)):
                        if ti_get(Lv, i) == node:
                            setbucketfill(pF, pbucket, bucketFill+1)
                            ti_set(Lv, i, nkmers+1)
                            sethashat(Lh, i, 0)
                            break
                    break

                for i in range(ind_get(Li, pbucket), ind_get(Li, pbucket+1)):
                    if ti_get(Lv, i) == node:
                        ti_set(Lv, i, prev_kmer_get(prev_kmer, pbucket))
                        choice = int(get_bits_at(ukmer, prev_kmer_get(prev_kmer,pbucket)*2, 2))
                        sethashat(Lh, i, choice)
                        break

                node = prev_kmer_get(prev_kmer, pbucket)

        return count

    @njit(nogil=True, locals=dict(i=int64, changes=boolean, kmer=uint32, bucket=uint32, choice=uint8, hashfunc=uint8, prevbucket = uint64))
    def findPaths(Lv, Li, Lh, bucketcost, bucketcount, prev_bucket, prev_kmer, set_bits_at, get_bits_at, ukmer, activebucket, pF, kmers, get_int_at, nkmers):
        """
        Calulate all path beginning at empty buckets.
        """

        for i in range(bucketcount):
            val = getbucketfill(pF, i)
            if val < bucketsize:
                bucketcost[i] = 0
                set_bits_at(activebucket, i, 1)

        changes = True
        count = 0
        while changes:
            changes = False
            for bucket in range(bucketcount):
                if int(get_bits_at(activebucket, bucket)) == 0:
                    continue

                set_bits_at(activebucket, bucket, 0)
                count += 1
                for i in range(ind_get(Li, bucket), ind_get(Li, bucket+1)):
                    kmer = ti_get(Lv, i)

                    if kmer == nkmers+1:
                        continue

                    hashfunc = int(gethashat(Lh, i))

                    choice = int(get_bits_at(ukmer, kmer*2, 2))

                    if choice == 0:
                        if int(get_bits_at(prev_bucket, kmer*2, 2)) == 0:
                            set_bits_at(prev_bucket, kmer*2, hashfunc, 2)
                        else:
                            prevbucketHashfunc = int(get_bits_at(prev_bucket, kmer*2, 2))
                            if prevbucketHashfunc == 1:
                                prevbucket = hf1(int(get_int_at(kmers, kmer)))[0]
                            elif prevbucketHashfunc == 2:
                                prevbucket = hf2(int(get_int_at(kmers, kmer)))[0]
                            elif prevbucketHashfunc == 3:
                                prevbucket = hf3(int(get_int_at(kmers, kmer)))[0]
                            if bucketcost[bucket]+hashfunc < bucketcost[prevbucket]+prevbucketHashfunc:
                                set_bits_at(prev_bucket, kmer*2, hashfunc, 2)
                        continue
                    elif choice == 1:
                        choicebucket = hf1(int(get_int_at(kmers, kmer)))[0]
                    elif choice == 2:
                        choicebucket = hf2(int(get_int_at(kmers, kmer)))[0]
                    elif choice == 3:
                        choicebucket = hf3(int(get_int_at(kmers, kmer)))[0]
                    else:
                        assert False, "No valid choice"

                    if bucketcost[choicebucket] <= bucketcost[bucket] -choice+hashfunc:
                        continue

                    set_bits_at(prev_bucket, kmer*2, hashfunc, 2)

                    set_bits_at(activebucket, choicebucket, 1)
                    bucketcost[choicebucket] = bucketcost[bucket] -choice + hashfunc
                    prev_kmer_set(prev_kmer, choicebucket, kmer)
                    changes = True

        print("Number of updates in bucketcost: ", count)

    @njit(nogil=True)
    def startPasses(pF, nok, bucketcount, prev_kmer, bucketcost, bucketsize, set_bits_at, get_bits_at, ukmer, visitedKmer, activebucket, kmers, get_int, nkmers, hf, prev_bucket, Li, Lv, Lh):
        passes = 1
        while nok > 0:
            print("pass: ", passes)
            print("Computing minimum cost paths")
            findPaths(Lv, Li, Lh, bucketcost, bucketcount, prev_bucket, prev_kmer, set_bits_at, get_bits_at, ukmer, activebucket, pF, kmers, get_int, nkmers)
            print("Moving and inserting elements")
#            checkL(Li, Lv, Lh, ukmer, get_bits_at, set_bits_at, nkmers, bucketcount)
            insertedKmer = alternatePaths(prev_kmer, prev_bucket, set_bits_at, get_bits_at, ukmer, pF, bucketcost, visitedKmer, kmers, get_int, nkmers, Li, Lh, Lv)
            print(f"Number of inserted k-mers:", insertedKmer)
#            checkL(Li, Lv, Lh, ukmer, get_bits_at, set_bits_at, nkmers, bucketcount)
            nok -= insertedKmer
            print(f"Number of open k-mers:", nok)
            bucketcost.fill(np.iinfo(np.int16).max)
            prev_kmer.fill(np.iinfo(np.uint32).max)

            for i in range(nkmers):
                set_bits_at(visitedKmer, i, 0)
                set_bits_at(prev_bucket, i, 0, 2)
            for i in range(bucketcount+1):
                prev_kmer_set(prev_kmer, i, nkmers+1)
            if insertedKmer == 0:
                print("unsolvable")
                assert False
            passes += 1

    kmerbits = int(np.ceil(np.log2(nkmers+1)))
    prev_kmerr = intbitarray(bucketcount, kmerbits)
    prev_kmer = prev_kmerr.array
    prev_kmer_set = prev_kmerr.set
    prev_kmer_get = prev_kmerr.get
    #init prev_kmer
    for i in range(bucketcount+1):
        prev_kmer_set(prev_kmer, i, nkmers+1)

    bucketcost = np.full(bucketcount, np.iinfo(np.int16).max, dtype=np.int16)
    visitedKmer = bitarray(nkmers).array
    activebucket = visitedKmer
    prev_bucket = bitarray(nkmers*2).array
    hff = intbitarray(nkmers*3, 2)
    hf = hff.array
    gethashat = hff.get
    sethashat = hff.set

    beg = datetime.datetime.now()
    nok = _init(pF, bucketcount, bucketsize, set_bits_at, get_bits_at, ukmer, kmers.array, kmers.get, nkmers)

    print("Compute T array")
    tisize = nok*3+(nkmers-nok)*2
    #define Tj
    hff = intbitarray(tisize, 2)
    hf = hff.array
    gethashat = hff.get
    sethashat = hff.set
    #define Ti
    tii = intbitarray(tisize, kmerbits)
    ti = tii.array
    ti_get = tii.get
    ti_set = tii.set
    #define Tstarts
    indd = intbitarray(bucketcount+1, np.ceil(np.log2(tisize)))
    ind = indd.array
    ind_get = indd.get
    ind_set = indd.set
    print("compute l")
    Li, Lv, Lh = getL(nkmers, kmers.array, kmers.get, bucketcount, hf, ukmer, ti, ind)

    startPasses(pF, nok, bucketcount, prev_kmer, bucketcost, bucketsize, set_bits_at, get_bits_at, ukmer, visitedKmer, activebucket, kmers.array, kmers.get, nkmers, hf, prev_bucket, Li, Lv, Lh)

    end = datetime.datetime.now()
    print("Time to calculate an optimal assignment:")
    print((end-beg).total_seconds())

    print(f"'{datetime.datetime.now()}': Calculate statistics")
    calcStats(get_bits_at, ukmer, pF)

    print(f"'{datetime.datetime.now()}': Calculate memory usage")
    show_memory(kmers, ukmer, pF, Li, Lv, Lh, bucketcost, prev_kmer, prev_bucket, visitedKmer, names = "elements assignments pagefill Tstarts Ti Tj bucket_cost prev_element prev_bucket emba".split())
    return ukmer

# main #########################################

def main(args):
    """main method for calculating an optimal hash assignment"""
    # needs: args.index (str),
    # optional: args.value (int)

    starttime = datetime.datetime.now()
    print(f"# {starttime:%Y-%m-%d %H:%M:%S}: Optimizing dump '{args.dump}'")

    # load dump
    print("# load dump")
    info = load_from_h5group(args.dump, "info")['info']
    data = load_from_h5group(args.dump, "data")

    print("# extract infos")
    universe = int(info['universe'])

    # get infos to init new bitarrays
    k = info['k']
    nkeys = int(info['kmers'])
    valuebits = int(info['valuebits'])
    subtablebits = int(info['subtablebits'])

    # check if new number of subtables is defined
    nsubtables = int(info['subtables'])
    if not args.subtables is None and args.subtables != nsubtables:
        nsubtables = args.subtables
        info['subtables'] = nsubtables

    pagesize = int(args.pagesize) if args.pagesize else int(info['pagesize'])
    info['pagesize'] = pagesize
    fill = args.fill if args.fill else float(info['fillrate'])
    info['fill'] = fill
    if nsubtables:
        npages = get_npages(nkeys//nsubtables, pagesize, fill)
    else:
        info['hashtype'] = info['hashtype'][1:]
        npages = get_npages(nkeys, pagesize, fill)
    rcmode = info['rcmode']
    if not isinstance(rcmode, str):
        rcmode = rcmode.decode()

    #check if new hash functions are defined
    hashfuncs = info['hashfuncs']
    if not args.hashfunctions is None:
        hashfuncs = args.hashfunctions
        info['hashfuncs'] = hashfuncs
    if not isinstance(hashfuncs, str):
        hashfuncs = hashfuncs.decode()

    print("# build new int bit arrays")
    keys = intbitarray(nkeys, 2*k, init=data['kmercodes'])
    print(f"# Codes array: Size is {(keys.capacity_bytes/2**30):.3f} GB (2**30 bytes).")
    # subkeys = intbitarray(nkeys, 2*k)
    subkeys = keys
    print(f"# Codes array: Size is {(subkeys.capacity_bytes/2**30):.3f} GB (2**30 bytes).")
    subtables = intbitarray(nkeys, subtablebits, init=data['subtables'])
    print(f"# Codes array: Size is {(subtables.capacity_bytes/2**30):.3f} GB (2**30 bytes).")

    beg = datetime.datetime.now()
    if nsubtables:
        hfs = st_parse_names(hashfuncs, 4)
        get_subtable_subkey_from_key, get_key_from_subtable_subkey = build_get_sub_subkey_from_key(hfs[0], universe, nsubtables)
        get_page_fpr1, get_subkey_from_page_fpr1 =  build_get_page_fpr_from_subkey(hfs[1], universe, npages, nsubtables) #nsubtables to update the universe for each subtable
        get_page_fpr2, get_subkey_from_page_fpr2 =  build_get_page_fpr_from_subkey(hfs[2], universe, npages, nsubtables) #nsubtables to update the universe for each subtable
        get_page_fpr3, get_subkey_from_page_fpr3 =  build_get_page_fpr_from_subkey(hfs[3], universe, npages, nsubtables) #nsubtables to update the universe for each subtable
        get_pf = (get_page_fpr1, get_page_fpr2, get_page_fpr3)
        subtable_sizes = np.zeros(nsubtables, dtype=np.uint64)

        print("# Calculate subkeys")
        calc_subkeys = compile_calc_subkeys(keys, subkeys, subtables, nkeys, get_subtable_subkey_from_key)
        calc_subkeys(keys.array, subkeys.array, subtables.array, subtable_sizes)
        subtable_keys = tuple(intbitarray(subtable_sizes[i], 2*k) for i in range(nsubtables))
        subtable_keys_arrays = tuple(i.array for i in subtable_keys)
        split_bitarrays = compile_split_bitarrays(subkeys, subtables, nkeys)
        split_bitarrays(subkeys.array, subtables.array, subtable_keys_arrays)

        print("# Start optimization")
        choice = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [ executor.submit(
                optimize, subtable_sizes[i], pagesize, npages, get_pf, subtable_keys[i])
                for i in range(nsubtables) ]
            wait(futures)
            for i in range(nsubtables):
                choice.append(futures[i].result())

        choices = intbitarray(nkeys, 2)
        get_choice = choices.get
        merge_choices = compile_merge_choices(choices, subtables, nkeys)
        merge_choices(choices.array, tuple(choice), subtables.array)
        choices = choices.array


    else:
        print("no subtabĺes")
        hf = hashfuncs.split(":")
        if len(hf) == 4:
            hf = hf[1:]
            hashfuncs = ":".join(hf)
        get_pf1, _ = build_get_page_fpr(hf[0], universe, npages)
        get_pf2, _ = build_get_page_fpr(hf[1], universe, npages)
        get_pf3, _= build_get_page_fpr(hf[2], universe, npages)
        get_pf = (get_pf1, get_pf2, get_pf3)
        choices = optimize(nkeys, pagesize, npages, get_pf, keys)
    end = datetime.datetime.now()
    print(f"Full time to calculate an optimal assignment for {nsubtables} subtables: {(end-beg).total_seconds()}")

    values = intbitarray(nkeys, valuebits, init=data['values'])
    if args.optdump:
        save_to_h5group(args.optdump, "info", info=info)
        save_to_h5group(args.optdump, "data", kmercodes=kmers.array, choices=choices, values=values.array)
    if args.optindex:
        # generate valueset
        P = get_valueset_parameters(info['valueset'].split(".")[1].split(), k=k, rcmode=rcmode)
        (valueset, valuestr, rcmode, k, parameters) = P

        # build hash table
        hashtype = info['hashtype'].decode()
        hashmodule = import_module(".hash_" + hashtype, __package__)
        build_hash = hashmodule.build_hash
        fill_from_arrays = hashmodule.fill_from_arrays
        aligned = info['aligned']
        nfingerprints = info['nfingerprints']
        maxwalk = 3
        if nsubtables:
            h = build_hash(universe, npages*pagesize*nsubtables, nsubtables, pagesize,
                            hashfuncs, valueset.NVALUES, valueset.update,
                            aligned=aligned, nfingerprints=nfingerprints,
                            maxwalk=maxwalk, shortcutbits=int(info['shortcutbits']))

            print(f"{npages*pagesize*nsubtables=}")
            print(f"{h.npages=}")
            total, failed, walkstats = fill_from_arrays(h, k, nkeys, subkeys, subtables, choices, values)
        else:
            h = build_hash(universe, npages*pagesize, nsubtables, pagesize,
                            hashfuncs, valueset.NVALUES, valueset.update,
                            aligned=aligned, nfingerprints=nfingerprints,
                            maxwalk=maxwalk, shortcutbits=int(info['shortcutbits']))

            print(f"{npages*pagesize*nsubtables=}")
            print(f"{h.npages=}")
            total, failed, walkstats = fill_from_arrays(h, k, nkeys, keys, choices, values)
        valuehist, fillhist, choicehist, shortcuthist = h.get_statistics(h.hashtable)
        print_histogram_tail(walkstats, [1,2,10], title="Extreme walk lengths:", shorttitle="walk", average=True)
        print_histogram(np.sum(valuehist, axis=0), title="Value statistics:", shorttitle="values", fractions="%")
        print_histogram(np.sum(fillhist, axis=0), title="Page fill statistics:", shorttitle="fill", fractions="%", average=True, nonzerofrac=True)
        print_histogram(np.sum(choicehist, axis=0), title="Choice statistics:", shorttitle="choice", fractions="%+", average="+")
        costs = 0
        for i,j in enumerate(np.sum(choicehist, axis=0)):
            costs += i*j
        print(f"### subtables: {h.subtables}, costs: {costs}")
        save_hash(args.optindex, h, valuestr,
            additional=dict(k=k, walkseed=info['walkseed'], rcmode=rcmode, fillrate=fill, valueset=valuestr))
