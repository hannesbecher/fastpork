"""
filter_b3c:
a blocked bloom filter with 
- block size 512 (cache line size)
- 3 hash functions addressing bits in the same block
"""

from math import ceil
from numba import njit, uint8, uint16, int64, uint64

from .fastcash_filter import create_FCFilter
from .lowlevel.bitarray import bitarray
from .subtable_hashfunctions import get_hashfunctions, build_get_sub_subkey_from_key, parse_names
from .nthash import NT64

def build_filter(k, universe, size, nsubfilter, hashfuncs):
    """
    Allocate an array and compile access methods for a blocked bloom filter.
    Return an FCFilter object.
    """
    if nsubfilter == 1:
        print(f"# Warning: The filter is not specialized for one subfilter")

    # Basic properties
    hashtype = "b3c"
    choices = 3
    blocksize = 512
    blockmask = uint64(blocksize - 1)

    size_bits = int(ceil(size * 1024*1024*1024*8))  # 2**33 (size given in Gigabytes)
    nblocks_per_subfilter = int(ceil((size_bits // nsubfilter) / 512))
    subfilterbits = nblocks_per_subfilter * 512
    filterbits = subfilterbits * nsubfilter

    fltr = bitarray(filterbits, alignment=64)
    print(f"# allocated {nsubfilter} times {fltr.array.dtype} filters of shape {fltr.array.shape}")

    get_value_at = fltr.get  # (array, startbit, nbits=1)
    set_value_at = fltr.set  # (array, startbit, value , nbits=1)
    popcount = fltr.popcount

    if hashfuncs == "random":
        firsthashfunc = parse_names(hashfuncs, 1)[0]
    else:
        firsthashfunc, hashfuncs = hashfuncs.split(":", 1)

    get_subfilter_subkey, _ = build_get_sub_subkey_from_key(firsthashfunc, universe, nsubfilter) 
    # Same hash function for subtables

    print(f"# subfilter hash function: {firsthashfunc}")
    print(f"# filter hash functions: {hashfuncs}")

    @njit(nogil=True, locals=dict(
            key=uint64, value=uint64, sf=uint64,
            pos1=uint64, pos2=uint64, pos3=uint64))
    def store_new(fltr, key, value=1):
        sf = get_subfilter_subkey(key)[0]
        set_value_in_subfilter(fltr, sf, key, value)

    @njit(nogil=True, locals=dict(
        key=uint64, sf=uint64,
        pos1=uint64, pos2=uint64, pos3=uint64))
    def get_value(fltr, key):
        sf = get_subfilter_subkey(key)[0]
        return get_value_in_subfilter(fltr, sf, key)

    @njit(nogil=True, locals=dict(
            sf=uint64, value=uint64, key=uint64,
            h=uint64, block=uint64,
            pos1=uint64, pos2=uint64, pos3=uint64))
    def set_value_in_subfilter(fltr, sf, key, value=1):
        h = NT64(key, k)
        block = (h>>27) % nblocks_per_subfilter
        pos1 = offset + uint64(h & blockmask)
        pos2 = offset + uint64((h >> 9) & blockmask)
        pos3 = offset + uint64((h >> 18) & blockmask)
        set_value_at(fltr, pos1, value)
        set_value_at(fltr, pos2, value)
        set_value_at(fltr, pos3, value)

    @njit(nogil=True, locals=dict(
            sf=uint64, value=uint64, key=uint64,
            h=uint64, block=uint64,
            pos1=uint64, pos2=uint64, pos3=uint64,
            v1=uint64, v2=uint64, v3=uint64))
    def get_value_in_subfilter(fltr, sf, key):
        h = NT64(key, k)
        block = (h>>27) % nblocks_per_subfilter
        pos1 = offset + uint64(h & blockmask)
        pos2 = offset + uint64((h >> 9) & blockmask)
        pos3 = offset + uint64((h >> 18) & blockmask)
        v1 = get_value_at(fltr, pos1)
        v2 = get_value_at(fltr, pos2)
        v3 = get_value_at(fltr, pos3)
        return v1 & v2 & v3


    @njit(nogil=True, locals=dict(
            sf=uint64, value=uint64, key=uint64,
            h=uint64, block=uint64,
            pos1=uint64, pos2=uint64, pos3=uint64,
            v1=uint64, v2=uint64, v3=uint64))
    def get_and_set_value_in_subfilter(fltr, sf, key, value=1):
        h = NT64(key, k)
        block = (h >> 27) % nblocks_per_subfilter
        offset = subfilterbits*sf + block*blocksize
        pos1 = offset + uint64(h & blockmask)
        pos2 = offset + uint64((h >> 9) & blockmask)
        pos3 = offset + uint64((h >> 18) & blockmask)
        v1 = get_value_at(fltr, pos1)
        v2 = get_value_at(fltr, pos2)
        v3 = get_value_at(fltr, pos3)
        set_value_at(fltr, pos1, value)
        set_value_at(fltr, pos2, value)
        set_value_at(fltr, pos3, value)
        return v1 & v2 & v3

    @njit(nogil=True)
    def get_fill_level(fltr):
        return popcount(fltr) / filterbits

    filter_array = fltr.array
    return create_FCFilter(locals())
