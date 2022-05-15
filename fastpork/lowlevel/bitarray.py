
from math import ceil
from collections import namedtuple

import numpy as np
from numba import njit, int64, uint64, void

from .prefetch import compile_prefetch_array_element
from .popcount import compile_popcount


BitArray = namedtuple("BitArray", [
    "size",
    "quickaccess",
    "capacity",
    "capacity_bytes",
    "capacity_ints",
    "alignment",
    "array",
    "popcount",
    "get",
    "set",
    "getquick",
    "setquick",
    "prefetch",
    ])


def _aligned_zeros(ints, alignment):
    """
    Allocate and return a buffer of zeros (uint64).
    Ensure that the byte-address of the buffer is divisible by 'alignment' bytes.
    A cache line typically has 512 bits = 64 bytes,
    so alignment=64 should ensure that the buffer starts at a chache line boundary.
    This may waste a few bytes (with alignment=64 up to 56 = 64-8).
    """
    if alignment % 8 != 0:
        raise ValueError("alignment must be a multiple of 8 bytes")
    slack = 64 - 8
    buf = np.zeros(ints+slack, dtype=np.uint64)
    address = buf.__array_interface__['data'][0]
    al = address % alignment
    shift = (alignment - al) // 8 if al != 0  else 0
    # shift start of buffer by 'shift' bytes to achieve the requested alignment
    assert 0 <= shift < slack
    b = buf[shift:shift+ints]
    address = b.__array_interface__['data'][0]
    assert address % alignment == 0
    return b


def bitarray(size, *, alignment=8, quickaccess=1):
    """
    Initialize and return a bitarray of 'size' bits.
    Ensure that the first element is aligned to an 'alignment'-byte address,
    e.g. use alignment=64 (bytes) for 512-bit alignment (cache line size).

    If bits are always read/written in blocks of 'quickaccess' bits
    (which must divide 64), the optimized methods getquick and setquick
    may be used (CAUTION: no range or error checking is performed!).
    """
    if quickaccess not in (1,2,4,8,16,32,64):
        raise ValueError("bitarray: quickaccess must be a power of 2.")
    ints = ceil(size / 64)
    btes = ints * 8
    capacity = ints * 64
    quickmask = uint64(2**quickaccess - 1)
    prefetch_index = compile_prefetch_array_element()
    popcount_value = compile_popcount("uint64")

    @njit( ###__signature__ uint64(uint64[:], int64, int64),
        nogil=True, locals=dict(
            start=int64, x=uint64, mask=uint64, mask1=uint64))
    def get(a, start, bits=1):
        """return 'bits' bits from a[start:start+bits], where bits <= 64"""
        if bits <= 0:  return 0
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        if startbit + bits <= 64:
            # bits are contained in a single uint64
            x = a[startint]
            if startbit > 0:
                x >>= startbit
        else:
            # bits are distributed over two uint64s,
            # less significant bits are the leftmost b1=(64-startbit) bits in a[startint]
            # more significant bits are the rightmost (bits-64+startbit) bits in a[startint+1]
            b1 = 64 - startbit
            mask1 = 2**b1 - 1
            x = uint64(a[startint] >> startbit) & mask1 
            x |= uint64(a[startint+1] << b1)
        # due to a bug in numba, do not use x = y if cond else z !!
        if bits >= 64:
            return x
        mask = uint64(2**bits - 1)
        x &= mask
        return x

    @njit(nogil=True, locals=dict(startint=uint64))
    def prefetch(a, start):
        startint = start // 64  # item starts in a[startint]
        prefetch_index(a, startint)


    @njit( ###__signature__ uint64(uint64[:], int64),
        nogil=True, locals=dict(
            start=int64, startint=uint64, startbit=uint64, x=uint64))
    def getquick(a, start):
        if quickaccess <= 0: return 0
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        x = a[startint]
        x = (x >> startbit) & quickmask
        return x


    @njit( ###__signature__ void(uint64[:], int64, uint64),
        nogil=True, locals=dict(
            start=int64, value=uint64, quicksetmask=uint64))
    def setquick(a, start, value):
        if quickaccess <= 0: return 0
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        quicksetmask = uint64(~(quickmask << startbit))
        a[startint] = (a[startint] & quicksetmask) | (value << startbit)


    @njit( ###__signature__ void(uint64[:], int64, uint64, int64),
        nogil=True, locals=dict(
            start=int64, value=uint64, v1=uint64, 
            mask=uint64, mask1=uint64, mask2=uint64))
    def set(a, start, value, bits=1):
        """set a[start:start+bits] to value, where bits <= 64"""
        if bits <= 0: return
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        if bits >= 64:
            mask = uint64(-1)
        else:
            mask = uint64(2**bits - 1)
        if startbit + bits <= 64:
            # bits are contained in a single uint64
            mask1 = ~(mask << startbit)
            a[startint] = (a[startint] & mask1) | (value << startbit)
        else:
            # b1 leftmost bits in a[startint] == b1 rightmost bits of v, 
            b1 = 64 - startbit
            v1 = (value & uint64(2**b1 - 1))  # v1 = b1 rightmost bits of v
            mask1 = uint64(2**startbit - 1)  # only keep startbit rightmost bits
            a[startint] = (a[startint] & mask1) | (v1 << startbit)
            # b2 rightmost bits in a[startint+1] = b2 leftmost bits of v
            b2 = bits - b1
            mask2 = uint64(~(2**b2 - 1))
            a[startint+1] = (a[startint+1] & mask2) | (value >> b1)

    # TODO: NOT WELL IMPLEMENTED !
    @njit(nogil=True, locals=dict(
            s=uint64, stemp=uint64))
    def popcount(a, start=0, end=size):
        s = 0
        startint = int64((start // 64)+1)  # item starts in a[startint]
        endint = int64(end//64)

        for i in range(start, startint*64):
            s += get(a, i)

        for i in range(startint, endint):
            value = a[i]
            s += popcount_value(value)

        for i in range(endint*64, end):
            s += get(a, i)

        return s

    array = _aligned_zeros(ints, alignment=alignment)

    b = BitArray(size=size, quickaccess=quickaccess,
        capacity=capacity, capacity_bytes=btes, capacity_ints=ints,
        alignment=alignment, array=array,
        popcount=popcount, get=get, set=set,
        getquick=getquick, setquick=setquick, prefetch=prefetch)
    return b
