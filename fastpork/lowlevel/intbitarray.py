
from math import ceil
from collections import namedtuple

import numpy as np
from numba import njit, int64, uint64, void

from .bitarray import _aligned_zeros


IntBitArray = namedtuple("IntBitArray", [
    "size",
    "width",
    "capacity",
    "capacity_bytes",
    "capacity_ints",
    "alignment",
    "array",
    "popcount",
    "get",
    "set",
    "is_equal",
    ])


StructBitArray = namedtuple("StructBitArray", [
    "size",
    "width",
    "widths",
    "signed",
    "capacity",
    "capacity_bytes",
    "capacity_ints",
    "alignment",
    "array",
    "popcount",
    "get",
    "set",
    "getters",
    "setters",
    "is_equal",
    ])



def intbitarray(size, width, *, init=None, alignment=8):
    """
    Initialize and return an array of 'size' (unsigned) integers,
    each of which has a given 'width' in bits.

    For example intbitarray(100, 7) is an array of 100 7-bit integers
    that consumes a total of 700 bits, rounded up to the next 64 bits,
    i.e. 704 = 11*64.

    Provide get and set methods to read and wite
    each width-bit word in the array.
    The width is fixed at creation time and cannot be changed.
    If more flexibility is required, use a bitarray.

    If width is in {1, 2, 4, 8, 16, 32, 64},
    optimized access functions are used,
    as no width-bit word can span two different uint64s.
    Other widths may be slower due to the involved bit shifting.

    """
    if width < 1 or width > 64:
        raise ValueError("intbitarray: width must be between 1 and 64.")
    ints = int(ceil(size*width/64))  # number of required uint64s
    btes = ints * 8                  # number of bytes in these uint64s
    capacity = ints * 64             # number of bits in these uint64s
    mask = uint64(2**width - 1)

    @njit( ###__signature__ uint64(uint64[:], int64, int64),
        nogil=True, locals=dict(
            start=int64, x=uint64, mask1=uint64))
    def get(a, i):
        """return the i-th word from a[:]"""
        start = i * width
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        if startbit + width <= 64:
            # bits are contained in a single uint64
            x = a[startint] >> startbit
        else:
            # bits are distributed over two uint64s,
            # less significant bits are the leftmost b1=(64-startbit) bits in a[startint]
            # more significant bits are the rightmost (bits-64+startbit) bits in a[startint+1]
            b1 = 64 - startbit
            mask1 = 2**b1 - 1
            x = uint64(a[startint] >> startbit) & mask1 
            x |= uint64(a[startint+1] << b1)
        return x & mask


    @njit( ###__signature__ uint64(uint64[:], int64),
        nogil=True, locals=dict(
            start=int64, startint=uint64, startbit=uint64, x=uint64))
    def getquick(a, i):
        """return the i-th word from a[:]"""        
        start = i * width
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        x = (a[startint] >> startbit) & mask
        return x

    @njit( ###__signature__ uint64(uint64[:], int64),
        nogil=True)
    def get64(a, i):
        """return the i-th word from a[:]"""
        return a[i]


    # setter methods

    @njit( ###__signature__ void(uint64[:], int64, uint64),
        nogil=True, locals=dict(
            i=int64, value=uint64, cmask=uint64))
    def set64(a, i, value):
        """set the i-th word of a to value"""
        a[i] = value

    @njit( ###__signature__ void(uint64[:], int64, uint64),
        nogil=True, locals=dict(
            start=int64, value=uint64, cmask=uint64))
    def setquick(a, i, value):
        """set the i-th word of a to value"""
        start = i * width
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        cmask = uint64(~(uint64(mask << startbit)))
        a[startint] = (a[startint] & cmask) | uint64(value << startbit)


    @njit( ###__signature__ void(uint64[:], int64, uint64, int64),
        nogil=True, locals=dict(
            start=int64, value=uint64, v1=uint64, 
            mask1=uint64, mask2=uint64))
    def set(a, i, value):
        """set the 'i'-th word of 'a' to 'value'"""
        start = i * width
        startint = start // 64  # item starts in a[startint]
        startbit = start & 63   # at bit number startbit
        if startbit + width <= 64:
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
            b2 = width - b1
            mask2 = uint64(~(uint64(2**b2 - 1)))
            a[startint+1] = (a[startint+1] & mask2) | (value >> b1)


    if width in (1,2,4,8,16,32):
        get = getquick
        set = setquick
    elif width == 64:
        get = get64
        set = set64


    # TODO: popcount NOT IMPLEMENTED !
    @njit( ###__signature__ (uint64[:], int64, int64),
        nogil=True)
    def popcount(a, start=0, end=size):
        return 0


    if init is None:
        array = _aligned_zeros(ints, alignment=alignment)
    else:
        if alignment != 8:
            raise ValuError("cannot re-align initialized intbitarrays.")
        array = init


    def is_equal(other):
        if not isinstance(other, IntBitArray):
            return False
        if size != other.size or width != other.width:
            return False
        return np.array_equal(array, other.array)


    b = IntBitArray(size=size, width=width,
            capacity=capacity, capacity_bytes=btes, capacity_ints=ints,
            alignment=alignment, array=array,
            popcount=popcount, get=get, set=set, is_equal=is_equal)
    return b



# Structured Bit Arrays 
# #####################


def _build_getter_setter(b, shift, this, *, signed=False):
    get = b.get
    set = b.set
    mask = 2**this - 1
    if signed:
        if this < 2:
            raise ValueError("can only use signed with >= 2 bits")
        offset = int64(2 ** (this-1))
    else:
        offset = int64(0)
    setmask = uint64((2**b.width - 1) - (mask << shift))
    ##print(f"shift={shift}, thiswidth={this}, mask={mask:b}, setmask={setmask:b}")

    if not signed:
        @njit(uint64(uint64[:], uint64),
            nogil=True, parallel=True, locals=dict(v=uint64))
        def _get(a, i):
            v = uint64(get(a, i) >> shift) & uint64(mask)
            return uint64(v)
        @njit(void(uint64[:], uint64, uint64),
            nogil=True, parallel=True, locals=dict(sv=uint64, v=uint64))
        def _set(a, i, value):
            ##print(mask)
            sv = uint64(value & mask) << shift
            v = uint64(get(a, i) & setmask) | sv
            set(a, i, v)
    else:
        @njit(int64(uint64[:], uint64),
            nogil=True, parallel=True, locals=dict(v=uint64))
        def _get(a, i):
            v = uint64(get(a, i) >> shift) & uint64(mask)
            return int64(int64(v) - offset)
        @njit(void(uint64[:], uint64, int64),
            nogil=True, parallel=True, locals=dict(sv=uint64, v=uint64))
        def _set(a, i, value):
            assert -offset <= value < offset
            sv = uint64((value + offset) & mask) << shift
            v = uint64(get(a, i) & setmask) | sv
            set(a, i, v)

    return _get, _set


def structbitarray(size, widths, *, signed=None, init=None, alignment=8):
    m = len(widths)
    if signed is None:
        signed = m * (False,)
    if len(signed) != m:
        raise ValueError("structbitarray: len(signed) must equal len(widths)")
    width = sum(widths)
    b = intbitarray(size, width, init=init, alignment=alignment)
    getters = []
    setters = []

    for i in range(m):
        shift = 0 if i == 0 else sum(widths[:i])
        this = widths[i]
        _get, _set = _build_getter_setter(b, shift, this, signed=signed[i])
        getters.append(_get)
        setters.append(_set)
    ##print(getters, setters)

    sb = StructBitArray(size=size, width=width, widths=widths, signed=signed,
            capacity=b.capacity, capacity_bytes=b.capacity_bytes,
            capacity_ints=b.capacity_ints, alignment=alignment, array=b.array,
            popcount=b.popcount, get=b.get, set=b.set, is_equal=b.is_equal,
            getters=tuple(getters), setters=tuple(setters))
    return sb
