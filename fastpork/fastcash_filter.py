"""
Module fastcash.fastcash_filter

This module provides
*  FCFilter, a namedtuple to store filter information

It provides factory functions to build an FCFilter
that are jit-compiled.
"""

from collections import namedtuple

FCFilter = namedtuple("FCFilter", [
    ### atributes
    "universe",
    "size",
    "nsubfilter",
    "hashfuncs",
    "hashtype",
    "choices",
    "subfilterbits",
    "filterbits",
    "mem_bytes",
    "filter_array",

    ### public API methods
    "store_new", # (table:uint64[:], key:uint64, value:uint64)
    "get_value", # (table:uint64[:], key:uint64) -> uint64
    "get_fill_level", # (table:uint64[:]) -> float

    ### private API methods (see below)
    "private",
    ])

SRHash_private = namedtuple("FCFilter_private", [
    # private API methods, may change !
    "get_value_in_subfilter", # (table:uint64[:], subfilter:uint64, key:uint64) -> uint64
    "set_value_in_subfilter", # (table:uint64[:], subfilter:uint64, key:uint64, value:uint64)
    "get_and_set_value_in_subfilter", # (table:uint64[:], subfilter:uint64, key:uint64, value:uint64)
    "get_value_at", # (table:uint64[:], pos:uint64) -> uint64
    "set_value_at", # (table:uint64[:], pos:uint64, value:uint64)

    ])


def create_FCFilter(d):
    """Return FCFilter initialized from values in dictionary d"""
    # The given d does not need to provide mem_bytes; it is computed here.
    # The given d is copied and reduced to the required fields.
    # The hashfuncs tuple is reduced to a single ASCII bytestring.
    d0 = dict(d)
    mem_bytes = 0
    mem_bytes += d0['filter_array'].nbytes
    d0['mem_bytes'] = mem_bytes
    private = { name: d0[name] for name in SRHash_private._fields }
    d0['private'] = SRHash_private(**private)
    d1 = { name: d0[name] for name in FCFilter._fields }
    d1['hashfuncs'] = (':'.join(d1['hashfuncs'])).encode("ASCII")
    return FCFilter(**d1)
