"""
Module fastcash.srhash

This module provides
*  SRHash, a namedtuple to store hash information

It provides factory functions to build an SRHash
that are jit-compiled.

"""

from math import ceil, log2
from random import randrange
from collections import namedtuple

import numpy as np
from numpy.random import randint
from numba import njit, uint64, uint32, int64, boolean, prange

from .mathutils import bitsfor, nextodd, inversemodprime, inversemodpow2
from .hashfunctions import (
    parse_names,
    extend_func_tuple,
    )


# An SRHash namedtuble is created at the end of build_hash()
# in each concrete hash implementation. e.g. hash_3c_fbcbvb, etc.
# The definition here specifies the attributes and public methods of SRHash
# that must be implemented by any hash type implementation.
SRHash = namedtuple("SRHash", [
    ### attributes
    "hashtype",
    "choices",
    "aligned",
    "universe",
    "n",
    "subtables",
    "npages",
    "pagesize",
    "nfingerprints",
    "nvalues",
    "mem_bytes",
    "shortcutbits",
    "hashfuncs",
    "maxwalk",
    "hashtable",

    ### public API methods
    "update",  # (table, key: uint64, value: uint64) -> (int32, uint64)
    "update_existing",  # (table, key: uint64, value: uint64) -> (int32, uint64)
    "store_new",  # (table, key: uint64, value: uint64) -> (int32, uint64)
    "overwrite",  # (table, key: uint64, value: uint64) -> (int32, uint64)
    "overwrite_existing",  # (table, key: uint64, value: uint64) -> (int32, uint64)

    "get_value",  # (table, key: uint64) -> uint64
    "get_value_and_choice",  # (table, key: uint64) -> (uint64, int32)
    "find_index",  # (table, key: uint64) -> uint64;  index where key is present or uint64(-1)

    "slots_nonempty",
    "slots_with_value",
    "get_statistics",  # (table) -> tuple of histograms (arrays)
    "is_tight",  # (table) -> boolean
    "compute_shortcut_bits",  # (table) -> None
    "count_items",
    "get_items",
    ### private API methods (see below)
    "private",   # hides private API methods (e.g., h.private.get_signature_at())
    ])


SRHash_private = namedtuple("SRHash_private", [
    # private API methods, may change !
    "get_pf",  # method tuple (get_pf1, get_pf2, ...)
    "get_ps",  # method tuple (get_ps1, get_ps2, ...)
    "is_slot_empty_at",  # returns True iff the given (page, slot) is empty
    "get_signature_at",  # returns a single int, unpack with signature_to_choice_fingerprint
    "set_signature_at",
    "get_value_at",
    "set_value_at",
    "get_choicebits_at",    # get raw choice bits as stored
    "get_item_at",
    "set_item_at",
    "get_shortcutbits_at",  # get ALL shortcut bits
    "set_shortcutbit_at",   # set ONE of the shortcut bits

    "get_subtable_subkey_from_key", # (key: uint64) -> (uint64, uint64)
    "get_key_from_subtable_subkey",  # (uint64, uint64) -> uint64

    "signature_to_choice_fingerprint",  # signature -> (choice, fingerprint)
    "signature_from_choice_fingerprint",   # (choice, fingerprint) -> signature
    "get_subkey_from_page_signature",
    "get_subkey_choice_from_page_signature",
    "prefetch_page",

    # Copies of the public functions but with subtable/subkey interface
    # (only defined for hashes with subtables; otherwise None)
    "update_ssk",  # (table, subtable:uint64, subkey: uint64, value: uint64) -> (int32, uint64)
    "update_existing_ssk",  # (table, subtable:uint64, subkey: uint64, value: uint64) -> (int32, uint64)
    "store_new_ssk",  # (table, subtable:uint64, subkey: uint64, value: uint64) -> (int32, uint64)
    "overwrite_ssk",  # (table, subtable:uint64, subkey: uint64, value: uint64) -> (int32, uint64)
    "overwrite_existing_ssk",  # (table, subtable:uint64, subkey: uint64, value: uint64) -> (int32, uint64)
    ])


def create_SRHash(d):
    """Return SRHash initialized from values in dictionary d"""
    # The given d does not need to provide mem_bytes; it is computed here.
    # The given d is copied and reduced to the required fields.
    # The hashfuncs tuple is reduced to a single ASCII bytestring.
    d0 = dict(d)
    d0['mem_bytes'] = d0['hashtable'].nbytes
    d0['slots_nonempty'] = make_slots_nonempty(d)
    d0['slots_with_value'] = make_slots_with_value(d)
    private = { name: d0[name] for name in SRHash_private._fields }
    d0['private'] = SRHash_private(**private)
    d1 = { name: d0[name] for name in SRHash._fields }
    d1['hashfuncs'] = (':'.join(d1['hashfuncs'])).encode("ASCII")
    return SRHash(**d1)


## Basic functions #########################################

def get_npages(n, pagesize, fill=1.0):
    return nextodd(ceil(n/fill/pagesize))
    # must be an odd number for equidistribution
    # TODO: write a more detailed reason

def get_nfingerprints(nfingerprints, universe, npages):
    if nfingerprints < 0:
        nfingerprints = int(ceil(universe / npages))
    elif nfingerprints == 0:
        nfingerprints = 1
    return nfingerprints

def check_bits(nbits, name, threshold=64):
    if threshold < 0:
        threshold = abs(threshold)
        if nbits < threshold:
            raise RuntimeError(f"cannot deal with {nbits} < {threshold} {name} bits")
    else:
        if nbits > threshold:
            raise RuntimeError(f"cannot deal with {nbits} > {threshold} {name} bits")


# Factories / makers for checking if a slot is empty ###################

# move, remove
def make_is_slot_empty_at_v(get_value_at, get_choicebits_at):
    """
    Factory for VALUE-controlled hash table layouts.
    Return a compiled function 'is_slot_empty_at(table, page, slot)'
    that returns whether a given slot is empty (check by vaue)
    """
    @njit(nogil=True, locals=dict(b=boolean))
    def is_slot_empty_at(table, page, slot):
        """Return whether a given slot is empty (check by value)"""
        v = get_value_at(table, page, slot)
        b = (v == 0)
        return b
    return is_slot_empty_at

# Makers for get_pagestatus ####################################

# modify, remove
def make_get_pagestatus_v(pagesize,
            get_value_at, get_signature_at,
            signature_parts, signature_full):
    """
    Factory for VALUE-controlled hash tables ('_v').
    [An empty slot is indicated by value == 0].
    Return a compiled function 'get_pagestatus(table, page, fpr, choice)'.
    """
    @njit(nogil=True, locals=dict(
            page=uint64, fpr=uint64, choice=int64,
            query=uint64, slot=int64, v=uint64, s=uint64))
    def get_pagestatus(table, page, fpr, choice):
        """
        Attempt to locate a (fingerprint, choice) pair on a page,
        assuming value == 0 indicates an empty space.
        Return (int64, uint64):
        Return (slot, value) if the fingerprint 'fpr' was found,
            where 0 <= slot < pagesize.
        Return (-1, fill)    if the fingerprint was not found,
            where fill >= 0 is the number of slots already filled.
        Note: Return type is always (int64, uint64) !
        """
        query = signature_full(choice, fpr)
        for slot in range(pagesize):
            v = get_value_at(table, page, slot)
            if v == 0:
                return (-1, uint64(slot))  # free slot, only valid if tight!
            s = get_signature_at(table, page, slot)
            if s == query:
                return (slot, v)
        return (-1, uint64(pagesize))
    return get_pagestatus


# Makers for is_tight #########################################

# TODO: move and remove
def make_is_tight_v(npages, pagesize,
        get_value_at, get_signature_at, signature_parts,
        get_key, get_pf, _get_pagestatus):
    """
    Factory for VALUE-controlled hash tables ('_v').
    [Empty slots are indicated by value == 0.]
    Return compiled 'is_tight(hashtable)' function.
    """
    choices = len(get_pf)
    if choices > 3:
        raise ValueError("make_is_tight currently supports only up to 3 hash functions")
    if choices <= 1:  # hash is always tight for a single hash func.
        @njit(nogil=True)
        def is_tight(ht):
            """return (0,0) if hash is tight, or problem (key, choice)"""
            return (uint64(0), 0)
        return is_tight

    (get_pf1, get_pf2, get_pf3) = extend_func_tuple(get_pf, 3)
    (get_key1, get_key2, get_key3) = extend_func_tuple(get_key, 3)

    @njit( ###__signature__ (uint64[:],),  # infer return type
        nogil=True, locals=dict(
            page=uint64, slot=int64, v=uint64, sig=uint64, c=uint64, 
            f=uint64, key=uint64, p=uint64, s=int64, fill=uint64))
    def is_tight(ht):
        """return (0,0) if hash is tight, or problem (key, choice)"""
        for subtable in range(subtables):
            for page in range(npages):
                for slot in range(pagesize):
                    v = get_value_at(ht, subtable, page, slot)
                    if v == 0: continue
                    sig = get_signature_at(ht, subtable, page, slot)
                    (c, f) = signature_parts(sig)
                    if c == 0: continue
                    if c == 1:
                        key = get_key2(page, f)
                        (p, f) = get_pf1(key)
                        (s, fill) = _get_pagestatus(ht, subtable, p, f, 0)
                        if s >= 0 or fill != pagesize:
                            return (uint64(key), 1)  # empty slot on 1st choice
                        continue  # ok
                    if c == 2:
                        key = get_key3(page, f)
                        p, f = get_pf2(key)
                        (s, fill) = _get_pagestatus(ht, subtable, p, f, 1)
                        if s >= 0 or fill != pagesize:
                            return (uint64(key), 2)  # empty slot on 2nd choice
                        p, f = get_pf1(key)
                        (s, fill) = _get_pagestatus(ht, subtable, p, f, 0)
                        if s >= 0 or fill != pagesize:
                            return (uint64(key), 1)  # empty slot on 1st choice
                        continue  # ok
                    return (uint64(key), 9)  # should never happen, c=0,1,2
        # all done, no problems
        return (uint64(0), 0)
    return is_tight



## make_get_value functions  ################################

# TODO: remove
def make_get_value(pagesize, get_pf, _get_pagestatus, bits, get_shortcutbits_at, *, base=0):
    """
    Factory function that returns a pair of compiled functions:
    ( get_value(table, key), get_value_and_choice(table, key) );
    see their docstrings.
    """
    choices = len(get_pf)
    if choices < 1 or choices > 3:
        raise ValueError("Only 1 to 3 hash functions are supported.")
    (get_pf1, get_pf2, get_pf3) = extend_func_tuple(get_pf, 3)

    @njit( ###__signature__ uint64(uint64[:], uint64),
        nogil=True, locals=dict(
            key=uint64,
            page1=uint64, fpr1=uint64, slot1=int64, fill1=uint64,
            page2=uint64, fpr2=uint64, slot2=int64, fill2=uint64,
            page3=uint64, fpr3=uint64, slot3=int64, fill3=uint64,
            pagebits=uint32, check2=uint32, check3=uint32))
    def get_value(table, key):
        """
        Return uint64: the value for the given key,
        or 0 if the key is not present.
        """
        NOTFOUND = uint64(0)

        page1, fpr1 = get_pf1(key)
        (slot1, fill1) = _get_pagestatus(table, page1, fpr1, base+0)
        if slot1 >= 0: return fill1
        if fill1 < pagesize or choices <=1:
            return NOTFOUND
        # test for shortcut
        pagebits = get_shortcutbits_at(table, page1)  # returns all bits set if bits==0
        if not pagebits: return NOTFOUND
        check2 = pagebits & 1
        check3 = pagebits & 2 if bits >= 2 else 1

        # assert choices >= 2 (otherwise we returned above)
        if check2:
            page2, fpr2 = get_pf2(key)
            (slot2, fill2) = _get_pagestatus(table, page2, fpr2, base+1)
            if slot2 >= 0: return fill2
            if fill2 < pagesize or choices <= 2: 
                return NOTFOUND
            # test for shortcuts
            if bits != 0:
                pagebits = get_shortcutbits_at(table, page2)
                if bits == 1:
                    check3 = pagebits  # 1 or 0
                else:
                    check3 &= pagebits & 2

        # try the third choice only if necessary
        if choices <= 2 or not check3:
            return NOTFOUND
        page3, fpr3 = get_pf3(key)
        (slot3, fill3) = _get_pagestatus(table, page3, fpr3, base+2)
        if slot3 >= 0: return fill3
        return NOTFOUND


    @njit( ###__signature__ (uint64[:], uint64),  # infer return type (uint64, uint32)
        nogil=True, locals=dict(
            key=uint64,
            page1=uint64, fpr1=uint64, slot1=int64, fill1=uint64,
            page2=uint64, fpr2=uint64, slot2=int64, fill2=uint64,
            page3=uint64, fpr3=uint64, slot3=int64, fill3=uint64,
            pagebits=uint32, check2=uint32, check3=uint32))
    def get_value_and_choice(table, key):
        """
        Return (value, choice) for given key,
        where value is uint64 and choice is in {1,2,3} if key was found,
        but value==0 and choice==0 if key was not found.
        """
        NOTFOUND = (uint64(0), uint32(0))

        page1, fpr1 = get_pf1(key)
        (slot1, fill1) = _get_pagestatus(table, page1, fpr1, base+0)
        if slot1 >= 0: return (fill1, uint32(1))
        if fill1 < pagesize or choices <=1:
            return NOTFOUND
        # test for shortcut
        if bits != 0:  # this is resolved at compile time
            pagebits = get_shortcutbits_at(table, page1)
            if not pagebits: return NOTFOUND
        else:
            pagebits = 3
        check2 = pagebits & 1
        check3 = pagebits & 2 if bits >= 2 else 1

        # assert choices >= 2 (otherwise we returned above)
        if check2:
            page2, fpr2 = get_pf2(key)
            (slot2, fill2) = _get_pagestatus(table, page2, fpr2, base+1)
            if slot2 >= 0: return (fill2, uint32(2))
            if fill2 < pagesize or choices <= 2: 
                return NOTFOUND
            # test for shortcuts
            if bits != 0:
                pagebits = get_shortcutbits_at(table, page2)
                if bits == 1:
                    check3 = pagebits  # 1 or 0
                else:
                    check3 &= pagebits & 2

        # try the third choice only if necessary
        if choices <= 2 or not check3:
            return NOTFOUND
        page3, fpr3 = get_pf3(key)
        (slot3, fill3) = _get_pagestatus(table, page3, fpr3, base+2)
        if slot3 >= 0: return (fill3, uint32(3))
        return NOTFOUND

    return (get_value, get_value_and_choice)


# make_store_item functions #################################

# TODO: remove
def make_store_item(pagesize, get_pf, get_key_from_signature,
        _get_pagestatus, get_value_at, get_signature_at,
        set_value_at, set_signature_at,
        update_value, *, base=0, maxwalk=500):
    """
    Factory function that returns a compiled function
    store_item(table, key, value) -> walk_length.
    """
    choices = len(get_pf)
    if choices < 1 or choices > 3:
        raise ValueError("Only 1 to 3 hash functions are supported.")
    (get_pf1, get_pf2, get_pf3) = extend_func_tuple(get_pf, 3)
    LOCATIONS = choices * pagesize

    @njit( ###__signature__ int64(uint64[:], uint64, int64), 
        nogil=True, locals=dict(
            key=uint64, value=uint64, v=uint64,
            page1=uint64, fpr1=uint64, slot1=int64, fill1=uint64,
            page2=uint64, fpr2=uint64, slot2=int64, fill2=uint64,
            page3=uint64, fpr3=uint64, slot3=int64, fill3=uint64,
            fc=uint64, fpr=uint64, c=uint64, page=uint64,
            oldpage=uint64, lastlocation=uint64, steps=int64))
    def store_item(table, key, value):
        """
        Attempt to store given key with given value in hash table.
        Return values:
        > 0: success; number of pages visited
        < 0: failure; absolute value is number of pages visited (>=maxwalk)
        """
        oldpage = uint64(-1)
        lastlocation = uint64(-1)
        steps = 0
        while steps <= maxwalk:
            page1, fpr1 = get_pf1(key)
            if page1 != oldpage: steps += 1
            (slot1, fill1) = _get_pagestatus(table, page1, fpr1, base+0)
            if slot1 != -1:  # found on page1/choice1
                v = update_value(fill1, value)
                if v != fill1:
                    set_value_at(table, page1, slot1, v)
                return steps
            elif fill1 < pagesize:  # not found, but space available
                set_signature_at(table, page1, fill1, fpr1, base+0)
                set_value_at(table, page1, fill1, value)
                return steps
            
            if choices >= 2:
                page2, fpr2 = get_pf2(key)
                if page2 != oldpage: steps += 1
                (slot2, fill2) = _get_pagestatus(table, page2, fpr2, base+1)
                if slot2 != -1:  # found on page2/choice2
                    v = update_value(fill2, value)
                    if v != fill2: 
                        set_value_at(table, page2, slot2, v)
                    return steps
                elif fill2 < pagesize:  # not found, but space available
                    set_signature_at(table, page2, fill2, fpr2, base+1)
                    set_value_at(table, page2, fill2, value)
                    return steps
            
            if choices >= 3:
                page3, fpr3 = get_pf3(key)
                if page3 != oldpage: steps += 1
                (slot3, fill3) = _get_pagestatus(table, page3, fpr3, base+2)
                if slot3 != -1:  # found on page3/choice3
                    v = update_value(fill3, value)
                    if v != fill3:
                        set_value_at(table, page3, slot3, v)
                    return steps
                elif fill3 < pagesize:  # not found, but space available
                    set_signature_at(table, page3, fill3, fpr3, base+2)
                    set_value_at(table, page3, fill3, value)
                    return steps
            
            # We get here iff all pages are full.
            if choices <= 1:
                if steps == 0: steps = 1  # better safe than sorry
                return -steps  # only page is full: failed
            # Pick a random location; store item there and continue with evicted item.
            while True:
                location = randint(LOCATIONS)
                if location != lastlocation: break
            slot = location // choices
            c = location % choices
            if c == 0:
                page = page1; fpr = fpr1
            elif c == 1:
                page = page2; fpr = fpr2
            else:  # c == 2
                page = page3; fpr = fpr3
            xval = get_value_at(table, page, slot)
            xsig = get_signature_at(table, page, slot)
            set_signature_at(table, page, slot, fpr, base+c)
            set_value_at(table, page, slot, value)
            value = xval
            key = get_key_from_signature(page, xsig)
            lastlocation = location
            oldpage = page
            # loop again
        # maxwalk step exceeded; some item was kicked out :(
        return -steps
    return store_item


def make_get_subkey_from_page_signature(
    get_subkey, signature_to_choice_fingerprint, *, base=0):
    """
    Factory function for both VALUE- and CHOICE-controlled hashes.
    [For VALUE-controlled hashes, use base=0; for CHOICE-controlled hashes, base=1.]
    Return a compiled function 'get_key_from_signature(page, signature)'
    that returns the kmer code (key) given a page number and a signature.
    A signature is the pair (choice, fingerprint).
    """
    choices = len(get_subkey)
    if choices < 1 or choices > 4:
        raise ValueError("Only 1 to 4 hash functions are supported.")
    (get_subkey1, get_subkey2, get_subkey3, get_subkey4) = extend_func_tuple(get_subkey, 4)

    @njit(nogil=True, locals=dict(
            page=uint64, sig=uint64, c=int64, fpr=uint64, key=uint64))
    def get_subkey_from_page_signature(page, sig):
        """
        Return the kmer-code (key) for a given page and signature.
        The signature 'sig' encodes both choice and fingerprint.
        """
        (c, fpr) = signature_to_choice_fingerprint(sig)
        c = c + 1 - base
        ##assert 1 <= c <= choices
        if c == 1:
            key = get_subkey1(page, fpr)
        elif c == 2:
            key = get_subkey2(page, fpr)
        elif c == 3:
            key = get_subkey3(page, fpr)
        elif c == 4:
            key = get_subkey4(page, fpr)
        else:
            key = uint64(0)
        return key
    return get_subkey_from_page_signature


def make_get_subkey_choice_from_page_signature(
    get_subkey, signature_to_choice_fingerprint, *, base=0):
    """
    Factory function for both VALUE- and CHOICE-controlled hashes.
    [For VALUE-controlled hashes, use base=0; for CHOICE-controlled hashes, base=1.]
    Return a compiled function 'get_key_choice_from_signature(page, signature)'
    that returns the pair (key, choice), given a page number and a signature.
    A signature is the pair (choice, fingerprint).
    """    
    choices = len(get_subkey)
    if choices < 1 or choices > 4:
        raise ValueError("Only 1 to 4 hash functions are supported.")
    (get_subkey1, get_subkey2, get_subkey3, get_subkey4) = extend_func_tuple(get_subkey, 4)

    @njit(nogil=True, locals=dict(
            page=uint64, sig=uint64, c=int64, fpr=uint64, key=uint64))
    def get_subkey_choice_from_page_signature(page, sig):
        """
        Return pair (key, choice) for the given page and signature,
        where choice is in {0,1,2} or -1 when empty.
        """
        (c, fpr) = signature_to_choice_fingerprint(sig)
        c = c + 1 - base
        ##assert 1 <= c <= choices
        if c == 1:
            key = get_subkey1(page, fpr)
        elif c == 2:
            key = get_subkey2(page, fpr)
        elif c == 3:
            key = get_subkey3(page, fpr)
        elif c == 4:
            key = get_subkey4(page, fpr)
        else:
            key = uint64(0)
        return (key, c)
    return get_subkey_choice_from_page_signature


# Define maker for get_statistics ###################################
# This  stays here in srhash, because it is generic.

def make_get_statistics(control, subtables,
        choices, npages, pagesize, nvalues, shortcutbits,
        get_value_at, get_signature_at,
        signature_parts, get_shortcutbits_at):
    """
    Return a compiled function 'get_statistics(table)' that returns
    a tuple of int64[:] histograms: (valuehist, fillhist, choicehist, shortcuthist),
    """
    if control not in frozenset("cvx"):
        raise ValueError("ERROR: control parameter must be in {'c', 'v', 'x'}.")
    ctrl_choice = (control == "c")
    ctrl_value = (control == "v")
    xsubtables = max(1,subtables)
    @njit(nogil=True, locals=dict(
            page=uint64, last=int64, slot=int64, x=uint64, c=uint64, v=uint64))
    def get_statistics(table):
        """
        Return a tuple of arrays (valuehist, fillhist, choicehist, shortcuthist),
        where valuehist[v] is the number of items with value v,
        fillhist[i] is the number of pages with i items filled,
        choicehist[i] is the number of slots with choice i,
        shortcuthust[i]
        """
        valuehist = np.zeros((xsubtables, nvalues), dtype=np.int64)
        fillhist = np.zeros((xsubtables, pagesize+1), dtype=np.int64)
        choicehist = np.zeros((xsubtables, choices+1), dtype=np.int64)
        schist = np.zeros((xsubtables, 2**shortcutbits), dtype=np.int64)
        for subtable in prange(xsubtables):
            for page in range(npages):
                last = -1
                if shortcutbits != 0:
                    if subtables:
                        schist[subtable, get_shortcutbits_at(table, subtable, page)] += 1
                    else:
                        schist[subtable, get_shortcutbits_at(table, page)] += 1
                for slot in range(pagesize):
                    if ctrl_choice:
                        if subtables:
                            sig = get_signature_at(table, subtable, page, slot)
                        else:
                            sig = get_signature_at(table, page, slot)
                        c = signature_parts(sig)[0]  # no +1 !
                        choicehist[subtable, c] += 1
                        if c != 0:
                            last = slot
                            if subtables:
                                v = get_value_at(table, subtable, page, slot)
                            else:
                                v = get_value_at(table, page, slot)
                            valuehist[subtable, v] += 1
                    elif ctrl_value:
                        if subtables:
                            v = get_value_at(table, subtable, page, slot)
                        else:
                            v = get_value_at(table, page, slot)
                        valuehist[subtable, v] += 1
                        if v == 0:
                            c = 0
                        else:
                            last = slot
                            if subtables:
                                sig = get_signature_at(table, subtable, page, slot)
                            else:
                                sig = get_signature_at(table, page, slot)
                            c = 1 + signature_parts(sig)[0]  # 1+ is correct!
                        choicehist[subtable, c] += 1
                    else:
                        pass  # other controls than choice/value not implemented
                fillhist[subtable, last+1] += 1
        return (valuehist, fillhist, choicehist, schist)
    return get_statistics


def make_slots_nonempty(kwargs):
    """compile the slots_nonempty function"""
    subtables = kwargs['subtables']
    npages = kwargs['npages']
    pagesize = kwargs['pagesize']
    is_slot_empty_at = kwargs['is_slot_empty_at']
    @njit(nogil=True, locals=dict(
            page=int64, slot=int64, empty=int64))
    def slots_nonempty(ht):
        """Return number of nonempty slots"""
        empty = 0
        for subtable in range(subtables):
            for page in range(npages):
                for slot in range(pagesize):
                    empty += is_slot_empty_at(ht, subtable, page, slot)
        return int64(int64(subtables*npages*pagesize) - empty)
    return slots_nonempty


def make_slots_with_value(kwargs):
    """compile the slots_with_value function"""
    npages = kwargs['npages']
    pagesize = kwargs['pagesize']
    is_slot_empty_at = kwargs['is_slot_empty_at']
    get_value_at = kwargs['get_value_at']
    @njit(nogil=True,
        locals=dict(page=int64, slot=int64, n=int64))
    def slots_with_value(ht, myvalue):
        """Return number of slots with a specific value"""
        n = 0
        for subtable in range(subtables):
            for page in range(npages):
                for slot in range(pagesize):
                    if is_slot_empty_at(ht, subtable, page, slot): continue
                    if get_value_at(ht, subtable, page, slot) == myvalue:
                        n += 1
        return n
    return slots_with_value
