"""
xengsort.hash_2v_fbcbvb:
a hash table with two choices,
page layout is [shortcutbits][slot]+,
where slot = [signature value],
where signature = [fingerprint choice]

Useful if number of values is a power of 2 or just a little less.
There is only one choice bit (0=choice1, 1=choice2).
Value 0 is reserved for empty slots, and used to recognize them.
"""
HASHTYPE = "2v_fbcbvb"


import numpy as np
from numba import njit, uint64, int64, void

from .mathutils import bitsfor, xbitsfor, nextpower
from .bitarray import bitarray
from .hashfunctions import (
    get_hashfunctions,
    )
from .srhash import (
    create_SRHash, get_npages, get_nfingerprints,
    check_bits,
    make_is_slot_empty_at_v,
    make_get_key_from_signature, make_get_key_choice_from_signature,
    make_get_pagestatus_v,
    make_get_value, make_store_item,
    make_is_tight_v, make_get_occupancy_v,
    )


def build_hash(universe, n, pagesize,
        hashfuncs, nvalues, update_value, *,
        aligned=False, nfingerprints=-1, init=True,
        maxwalk=500, shortcutbits=0):
    """
    Allocate an array and compile access methods for a hash table.
    Return an SRHash object with the hash information.
    """

    # Basic properties
    hashtype = HASHTYPE
    choices = 2
    base = 0  # value-based hashes have base == 0
    shortcutbits = shortcutbits
    npages = get_npages(n, pagesize)
    nfingerprints = get_nfingerprints(nfingerprints, universe, npages)
    fprbits, ffprbits = xbitsfor(nfingerprints)
    choicebits = bitsfor(choices)
    sigbits = fprbits + choicebits
    valuebits = bitsfor(nvalues)
    check_bits(sigbits, "signataure")
    check_bits(valuebits, "value")
    fprmask = uint64(2**fprbits - 1)
    choicemask = uint64(2**choicebits - 1)
    sigmask = uint64(2**sigbits - 1)  # fpr + choice, no values
    entrybits = sigbits + valuebits
    neededbits = entrybits * pagesize + shortcutbits  # specific
    pagesizebits = nextpower(neededbits)  if aligned else neededbits
    tablebits = int(npages * pagesizebits)
    fprloss = pagesize * npages * (fprbits-ffprbits) / 2**23  # in MB
    print(f"# fingerprintbits: {ffprbits} -> {fprbits}; loss={fprloss:.1f} MB")
    print(f"# npages={npages}, slots={pagesize*npages}, n={n}")
    print(f"# Bits per entry: {entrybits}; per page: {neededbits} -> {pagesizebits}")
    print(f"# Table bits: {tablebits};  MB: {tablebits/2**23:.1f};  GB: {tablebits/2**33:.3f}")

    # allocate the underlying array
    if init == True:
        hasharray = bitarray(tablebits, alignment=64)  # (#bits, #bytes)
        print(f"# Allocated {hasharray.array.dtype} hash table of shape {hasharray.array.shape}")
    else:
        hasharray = bitarray(0)
    hashtable = hasharray.array  # the raw bit array
    get_bits_at = hasharray.get  # (array, startbit, nbits=1)
    set_bits_at = hasharray.set  # (array, startbit, value, nbits=1)
    prefetch = hasharray.prefetch
    
    hashfuncs, get_pf, get_key = get_hashfunctions(
        hashfuncs, choices, universe, npages, nfingerprints)
    print(f"# Final hash functions: {hashfuncs}")
    
    # Define hash table accssor methods
    @njit( ###__signature__ uint64(uint64[:], uint64, int64),
        nogil=True, locals=dict(
            page=int64, startbit=int64, v=uint64))
    def get_shortcutbits_at(table, page):
        """Return the shortcut bits at the given page."""
        if shortcutbits == 0: 
            return uint64(3)
        startbit = page * pagesizebits
        v = get_bits_at(table, startbit, shortcutbits)
        return v

    @njit(nogil=True, locals=dict(
            page=int64, slot=uint64, startbit=int64, v=uint64))
    def get_value_at(table, page, slot):
        """Return the value at the given page and slot."""
        if valuebits == 0: return 0
        startbit = page * pagesizebits + slot * entrybits + shortcutbits + sigbits
        v = get_bits_at(table, startbit, valuebits)
        return v

    @njit(nogil=True, locals=dict(
            page=int64, slot=uint64, startbit=int64, c=uint64))
    def get_choicebits_at(table, page, slot):
        """Return the choice at the given page and slot; choices start with 0."""
        startbit = page * pagesizebits + slot * entrybits + shortcutbits + fingerprintbits
        c = get_bits_at(table, startbit, choicebits)
        return c

    @njit( ###__signature__ uint64(uint64[:], uint64, int64),
        nogil=True, locals=dict(
            page=int64, slot=uint64, startbit=int64, sig=uint64))
    def get_signature_at(table, page, slot):
        """Return the signature (choice, fingerprint) at the given page and slot."""
        startbit = page * pagesizebits + slot * entrybits + shortcutbits
        sig = get_bits_at(table, startbit, sigbits)
        return sig

    @njit(nogil=True, locals=dict(page=int64))
    def prefetch_page(table, page):
        startbit = page * pagesizebits
        prefetch(table, startbit)

    @njit( ###__signature__ (uint64,),  # infer return type
        nogil=True, locals=dict(
            sig=uint64, c=uint64, fpr=uint64))
    def signature_parts(sig):
        """Return (choice, fingerprint) from signature"""
        fpr = sig >> uint64(choicebits)
        c = sig & choicemask
        return (c, fpr)

    @njit(nogil=True, locals=dict(
            sig=uint64, c=uint64, fpr=uint64))
    def signature_full(c, fpr):
        """Return signature from (choice, fingerprints)"""
        sig = (fpr << uint64(choicebits)) | c
        return sig

    @njit(nogil=True, locals=dict(
            page=int64, bit=uint64, startbit=uint64))
    def set_shortcutbit_at(table, page, bit):
        """Set ONE of the shortcut bits ('bit') at the given page."""
        if shortcutbits == 0: return
        # assert 1 <= bit <= shortcutbits
        startbit = page * pagesizebits + bit - 1
        set_bits_at(table, startbit, 1, 1)  # set exactly one bit to 1

    @njit(nogil=True, locals=dict(
            page=int64, slot=int64, fpr=uint64, choice=uint64, sig=uint64))
    def set_signature_at(table, page, slot, fpr, choice):
        """Set the signature = (choice, fpr) at the given page and slot."""
        sig = signature_full(choice, fpr)
        startbit = page * pagesizebits + slot * entrybits + shortcutbits
        set_bits_at(table, startbit, sig, sigbits)
    
    @njit(nogil=True, locals=dict(
            page=int64, slot=int64, value=int64))
    def set_value_at(table, page, slot, value):
        if valuebits == 0: return
        """Set the value at the given page and slot."""
        startbit = page * pagesizebits + slot * entrybits + sigbits + shortcutbits
        set_bits_at(table, startbit, value, valuebits)

    # define the is_slot_empty_at function
    is_slot_empty_at = make_is_slot_empty_at_v(
        get_value_at, get_choicebits_at)

    # define the get_key_from_signature function
    get_key_from_signature = make_get_key_from_signature(
        get_key, signature_parts, base=base)
    get_key_choice_from_signature = make_get_key_choice_from_signature(
        get_key, signature_parts, base=base)

    # define the _get_pagestatus function
    _get_pagestatus = make_get_pagestatus_v(pagesize,
            get_value_at, get_signature_at,
            signature_parts, signature_full)

    # define the store_item function
    store_item = make_store_item(pagesize,
            get_pf, get_key_from_signature, _get_pagestatus,
            get_value_at, get_signature_at,
            set_value_at, set_signature_at,
            update_value, base=base, maxwalk=maxwalk)

    # define the value getter functions
    (get_value, get_value_choice) = make_get_value(
            pagesize, get_pf, _get_pagestatus, 
            shortcutbits, get_shortcutbits_at, base=base)

    # define the occupancy computation function
    get_occupancy = make_get_occupancy_v(
            choices, npages, pagesize, nvalues, 
            shortcutbits,
            get_value_at, get_signature_at, 
            signature_parts, get_shortcutbits_at)

    # define the tightness test
    is_tight = make_is_tight_v(
        npages, pagesize,
        get_value_at, get_signature_at, signature_parts,
        get_key, get_pf, _get_pagestatus)

    # all methods are defined; return the hash object
    return create_SRHash(locals())
