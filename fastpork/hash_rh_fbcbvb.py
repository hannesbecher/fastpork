"""
hash_3c_fbcbvb:
a hash table with three choices,
page layout is (low) ... [shortcutbits][slot]+ ... (high),
where slot is  (low) ... [signature value] ...     (high),
where signature is (low) [fingerprint choice] .....(high).
signature as bitmask: ccffffffffffffff (choice is at HIGH bits!)

This layout allows fast access because bits are separated.
It is memory-efficient if the number of values is a power of 2,
or just a little less.
"""

# fastpork count --fastq <(zcat data/ecoli.fq.gz) --subtables 0 -n 2262962 -k 25 -o test_s9.h5 --fill 0.9 --type rh_fbcbvb

import numpy as np
from numpy.random import randint
from numba import njit, uint64, int64, uint32, int32, boolean, void

from .mathutils import bitsfor, xbitsfor, nextpower
from .lowlevel.bitarray import bitarray
from .lowlevel.intbitarray import intbitarray
from .hashfunctions import get_hashfunctions
from .srhash import (
    create_SRHash,
    check_bits,
    get_npages,
    get_nfingerprints,
    make_get_subkey_from_page_signature,
    make_get_subkey_choice_from_page_signature,
    make_get_statistics,
    make_get_pagestatus_v,
    )


def build_hash(universe, n, subtables, pagesize, # _ for subtables
        hashfuncs, nvalues, update_value, *,
        aligned=False, nfingerprints=-1, init=True,
        maxwalk=500, shortcutbits=0, prefetch=False):
    """
    Allocate an array and compile access methods for a hash table.
    Return an SRHash object with the hash information.
    """

    # Basic properties
    assert subtables == 0
    hashtype = "rh_fbcbvb"
    choices = 7 # LSP
    base = 1
    get_subtable_subkey_from_key = None
    npages = get_npages(n, pagesize)
    nfingerprints = get_nfingerprints(nfingerprints, universe, npages)
    fprbits, ffprbits = xbitsfor(nfingerprints)
    choicebits = bitsfor(choices)
    sigbits = fprbits + choicebits
    valuebits = bitsfor(nvalues)
    check_bits(sigbits, "signataure")
    check_bits(valuebits, "value")
    if shortcutbits < 0 or shortcutbits > 2:
        print(f"# warning: illegal number {shortcutbits} of shortcutbits, using 0.")
        shortcutbits = 0

    fprmask = uint64(2**fprbits - 1)
    choicemask = uint64(2**choicebits - 1)
    sigmask = uint64(2**sigbits - 1)  # fpr + choice, no values
    slotbits = sigbits + valuebits  # sigbits: bitsfor(fpr x choice)
    neededbits = slotbits * pagesize + shortcutbits  # specific
    pagesizebits = nextpower(neededbits)  if aligned else neededbits
    tablebits = int((npages+choices) * pagesizebits)
    fprloss = pagesize * npages * (fprbits-ffprbits) / 2**23  # in MB
    print(f"# fingerprintbits: {ffprbits} -> {fprbits}; loss={fprloss:.1f} MB")
    print(f"# npages={npages}, slots={pagesize*npages}, n={n}")
    print(f"# bits per slot: {slotbits}; per page: {neededbits} -> {pagesizebits}")
    print(f"# table bits: {tablebits};  MB: {tablebits/2**23:.1f};  GB: {tablebits/2**33:.3f}")
    print(f"# shortcutbits: {shortcutbits}")

    # allocate the underlying array
    if init == True:
        hasharray = bitarray(tablebits, alignment=64)  # (#bits, #bytes)
        print(f"# allocated {hasharray.array.dtype} hash table of shape {hasharray.array.shape}")
    else:
        hasharray = bitarray(0)
    hashtable = hasharray.array  # the raw bit array
    get_bits_at = hasharray.get  # (array, startbit, nbits=1)
    set_bits_at = hasharray.set  # (array, startbit, value, nbits=1)
    hprefetch = hasharray.prefetch
    
    hashfuncs, get_pf, get_key = get_hashfunctions(
        hashfuncs, 1, universe, npages, nfingerprints)
    print(f"# final hash functions: {hashfuncs}")
    get_ps = make_getps_from_getpf(get_pf[0], 1, fprbits)
    get_key = get_key[0]
    get_key_from_subtable_subkey = None

    @njit(nogil=True, inline='always', locals=dict(
        page=int64, startbit=uint64))
    def prefetch_page(table, page):
        startbit = page * pagesizebits
        hprefetch(table, startbit)

    # Define private low-level hash table accssor methods
    @njit(nogil=True, locals=dict(
            page=int64, startbit=int64, v=uint64))
    def get_shortcutbits_at(table, page):
        """Return the shortcut bits at the given page."""
        if shortcutbits == 0:
            return uint64(3)
        startbit = page * pagesizebits
        v = get_bits_at(table, startbit, shortcutbits)
        return v

    @njit(nogil=True,  locals=dict(
            page=int64, slot=uint64, startbit=int64, v=uint64))
    def get_value_at(table, page, slot):
        """Return the value at the given page and slot."""
        if valuebits == 0: return 0
        startbit = page * pagesizebits + slot * slotbits + shortcutbits + sigbits
        v = get_bits_at(table, startbit, valuebits)
        return v

    @njit(nogil=True, locals=dict(
            page=int64, slot=uint64, startbit=int64, c=uint64))
    def get_choicebits_at(table, page, slot):
        """Return the choice at the given page and slot; choices start with 1."""
        startbit = page * pagesizebits + slot * slotbits + shortcutbits + fprbits
        c = get_bits_at(table, startbit, choicebits)
        return c

    @njit(nogil=True,  locals=dict(
            page=int64, slot=uint64, startbit=int64, sig=uint64))
    def get_signature_at(table, page, slot):
        """Return the signature (choice, fingerprint) at the given page and slot."""
        startbit = page * pagesizebits + slot * slotbits + shortcutbits
        sig = get_bits_at(table, startbit, sigbits)
        return sig

    @njit(nogil=True, locals=dict(
            page=int64, slot=uint64, startbit=int64, sig=uint64, v=uint64))
    def get_item_at(table, page, slot):
        """Return the signature (choice, fingerprint) at the given page and slot."""
        startbit = page * pagesizebits + slot * slotbits + shortcutbits
        sig = get_bits_at(table, startbit, sigbits)
        if valuebits > 0:
            v = get_bits_at(table, startbit+sigbits, valuebits)
            return (sig, v)
        return (sig, uint64(0))


    @njit(nogil=True, inline='always', locals=dict(
            sig=uint64, c=uint64, fpr=uint64))
    def signature_to_choice_fingerprint(sig):
        """Return (choice, fingerprint) from signature"""
        fpr = sig & fprmask
        c = (sig >> uint64(fprbits)) & choicemask
        return (c, fpr)

    @njit(nogil=True, inline='always', locals=dict(
            sig=uint64, choice=uint64, fpr=uint64))
    def signature_from_choice_fingerprint(choice, fpr):
        """Return signature from (choice, fingerprints)"""
        sig = (choice << uint64(fprbits)) | fpr
        return sig

    @njit(nogil=True, locals=dict(
            page=int64, bit=uint64, startbit=uint64))
    def set_shortcutbit_at(table, page, bit):
        """Set the shortcut bits at the given page."""
        if shortcutbits == 0: return
        # assert 1 <= bit <= shortcutbits
        startbit = page * pagesizebits + bit - 1
        set_bits_at(table, startbit, 1, 1)  # set exactly one bit to 1

    @njit(nogil=True, locals=dict(
            page=int64, slot=int64, sig=uint64))
    def set_signature_at(table, page, slot, sig):
        """Set the signature at the given page and slot."""
        startbit = page * pagesizebits + slot * slotbits + shortcutbits
        set_bits_at(table, startbit, sig, sigbits)
    
    @njit(nogil=True, locals=dict(
            page=int64, slot=int64, value=int64))
    def set_value_at(table, page, slot, value):
        if valuebits == 0: return
        """Set the value at the given page and slot."""
        startbit = page * pagesizebits + slot * slotbits + sigbits + shortcutbits
        set_bits_at(table, startbit, value, valuebits)

    @njit(nogil=True, locals=dict(
            page=int64, slot=int64, sig=uint64, value=uint64))
    def set_item_at(table, page, slot, sig, value):
        startbit = page * pagesizebits + slot * slotbits + shortcutbits
        set_bits_at(table, startbit, sig, sigbits)
        if valuebits == 0: return
        set_bits_at(table, startbit+sigbits, value, valuebits)


    # define the is_slot_empty_at function
    @njit(nogil=True, inline='always', locals=dict(b=boolean))
    def is_slot_empty_at(table, page, slot):
        """Return whether a given slot is empty (check by choice)"""
        c = get_choicebits_at(table, page, slot)
        b = (c == 0)
        return b

    # define the get_subkey_from_page_signature function
    get_subkey_from_page_signature = make_get_subkey_from_page_signature( #TODO change in srhash?
        (get_key,), signature_to_choice_fingerprint, base=base)
    get_subkey_choice_from_page_signature = make_get_subkey_choice_from_page_signature(
        (get_key,), signature_to_choice_fingerprint, base=base)

    # define the _find_signature_at function
    @njit(nogil=True, inline="always", locals=dict(
            page=uint64, fpr=uint64, choice=uint64,
            query=uint64, slot=int64, v=uint64, s=uint64))
    def _find_signature_at(table, page, query):
        """
        Attempt to locate signature on a page,
        assuming choice == 0 indicates an empty space.
        Return (int64, uint64):
        Return (slot, value) if the signature 'query' was found,
            where 0 <= slot < pagesize.
        Return (-1, fill) if the signature was not found,
            where fill >= 0 is the number of slots already filled.
        """
        for slot in range(pagesize):
            s = get_signature_at(table, page, slot)
            if s == query:
                v = get_value_at(table, page, slot)
                return (slot, v)
            c, _ = signature_to_choice_fingerprint(s)
            if c == 0:
                return (int64(-1), uint64(slot))  # free slot, only valid if tight!
        return (int64(-1), uint64(pagesize))

    # define the update/store/overwrite functions

    update, update_ssk \
        = make_update_by_randomwalk(pagesize, choices,
            get_ps, get_signature_at, get_value_at,
            get_item_at, set_item_at, set_value_at,
            get_subkey_from_page_signature,
            prefetch_page,signature_to_choice_fingerprint,
            signature_from_choice_fingerprint,
            is_slot_empty_at, npages,
            update_value=update_value, overwrite=False,
            allow_new=True, allow_existing=True,
            maxwalk=maxwalk, prefetch=prefetch)

    update_existing, update_existing_ssk \
        = make_update_by_randomwalk(pagesize, choices,
            get_ps, get_signature_at, get_value_at,
            get_item_at, set_item_at, set_value_at,
            get_subkey_from_page_signature,
            prefetch_page,signature_to_choice_fingerprint,
            signature_from_choice_fingerprint,
            is_slot_empty_at, npages,
            update_value=update_value, overwrite=False,
            allow_new=False, allow_existing=True,
            maxwalk=maxwalk, prefetch=prefetch)

    store_new, store_new_ssk \
        = make_update_by_randomwalk(pagesize, choices,
            get_ps, get_signature_at, get_value_at,
            get_item_at, set_item_at, set_value_at,
            get_subkey_from_page_signature,
            prefetch_page,signature_to_choice_fingerprint,
            signature_from_choice_fingerprint,
            is_slot_empty_at, npages,
            update_value=None, overwrite=True,
            allow_new=True, allow_existing=False,
            maxwalk=maxwalk, prefetch=prefetch)

    overwrite, overwrite_ssk \
        = make_update_by_randomwalk(pagesize, choices,
            get_ps, get_signature_at, get_value_at,
            get_item_at, set_item_at, set_value_at,
            get_subkey_from_page_signature,
            prefetch_page,signature_to_choice_fingerprint,
            signature_from_choice_fingerprint,
            is_slot_empty_at, npages,
            update_value=update_value, overwrite=True,
            allow_new=True, allow_existing=True,
            maxwalk=maxwalk, prefetch=prefetch)

    overwrite_existing, overwrite_existing_ssk \
        = make_update_by_randomwalk(pagesize, choices,
            get_ps, get_signature_at, get_value_at,
            get_item_at, set_item_at, set_value_at,
            get_subkey_from_page_signature,
            prefetch_page,signature_to_choice_fingerprint,
            signature_from_choice_fingerprint,
            is_slot_empty_at, npages,
            update_value=update_value, overwrite=True,
            allow_new=False, allow_existing=True,
            maxwalk=maxwalk, prefetch=prefetch)


    # define the "reading" functions find_index, get_value, etc.

    @njit(nogil=True, locals=dict(
            key=uint64, default=uint64, NOTFOUND=uint64,
            page1=uint64, sig1=uint64, slot1=int64, val1=uint64,
            page2=uint64, sig2=uint64, slot2=int64, val2=uint64,
            page3=uint64, sig3=uint64, slot3=int64, val3=uint64,
            pagebits=uint32, check2=uint32, check3=uint32))
    def find_index(table, key, default=uint64(-1)):
        """
        Return uint64: the linear table index the given key,
        or the default if the key is not present.
        """
        NOTFOUND = uint64(default)
        page1, sig1 = get_ps1(key)
        (slot1, val1) = _find_signature_at(table, page1, sig1)
        if slot1 >= 0: return uint64(uint64(page1 * pagesize) + slot1)
        if val1 < pagesize: return NOTFOUND
        # test for shortcut
        pagebits = get_shortcutbits_at(table, page1)  # returns all bits set if bits==0
        if not pagebits: return NOTFOUND
        check2 = pagebits & 1
        check3 = pagebits & 2 if shortcutbits >= 2 else 1

        if check2:
            page2, sig2 = get_ps2(key)
            (slot2, val2) = _find_signature_at(table, page2, sig2)
            if slot2 >= 0: return uint64(uint64(page2 * pagesize) + slot2)
            if val2 < pagesize: return NOTFOUND
            # test for shortcuts
            if shortcutbits != 0:
                pagebits = get_shortcutbits_at(table, page2)
                if shortcutbits == 1:
                    check3 = pagebits  # 1 or 0
                else:
                    check3 &= pagebits & 2

        # try the third choice only if necessary
        if not check3: return NOTFOUND
        page3, sig3 = get_ps3(key)
        (slot3, val3) = _find_signature_at(table, page3, sig3)
        if slot3 >= 0: return uint64(uint64(page3 * pagesize) + slot3)
        return NOTFOUND


    @njit(nogil=True, locals=dict(
            key=uint64, default=uint64, NOTFOUND=uint64,
            page1=uint64, sig1=uint64, slot1=int64, val1=uint64,
            page2=uint64, sig2=uint64, slot2=int64, val2=uint64,
            page3=uint64, sig3=uint64, slot3=int64, val3=uint64,
            pagebits=uint32, check2=uint32, check3=uint32))
    def get_value(table, key, default=uint64(0)):
        """
        Return uint64: the value for the given key,
        or the default if the key is not present.
        """
        NOTFOUND = uint64(default)
        page1, sig1 = get_ps1(key)
        (slot1, val1) = _find_signature_at(table, page1, sig1)
        if slot1 >= 0: return val1
        if val1 < pagesize: return NOTFOUND
        # test for shortcut
        pagebits = get_shortcutbits_at(table, page1)  # returns all bits set if bits==0
        if not pagebits: return NOTFOUND
        check2 = pagebits & 1
        check3 = pagebits & 2 if shortcutbits >= 2 else 1

        if check2:
            page2, sig2 = get_ps2(key)
            (slot2, val2) = _find_signature_at(table, page2, sig2)
            if slot2 >= 0: return val2
            if val2 < pagesize: return NOTFOUND
            # test for shortcuts
            if shortcutbits != 0:
                pagebits = get_shortcutbits_at(table, page2)
                if shortcutbits == 1:
                    check3 = pagebits  # 1 or 0
                else:
                    check3 &= pagebits & 2

        # try the third choice only if necessary
        if not check3: return NOTFOUND
        page3, sig3 = get_ps3(key)
        (slot3, val3) = _find_signature_at(table, page3, sig3)
        if slot3 >= 0: return val3
        return NOTFOUND


    @njit(nogil=True, locals=dict(
            key=uint64, default=uint64,
            page1=uint64, sig1=uint64, slot1=int64, val1=uint64,
            page2=uint64, sig2=uint64, slot2=int64, val2=uint64,
            page3=uint64, sig3=uint64, slot3=int64, val3=uint64,
            pagebits=uint32, check2=uint32, check3=uint32))
    def get_value_and_choice(table, key, default=uint64(0)):
        """
        Return (value, choice) for given key,
        where value is uint64 and choice is in {1,2,3} if key was found,
        but value=default and choice=0 if key was not found.
        """
        NOTFOUND = (uint64(default), uint32(0))
        page1, sig1 = get_ps1(key)
        (slot1, val1) = _find_signature_at(table, page1, sig1)
        if slot1 >= 0: return (val1, uint32(1))
        if val1 < pagesize: return NOTFOUND
        # test for shortcut
        pagebits = get_shortcutbits_at(table, page1)  # returns all bits set if bits==0
        if not pagebits: return NOTFOUND
        check2 = pagebits & 1
        check3 = pagebits & 2 if shortcutbits >= 2 else 1

        if check2:
            page2, sig2 = get_ps2(key)
            (slot2, val2) = _find_signature_at(table, page2, sig2)
            if slot2 >= 0: return (val2, uint32(2))
            if val2 < pagesize: return NOTFOUND
            # test for shortcuts
            if shortcutbits != 0:
                pagebits = get_shortcutbits_at(table, page2)
                if shortcutbits == 1:
                    check3 = pagebits  # 1 or 0
                else:
                    check3 &= pagebits & 2

        # try the third choice only if necessary
        if not check3: return NOTFOUND
        page3, sig3 = get_ps3(key)
        (slot3, val3) = _find_signature_at(table, page3, sig3)
        if slot3 >= 0: return (val3, uint32(3))
        return NOTFOUND


    @njit(nogil=True, locals=dict(
            page=uint64, slot=int64, v=uint64, sig=uint64, c=uint64,
            f=uint64, key=uint64, p=uint64, s=int64, fill=uint64))
    def is_tight(ht):
        """
        Return (0,0) if hash is tight, or problem (key, choice).
        In the latter case, it means that there is an empty slot
        for key 'key' on page choice 'choice', although key is
        stored at a higher choice.
        """
        for page in range(npages):
            for slot in range(pagesize):
                sig = get_signature_at(ht, page, slot)
                (c, f) = signature_to_choice_fingerprint(sig)  # should be in 0,1,2,3.
                if c <= 1: continue
                # c >= 2
                key = get_key2(page, f)
                p, s = get_ps1(key)
                (slot, val) = _find_signature_at(ht, p, s)
                if slot >= 0 or val != pagesize:
                    return (uint64(key), 1)  # empty slot on 1st choice
                if c >= 3:
                    key = get_key3(page, f)
                    p, s = get_ps2(key)
                    (slot, val) = _find_signature_at(ht, p, s)
                    if slot >= 0 or val != pagesize:
                        return (uint64(key), 2)  # empty slot on 2nd choice
                if c >= 4:
                    return (uint64(key), 9)  # should never happen, c=1,2,3.
        # all done, no problems
        return (0, 0)

    @njit(nogil=True, locals=dict(counter=uint64))
    def count_items(ht, filter_func):
        """
        ht: uint64[:]  # the hash table
        filter_func(key: uint64, value: uint64) -> bool  # function
        Return number of items satisfying the filter function (uint64).
        """
        counter = 0
        for p in range(npages):
            for s in range(pagesize):
                if is_slot_empty_at(ht, p, s):  continue
                sig = get_signature_at(ht, p, s)
                value = get_value_at(ht, p, s)
                key = get_subkey_from_page_signature(p, sig)
                if filter_func(key, value):
                    counter += 1
        return counter

    @njit(nogil=True, locals=dict(pos=uint64))
    def get_items(ht, filter_func, buffer):
        """
        ht: uint64[:]  # the hash table
        filter_func: bool(key(uint64), value(uint64))
        buffer: buffer to store keys (filled till full)
        Return the number of items satisfying the filter function.
        """
        B = buffer.size
        pos = 0
        for p in range(npages):
            for s in range(pagesize):
                if is_slot_empty_at(ht, p, s):  continue
                sig = get_signature_at(ht, p, s)
                value = get_value_at(ht, p, s)
                key = get_subkey_from_page_signature(p, sig)
                if filter_func(key, value):
                    if pos < B:
                        buffer[pos] = key
                    pos += 1
        return pos

    # define the occupancy computation function
    get_statistics = make_get_statistics("c", subtables,
        choices, npages+choices, pagesize, nvalues, shortcutbits,
        get_value_at, get_signature_at,
        signature_to_choice_fingerprint, get_shortcutbits_at)


    # define the compute_shortcut_bits fuction,
    # depending on the number of shortcutbits
    if shortcutbits == 0:
        @njit
        def compute_shortcut_bits(table):
            pass
    elif shortcutbits == 1:
        @njit
        def compute_shortcut_bits(table):
            for page in range(npages):
                for slot in range(pagesize):
                    if is_slot_empty_at(table, page, slot):
                        continue
                    key, c = get_subkey_choice_from_page_signature(
                        page, get_signature_at(table, page, slot))
                    assert c >= 1
                    if c == 1: continue  # first choice: nothing to do
                    # treat c >= 2
                    firstpage, _ = get_pf1(key)
                    set_shortcutbit_at(table, firstpage, 1)
                    if c >= 3:
                        secpage, _ = get_pf2(key)
                        set_shortcutbit_at(table, secpage, 1)
    elif shortcutbits == 2:
        @njit
        def compute_shortcut_bits(table):
            for page in range(npages):
                for slot in range(pagesize):
                    if is_slot_empty_at(table, page, slot):
                        continue
                    key, c = get_subkey_choice_from_page_signature(
                        page, get_signature_at(table, page, slot))
                    assert c >= 1
                    if c == 1:
                        continue
                    if c == 2:
                        firstpage, _ = get_pf1(key)
                        set_shortcutbit_at(table, firstpage, 1)
                        continue
                    # now c == 3:
                    firstpage, _ = get_pf1(key)
                    set_shortcutbit_at(table, firstpage, 2)
                    secpage, _ = get_pf2(key)
                    set_shortcutbit_at(table, secpage, 2)
    else:
        raise ValueError(f"illegal number of shortcutbits: {shortcutbits}")

    # all methods are defined; return the hash object
    return create_SRHash(locals())


#######################################################################


def make_getps_from_getpf(get_pfx, choice, fprbits):
    @njit(nogil=True, inline='always', locals=dict(
            p=uint64, f=uint64, sig=uint64))
    def get_psx(code):
        (p, f) = get_pfx(code)
        sig = uint64((choice << uint64(fprbits)) | f)
        return (p, sig)
    return get_psx


def make_update_by_randomwalk(pagesize, choices,
            get_ps, get_signature_at, get_value_at,
            get_item_at, set_item_at,
            set_value_at,
            get_subkey_from_page_signature,
            prefetch_page,
            signature_to_choice_fingerprint,
            signature_from_choice_fingerprint,
            is_slot_empty_at, npages,
            *,
            update_value=None, overwrite=False,
            allow_new=False, allow_existing=False,
            maxwalk=1000, prefetch=False):
    """return a function that stores or modifies an item"""
    LOCATIONS = choices * pagesize
    if LOCATIONS < 2:
        raise ValueError(f"ERROR: Invalid combination of pagesize={pagesize} * choices={choices}")
    if (update_value is None or overwrite) and allow_existing:
        update_value = njit(
            nogil=True, locals=dict(old=uint64, new=uint64)
            )(lambda old, new: new)
    if not allow_existing:
        update_value = njit(
            nogil=True, locals=dict(old=uint64, new=uint64)
            )(lambda old, new: old)

    @njit(nogil=True, locals=dict(
            key=uint64, value=uint64, v=uint64,
            page1=uint64, sig1=uint64, slot1=int64, val1=uint64,
            page2=uint64, sig2=uint64, slot2=int64, val2=uint64,
            page3=uint64, sig3=uint64, slot3=int64, val3=uint64,
            fc=uint64, fpr=uint64, c=uint64, page=uint64, lsp=uint64,
            oldpage=uint64, lastlocation=uint64, steps=uint64,
            xsig=uint64, xval=uint64))
    def update(table, key, value):
        """
        Attempt to store given key with given value in hash table.
        If the key exists, the existing value may be updated or overwritten,
        or nothing may happen, depending on how this function was compiled.
        If the key does not exist, it is stored with the provided value,
        or nothing may happen, depending on how this function was compiled.

        Returns (status: int32, result: uint64).

        status: if status == 0, the key was not found,
            and, if allow_new=True, it could not be inserted either.
            If (status & 127 =: c) != 0, the key exists or was inserted w/ choice c.
            If (status & 128 != 0), the key was aleady present.

        result: If the key was already present (status & 128 != 0),
            and result is the new value that was stored.
            Otherwise, result is the walk length needed
            to store the new key, value pair.
        """
        oldpage = uint64(-1)
        lastlocation = uint64(-1)
        steps = 0
        page, sig = get_ps(key)
        lsp, fpr = signature_to_choice_fingerprint(sig)
        assert lsp == 1 # 0 is a special value that indicates that the slot is empty
        # sig = signature_from_choice_fingerprint(lsp, fpr)
        while steps <= maxwalk and page < npages+choices:

            sig = signature_from_choice_fingerprint(lsp, fpr)
            # check each slot on page
            min_slot = -1
            min_lsp = 8
            # check all slots on page
            for slot in range(pagesize):
                steps += 1
                # found an empty slot; insert element
                if is_slot_empty_at(table, page, slot):
                    set_item_at(table, page, slot, sig, value)
                    return (int32(1), steps)

                c_sig = get_signature_at(table, page, slot)
                c_lsp, c_fpr = signature_to_choice_fingerprint(sig)
                # found element; Update value
                if c_sig == sig:
                    val = get_value_at(table, page, slot)
                    v = update_value(val, value)
                    if v != val: set_value_at(table, page, slot, v)
                    return (int32(128|1), steps)

                # check if there exists an element assigned to this page with a lower lsp
                if c_lsp < lsp and c_lsp < min_lsp:
                    min_slot = slot
                    min_lsp = c_lsp

            # Element is not in this page and there is no empty slot

            # can we swap the current element with one with a lower lsp?
            # on the current page was at least one element with a lower lsp
            if min_slot != -1:
                new_sig, new_val = get_item_at(table, page, min_slot)
                set_item_at(table, page, min_slot, sig, value)
                c, fpr = signature_to_choice_fingerprint(new_sig)
                assert c == min_lsp
                lsp = min_lsp

            # increase lsp and check next page
            page += 1
            lsp += 1

            if lsp > choices:
                print("min_lsp: ", min_lsp)
                break
                # we reached the maximum lsp

        # maxwalk step exceeded; some item was kicked out :(
        print("steps: ", steps)
        return (int32(0), steps)
    return update, None


#######################################################################
## Module-level functions
#######################################################################
## define the fill_from_dump function

def fill_from_arrays(h, k, nkmers, codes, achoices, values):
    npages = h.npages
    pagesize = h.pagesize
    (get_pf1, get_pf2, get_pf3) = h.private.get_pf
    set_signature_at = h.private.set_signature_at
    set_value_at = h.private.set_value_at
    is_slot_empty_at = h.private.is_slot_empty_at
    signature_from_choice_fingerprint = h.private.signature_from_choice_fingerprint
    choices = intbitarray(nkmers, 2, init=achoices)
    acodes = codes.array
    avalues = values.array
    get_code = codes.get
    get_value = values.get
    get_choice = choices.get

    @njit(nogil=True, locals=dict(choice=uint64))
    def _insert_elements(ht):
        total = 0
        for i in range(nkmers):
            total += 1
            code = get_code(acodes, i)
            value = get_value(avalues, i)
            choice = get_choice(achoices, i)
            assert choice >= 1
            if choice == 1:
                page, fpr = get_pf1(code)
            elif choice == 2:
                page, fpr = get_pf2(code)
            elif choice == 3:
                page, fpr = get_pf3(code)
            else:
                assert False
            for slot in range(pagesize):
                if is_slot_empty_at(ht, page, slot): break
            else:
                assert False
            sig =  signature_from_choice_fingerprint(choice, fpr)
            set_signature_at(ht, page, slot, sig)
            set_value_at(ht, page, slot, value)
        return total

    total = _insert_elements(h.hashtable)
    walklength = np.zeros(h.maxwalk+5, dtype=np.uint64)
    walklength[0] = total
    return (total, 0, walklength)


def make_calc_shortcut_bits_old(h): #TODO Remove?
    choices = len(h.get_pf)
    if choices != 3:
        raise ValueError("shortcut bits only implemented for 3 hash functions")
    bits = h.shortcutbits  # 0, 1 or 2
    is_slot_empty_at = h.is_slot_empty_at
    signature_parts = h.signature_parts
    get_signature_at = h.get_signature_at
    set_shortcutbit_at = h.set_shortcutbit_at
    (get_pf1, get_pf2, get_pf3) = h.get_pf
    get_key_choice_from_signature = h.get_key_choice_from_signature
    npages = h.npages
    pagesize = h.pagesize

    if bits < 0 or bits > 2: 
        print("# WARNING: Illegal number of shortcut bits; using 0")
        bits = 0

    if bits == 0:
        @njit
        def calc_shortcut_bits(table):
            pass
        return calc_shortcut_bits

    if bits == 1:
        @njit
        def calc_shortcut_bits(table):
            for page in range(npages):
                for slot in range(pagesize):
                    if is_slot_empty_at(table, page, slot):
                        continue
                    key, c = get_key_choice_from_signature(
                        page, get_signature_at(table, page, slot))
                    assert c >= 1
                    if c == 1: continue  # first choice: nothing to do
                    # treat c >= 2
                    firstpage, _ = get_pf1(key)
                    set_shortcutbit_at(table, firstpage, 1)
                    if c >= 3:
                        secpage, _ = get_pf2(key)
                        set_shortcutbit_at(table, secpage, 1)
        return calc_shortcut_bits
    
    # bits == 2
    @njit
    def calc_shortcut_bits(table):
        for page in range(npages):
            for slot in range(pagesize):
                if is_slot_empty_at(table, page, slot):
                    continue
                key, c = get_key_choice_from_signature(
                    page, get_signature_at(table, page, slot))
                assert c >= 1
                if c == 1:
                    continue
                if c == 2:
                    firstpage, _ = get_pf1(key)
                    set_shortcutbit_at(table, firstpage, 1)
                    continue
                # now c == 3:
                firstpage, _ = get_pf1(key)
                set_shortcutbit_at(table, firstpage, 2)
                secpage, _ = get_pf2(key)
                set_shortcutbit_at(table, secpage, 2)
    return calc_shortcut_bits

# 0: 2765848
# 1: 2237435 (98.872%)
# 2: 23878 (1.055%)
# 3: 1493 (0.066%)
# 4: 132 (0.006%)
# 5: 15 (0.001%)
# 6: 8 (0.000%)
# 7: 1 (0.000%)
