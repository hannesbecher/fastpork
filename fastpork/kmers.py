"""
fastcash/kmers.py

Factory functions for k-mer iterators / processors / producers
"""

from numba import njit, uint8, void, int64, uint64, int32, boolean
from .dnaencode import generate_revcomp_and_canonical_code


def compile_kmer_iterator(shp, rcmode="f"):
    """
    Return (k, iterator),
    where k is the k-mer length and
    iterator is a compiled k-mer iterator (generator function)
    for the given input shape 'shp' , which can be 
    - an integer k for a contiguous shape,
    - or a tuple of growing indices, where k is the length of the tuple,
    and for the given rcmode (from {"both", "f", "r", min", "max"})
    """
    both = (rcmode == "both")
    if isinstance(shp, int):
        # special case: contiguous k-shape
        k = shp
        shp = None
    elif isinstance(shp, tuple):
        k = len(shp)
        if shp == tuple(range(k)): shp = None  # back to special case
    else:
        raise TypeError(f"shape shp={shp} must be int or k-tuple, but is {type(shp)}.")
    if k < 1 or k > 32:
        raise ValueError(f"only 1<= k <= 32 is supported, but k={k}.")
    codemask = uint64(4**(k-1) - 1)
    revcomp, ccode = generate_revcomp_and_canonical_code(k, rcmode)

    if shp is None:
        # special case: contiguous k-mer
        print(f"# processing contiguous {k}-mers")
        @njit( ###__signature__ (uint8[:], int64, int64),
            nogil=True, locals=dict(
                code=uint64, endpoint=int64, i=int64, j=int64, c=uint64))
        def kmers(seq, start, end):
            endpoint = end - (k-1)
            valid = False
            i = start
            while i < endpoint:
                if not valid:
                    code = 0
                    for j in range(k):
                        c = seq[i+j]
                        if c > 3:
                            i += j + 1  # skip invalid
                            break
                        code = (code << 2) | c
                    else:  # no break
                        valid = True
                    if not valid: continue  # with while
                else:  # was valid, we have an old code
                    c = seq[i+k-1]
                    if c > 3:
                        valid = False
                        i += k  # skip invalid
                        continue  # with while
                    code = ((code & codemask) << 2) | c
                # at this point, we have a valid code
                if both:
                    yield code
                    yield revcomp(code)
                else:
                    yield ccode(code)
                i += 1
            pass  # all done here
    else:
        # general shape: k:int and shp:tuple are set
        print(f"# processing general {k}-mers: {shp}")
        @njit( ###__signature__ (uint8[:], int64, int64),
            nogil=True, locals=dict(
                code=uint64, startpoint=int64, i=int64, j=int64, c=uint64))
        def kmers(seq, start, end):
            startpoints = (end - start) - shp[k-1]
            for i in range(start, start+startpoints):
                code = 0
                for j in shp:
                    c = seq[i+j]
                    if c > 3:
                        break
                    code = (code << 2) + c
                else:  # no break
                    if both:
                        yield code
                        yield revcomp(code)
                    else:
                        yield ccode(code)
            # all done here

    return k, kmers


def compile_kmer_subarray_iterator(k):
    """
    UNTESTED!

    Return a pair (k, kmers),
    where kmers is a compiled k-subarray iterator (generator function)
    for the given value of k,
    which yields each (valid) contiguous sub-array of a sequence.
    """
    # TODO: improve efficiency (rolling)
    @njit(nogil=True, locals=dict(
            code=uint64, startpoint=int64, i=int64, j=int64, c=uint64))
    def kmers(seq, start, end):
        startpoints = (end - start) - (k-1)
        for i in range(start, start+startpoints):
            for j in range(k):
                c = seq[i+j]
                if c > 3:
                    break
                else:  # no break
                    yield seq[i:i+k]  # should be a view
            # all done here
    return k, kmers



# efficient k-mer processor for arbitrary shapes with function injection ###

def compile_kmer_processor(shp, func, rcmode="f"):
    """
    Return (k, processor),
    where k is the k-mer length and
    processor is a compiled k-mer processor.

    The compiled k-mer processor executes a function 'func'
    for each valid k-mer of for the given shape 'shp', which can be 
    - an integer k for a contiguous shape,
    - or a tuple of growing indices, where k is the length of the tuple.

    Signature of func must be as follows:
    def func(hashtable, kmercode, param1, param2, param3, ...):
        ...
        return boolean(failure)
    Parameters param1, ... can be an arbitrary number of arrays.

    The given 'rcmode' must be from {"both", "f", "r", min", "max"}
    and specifies how to deal with reverse complementarity.
    """

    both = (rcmode == "both")
    if isinstance(shp, int):
        # special case: contiguous k-shape
        k = shp
        shp = None
    elif isinstance(shp, tuple):
        k = len(shp)
        if shp == tuple(range(k)): shp = None  # back to special case
    else:
        raise TypeError(f"shape shp={shp} must be int or k-tuple, but is {type(shp)}.")
    if k < 1 or k > 32:
        raise ValueError(f"only 1<=k<=32 is supported, but k={k}.")
    codemask = uint64(4**(k-1) - 1)
    revcomp, ccode = generate_revcomp_and_canonical_code(k, rcmode)
    
    if shp is None:
        # special case: contiguous k-mer
        ##print(f"# processing contiguous {k}-mers")
        @njit(nogil=True, locals=dict(
                code=uint64, endpoint=int64, i=int64, j=int64, c=uint64))
        def kmers(ht, seq, start, end, *parameters):
            endpoint = end - (k-1)
            valid = failed = False
            i = start
            while i < endpoint:
                if not valid:
                    code = 0
                    for j in range(k):
                        c = seq[i+j]
                        if c > 3:
                            i += j + 1  # skip invalid
                            break
                        code = (code << 2) | c
                    else:  # no break
                        valid = True
                    if not valid: continue  # with while
                else:  # was valid, we have an old code
                    c = seq[i+k-1]
                    if c > 3:
                        valid = False
                        i += k  # skip invalid
                        continue  # with while
                    code = ((code & codemask) << 2) | c
                # at this point, we have a valid code
                if both:
                    failed  = func(ht, code, *parameters)
                    failed |= func(ht, revcomp(code), *parameters)
                else:
                    failed = func(ht, ccode(code), *parameters)
                i += 1
                if failed: break
            pass  # all done here; end of def kmers(...).
    else:
        # general shape: k:int and shp:tuple are given
        ##print(f"# processing general {k}-mers: {shp}")
        @njit(nogil=True, locals=dict(
                code=uint64, startpoint=int64, i=int64, j=int64, c=uint64))
        def kmers(ht, seq, start, end, *parameters):
            startpoints = (end - start) - shp[k-1]
            failed = False
            for i in range(start, start+startpoints):
                code = 0
                for j in shp:
                    c = seq[i+j]
                    if c > 3:
                        break
                    code = (code << 2) + c
                else:  # no break
                    if both:
                        failed  = func(ht, code, *parameters)
                        failed |= func(ht, revcomp(code), *parameters)
                    else:
                        failed = func(ht, ccode(code), *parameters)
                if failed: break
            # all done here

    return k, kmers


def compile_positional_kmer_processor(shp, func, rcmode="f"):
    """
    like compile_kmer_processor, but also uses the current k-mer start position
    as an additional argument to func:
        func(hashtable, kmercode, position, *parameters)
    """
    both = (rcmode == "both")
    if isinstance(shp, int):
        # special case: contiguous k-shape
        k = shp
        shp = None
    else:
        raise TypeError(f"shape shp={shp} must be int (contiguous shape), but is {type(shp)}.")
    if k < 1 or k > 32:
        raise ValueError(f"only 1<=k<=32 is supported, but k={k}.")
    codemask = uint64(4**(k-1) - 1)
    revcomp, ccode = generate_revcomp_and_canonical_code(k, rcmode)
    
    ##print(f"# processing contiguous {k}-mers")
    @njit(nogil=True, locals=dict(
            code=uint64, endpoint=int64, i=int64, j=int64, c=uint64))
    def processor(ht, seq, start, end, *parameters):
        endpoint = end - (k-1)
        valid = failed = False
        i = start
        while i < endpoint:
            if not valid:
                code = 0
                for j in range(k):
                    c = seq[i+j]
                    if c > 3:
                        i += j + 1  # skip invalid
                        break
                    code = (code << 2) | c
                else:  # no break
                    valid = True
                if not valid: continue  # with while
            else:  # was valid, we have an old code
                c = seq[i+k-1]
                if c > 3:
                    valid = False
                    i += k  # skip invalid
                    continue  # with while
                code = ((code & codemask) << 2) | c
            # at this point, we have a valid code
            if both:
                failed  = func(ht, code, i, *parameters)
                failed |= func(ht, revcomp(code), i, *parameters)
            else:
                failed = func(ht, ccode(code), i, *parameters)
            i += 1
            if failed: break
        pass  # all done here, return None

    return k, processor

