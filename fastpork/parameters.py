# parameters.py: parameter manipulations
# (c) Sven Rahmann, 2021

import os
from importlib import import_module


def set_threads(args, argname="threads"):
    threads = getattr(args, argname, None)
    if threads is None: return
    threads = str(threads)
    env = os.environ
    env["OMP_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["VECLIB_MAXIMUM_THREADS"] = threads
    env["NUMEXPR_NUM_THREADS"] = threads
    env["NUMBA_NUM_THREADS"] = threads
    env["NUMBA_THREADING_LAYER"] = "omp"


def get_valueset_parameters(valueset, *, rcmode=None, k=None, strict=True):
    # process valueset
    vimport = "values." + valueset[0]
    vmodule = import_module("."+vimport, __package__)
    values = vmodule.initialize(*(valueset[1:]))
    vstr =  " ".join([vimport] + valueset[1:])
    if not rcmode: rcmode = values.RCMODE
    return (values, vstr, rcmode, k, None)


def parse_parameters(parameters, args, *, singles=True):
    if parameters is not None:
        (nobjects, hashtype, aligned, hashfuncs, pagesize, nfingerprints, fill) = parameters
    else:  # defaults
        (nobjects, hashtype, aligned, hashfuncs, pagesize, nfingerprints, fill)\
            = (10_000_000, "default", False, "random", 0, -1, 0.9)
    # process args.parameters first
    P = args.parameters
    if P:
        print(f"# Overwriting parameters with {P}")
        p = P[0]
        if p != "_":
            nobjects = int(p)
        if len(P) > 1:
            p = P[1]
            if p != "_":
                hashtype, _, al = p.partition(":")
                if al:
                    aligned = (al.lower() == "a")
        if len(P) > 2:
            p = P[2]
            if p != "_":
                hashfuncs = p
        if len(P) > 3:
            p = P[3]
            if p != "_":
                pagesize = int(p)
        if len(P) > 4:
            p = P[4]
            if p != "_":
                fill = float(p)
    # process single command line arguments
    if singles:
        if args.nobjects is not None:
            nobjects = args.nobjects
        if args.type is not None:
            hashtype = args.type
        if args.aligned is not None:
            aligned = args.aligned
        if args.hashfunctions is not None:
            hashfuncs = args.hashfunctions
        if args.pagesize is not None:
            pagesize = args.pagesize
        if args.fill is not None:
            fill = args.fill
        if args.nfingerprints is not None:
            nfingerprints = args.nfingerprints
    # pack up and return
    parameters = (nobjects, hashtype, aligned, hashfuncs, pagesize, nfingerprints, fill)
    return parameters
