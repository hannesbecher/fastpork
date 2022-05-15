"""
fastpork/main.py
"""

import argparse
from importlib import import_module  # dynamically import subcommand

from ._version import VERSION, DESCRIPTION
from .parameters import set_threads
from .fastcash_main import (
    add_hash_table_arguments,
    add_hash_arguments,
    dump,
    optimize,
    )


def count(p):
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument("--fastq", "-q", metavar="FASTQ", nargs="+",
        help="FASTQ file(s) in which to count k-mers")
    s.add_argument("--fasta", "-f", metavar="FASTA", nargs="+",
        help="FASTA file(s) in which to count k-mers")
    p.add_argument("-o", "--out", metavar="OUT_HDF5", required=True,
        help="name of output file with k-mer counts")
    p.add_argument('-v', '--valueset', nargs='+',
        default=['count', '32'], metavar="PARAMETER",
        help="value set with arguments ['count 32']")
    p.add_argument("--subsample", type=int, metavar="S", default=1,
        help="only count each S-th read in FASTQ input (for huge datasets)")
    p.add_argument("-C", "--chunksize", metavar="FLOAT_SIZE_MB",
        type=float, default=8.0,
        help="chunk size in MB [default: 8.0]; one chunk is allocated per thread.")
    p.add_argument("-R", "--chunkreads", metavar="INT", type=int,
        help="maximum number of reads per chunk per thread [SIZE_MB * 2**20 / 200]")
    p.add_argument("--substatistics", action="store_true",
        help="show the statistics for all subtables")
    add_hash_arguments(p)  # k, rcmode


def filtercount(p):
    p.add_argument("--filter", metavar="FASTQ", nargs="+",
        help="FASTQ file(s) from which to filter k-mers")
    p.add_argument("--filtersize1", "--filtersize", "-f1", "-s1", "-s",
        metavar="GB", type=float, default=5.0,
        help="filter1 size in gigabytes [5.0]")
    p.add_argument("--filtersize2", "-f2", "-s2",
        metavar="GB", type=float,  # no default!
        help="filter2 size in gigabytes")
    p.add_argument("--filtersize3", "-f3", "-s3",
        metavar="GB", type=float,  # no default!
        help="filter3 size in gigabytes")
    count(p)


def debug(p):
    p.add_argument('-k', '--kmersize', metavar="INT", type=int, default=27,
        help="k-mer size")
    p.add_argument("--rcmode", metavar="MODE", default="max",
        choices=("f", "r", "both", "min", "max"),
        help="mode specifying how to encode k-mers")
    p.add_argument("--fastq", "-q", metavar="FASTQ", nargs="+",
        help="FASTQ file(s) to index and count",
        required=True)
    p.add_argument("-o", "--out", metavar="OUT_HDF5",
        help="name of output file (dummy, unused)")


##### main argument parser #############################

def get_argument_parser():
    """
    return an ArgumentParser object
    that describes the command line interface (CLI)
    of this application
    """
    p = argparse.ArgumentParser(
        description = DESCRIPTION,
        epilog = "by Algorithmic Bioinformatics, Saarland University."
        )
    p.add_argument("--version", action="version", version=VERSION,
        help="show version and exit")

    subcommands = [
         ("count",
         "count k-mers in FASTQ files (serial, or parallel using subtables)",
         count,
         "count", "main"),
         ("filtercount",
         "filter and count k-mers in FASTQ files (serial, or parallel using subtables)",
         filtercount,
         "filtercount", "main"),
         # imported:
         ("dump",
         "dump all k-mers from a given hash table",
         dump,
         ".fastcash_dump", "main"),
         ("optimize",
         "optimize the assignment of given k-mer dump",
         optimize,
         ".fastcash_optimize", "main"),
    ]
    # add global options here
    # (none)
    # add subcommands to parser
    sps = p.add_subparsers(
        description="The fastpork application supports the following commands. "
            "Run 'fastpork COMMAND --help' for detailed information on each command.",
        metavar="COMMAND")
    sps.required = True
    sps.dest = 'subcommand'
    for (name, helptext, f_parser, module, f_main) in subcommands:
        if name.endswith('!'):
            name = name[:-1]
            chandler = 'resolve'
        else:
            chandler = 'error'
        sp = sps.add_parser(name, help=helptext,
            description=f_parser.__doc__, conflict_handler=chandler)
        sp.set_defaults(func=(module,f_main))
        f_parser(sp)
    return p


def main(args=None):
    p = get_argument_parser()
    pargs = p.parse_args() if args is None else p.parse_args(args)
    set_threads(pargs, "threads")  # limit number of threads in numba/prange
    (module, f_main) = pargs.func
    m = import_module("."+module, __package__)
    mymain = getattr(m, f_main)
    mymain(pargs)
