"""
fastcash_main.py
"""

import argparse
from importlib import import_module  # dynamically import subcommand

from ._version import VERSION, DESCRIPTION
from .parameters import set_threads


def add_hash_table_arguments(p):
    p.add_argument("--pagesize", "-p", type=int, default="6",
        help="page size, i.e. number of elements on a page")
    p.add_argument("--fill", type=float, default="0.9",
        help="desired fill rate of the hash table")
    p.add_argument("--subtables", type=int, default=5, 
        help="number of subtables used; subtables+1 threads are used")


def add_hash_arguments(p):
    p.add_argument('-k', '--kmersize', metavar="INT", type=int, default=27,
        help="k-mer size")
    p.add_argument("--rcmode", metavar="MODE", default="max",
        choices=("f", "r", "both", "min", "max"),
        help="mode specifying how to encode k-mers")
    p.add_argument("--parameters", "-P", metavar="PARAMETER", nargs="+",
        help="provide parameters directly: "
            "[NOBJECTS TYPE[:ALIGNED] HASHFUNCTIONS PAGESIZE FILL], where "
            "NOBJECTS is the number of objects to be stored, "
            "TYPE[:ALIGNED] is the hash type implemented in hash_{TYPE}.py, "
            "and ALIGNED can be a or u for aligned and unaligned, "
            "HASHFUNCTIONS is a colon-separated list of hash functions, "
            "PAGESIZE is the number of elements on a page"
            "FILL is the desired fill rate of the hash table.") 
    p.add_argument("--shortcutbits", "-S", metavar="INT", type=int, choices=(0,1,2),
        help="number of shortcut bits (0,1,2), default: 0", default=0)
    # single parameter options for parameters
    p.add_argument("-n", "--nobjects", metavar="INT", type=int,
        help="number of objects to be stored")
    p.add_argument("--type", default="default",#default="3c_fbcbvb",
        help="hash type (e.g. [s]3c_fbcbvb, 2v_fbcbvb), implemented in hash_{TYPE}.py")
    p.add_argument("--unaligned", action="store_const", 
        const=False, dest="aligned", default=None,
        help="use unaligned pages (smaller, slightly slower; default)")
    p.add_argument("--aligned", action="store_const",
        const=True, dest="aligned", default=None,
        help="use power-of-two aligned pages (faster, but larger)")
    p.add_argument("--hashfunctions", "--functions", 
        help="hash functions: 'default', 'random', or func1:func2[:func3]")
    p.add_argument("--nfingerprints", type=int,
        help="override number of fingerprints (CAREFUL!)")
    # less important hash options
    p.add_argument("--nostatistics", "--nostats", action="store_true",
        help="do not compute or show index statistics at the end")
    p.add_argument("--maxwalk", metavar="INT", type=int, default=500,
        help="maximum length of random walk through hash table before failing [500]")
    p.add_argument("--maxfailures", metavar="INT", type=int, default=0, 
        help="continue even after this many failures [default:0; forever:-1]")
    p.add_argument("--walkseed", type=int, default=7,
        help="seed for random walks while inserting elements [7]")
    add_hash_table_arguments(p)


def dump(p):
    p.add_argument("--index", "-i", metavar="INDEX_HDF5", required=True,
        help="name of the index HDF5 file (input)")
    p.add_argument("--dump", "--out", "-d", "-o", metavar="DUMP_HDF5",
        help="name of the dump HDF5 file (output)") 
    p.add_argument("--text", metavar="DUMP_CSV",
        help="dump kmers to a csv ")
    p.add_argument("--values", metavar='INT', nargs='+',
        help="dump only kmers with given value(s), also for ranges, e.g. --values 1 2 5-9")
    p.add_argument("--fast", action="store_true",
        help="gain speed by not counting exact number of slots first (overallocate memory)")


def optimize(p):
    p.add_argument("--dump", "-d", metavar="DUMP_HDF5", required=True,
        help="name of the dumped index HDF5 file (input)")
    p.add_argument("--optdump", metavar="DUMP_HDF5",
        help="name of the optimized dump HDF5 file (output)")
    p.add_argument("--optindex", metavar="INDEX_HDF5",
        help="name of the optimized index HDF5 file")
    p.add_argument('--newvalueset', nargs='+',
        default=['strongunique'], metavar="PARAMETER",
        help="value set with arguments, implemented in values.{VALUESET}.py")
    p.add_argument("--pagesize", "-p", type=int,
        help="page size, i.e. number of elements on a page")
    p.add_argument("--fill", type=float,
        help="desired fill rate of the hash table")
    p.add_argument("--subtables", type=int,
        help="desired number of subtables")
    p.add_argument("--hashfunctions", "--functions",
        help="hash functions: 'default', 'random', or func1:func2:func3:[:func4]")
    p.add_argument("-t", "--threads", default=1, type=int,
        help="number of threads to optimize. Works only in combinations with subtables.s")


##### main argument parser #############################

def get_argument_parser():
    """
    return an ArgumentParser object
    that describes the command line interface (CLI)
    of this application
    """
    p = argparse.ArgumentParser(
        description = DESCRIPTION,
        epilog = "by Genome Informatics, University of Duisburg-Essen."
        )
    p.add_argument("--version", action="version", version=VERSION,
        help="show version and exit")

    subcommands = [
         ("dump",
         "dump all k-mers from a given hash table",
         dump,
         "fastcash_dump", "main"),
         ("optimize",
         "optimize the assignment of given k-mer dump",
         optimize,
         "fastcash_optimize", "main"),
    ]
    # add global options here
    # (none)
    # add subcommands to parser
    sps = p.add_subparsers(
        description="The srhash tool supports the following commands. "
            "Run 'srhash COMMAND --help' for detailed information on each command.",
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
