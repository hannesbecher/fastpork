"""
create_genelist.py:

Take the tables for ossabaw and minipig created by create_gene_table.py,
and select "interesting" genes
(those heavily modified in Ossabaw pigs, but not in minipigs)

Sven Rahmann, 2022
"""

from argparse import ArgumentParser
from collections import namedtuple
from sys import stdout, stderr

Record = namedtuple("Record", "length nv nl dv dl names")


def dict_from_table(fname):
    d = dict()
    with open(fname, "rt") as ftable:
        for line in ftable:
            fields = line.strip().split('\t')
            idx, xdlost, xdvar, xnames, region, xlength, xnl, xnv = fields
            dlost = int(xdlost.split()[0])
            dvar = int(xdvar.split()[0])
            name = xnames.split(",")[0].strip()
            length = int(xlength.partition("=")[2])
            nl = int(xnl.partition("=")[2])
            nv = int(xnv.partition("=")[2])
            d[name] = Record(length=length, nv=nv, nl=nl, dv=dvar, dl=dlost, names=xnames)
    return d


def interesting_names(do, dm):
    names = list(do.keys())
    for name in names:
        ro = do[name]
        rm = dm[name]
        if name.startswith(("TRNA", "LOC", "MIR")): continue
        if ro.nv > 0 and rm.nv == 0:
            yield name


def get_argument_parser():
    p = ArgumentParser(description="Create a gene list from tables")
    p.add_argument('ossabaw',
        help='gene table for Ossabaw pigs')
    p.add_argument('minipig',
        help='gene table for Minipigs')
    #p.add_argument("--sortby", "-s",
    #    choices=("variants", "losses"), default="losses",
    #    help="how to sort the results")
    #p.add_argument("--swap-pigs", action="store_true",
    #    help="swap roles of Ossabaw and Minipigs (i.e. find genes for Minipigs)")
    return p


def main(args):
    do = dict_from_table(args.ossabaw)
    dm = dict_from_table(args.minipig)
    assert set(do.keys()) == set(dm.keys()), "Keys (gene names) do not agree"
    # TODO: ability to swap pigs?
    for name in interesting_names(do, dm):
        #print(do[name].names)  # with alternatives
        print(name)


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
