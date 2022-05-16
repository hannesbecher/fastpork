"""
combine_gene_tables.py:

Combine two gene tables created by create_gene_table.py,
one for ossabaw, one for minipig (with --swap-pigs),
to find genes that rank top in the first table and bottom in the second.

Typical use is to find genes that have many variants in Ossabaw,
but few in Minipig (or vice versa).

Genes that do not have regular names will be left out.
This will print only a list of gene names for use with enrichr.
"""


from argparse import ArgumentParser
from sys import stdout, stderr
from collections import namedtuple

from rich.console import Console
_console = Console()
rprint = _console.print


Record = namedtuple("Record",
    ["rank", "key1", "key2", "gene", "regionstring", "length", "nloss", "nvar"])


def load_table(fname):
    """
    Load gene table
    """
    rec = []
    with open(fname, "rt") as f:
        for line in f:
            fields = list(map(str.strip, line.strip().split('\t')))
            gene = fields[3]
            if ',' in gene:
                fields[3] = gene[:gene.find(",")]
            rec.append(Record(*fields))
    return rec


def find_gene_rank(table, name):
    for i, record in enumerate(table):
        if record.gene == name: return i
    return -1


def get_argument_parser():
    p = ArgumentParser(description="Combine gene tables")
    p.add_argument("first",
        help="first gene table")
    p.add_argument("second",
        help="second gene table")
    return p


def main(args):
    rprint(f'# Hello, this is [bold green]combine_gene_tables.py.')
    t1 = load_table(args.first)
    t2 = load_table(args.second)
    n = 0
    for i, record in enumerate(t1):
        r1 = int(record.rank)
        assert r1 == i+1
        g = record.gene
        if g.startswith(("LOC", "MIR", "TRNA")): continue
        r2 = find_gene_rank(t2, g)
        if r2 < 10000 or r2 - r1 < 2000: continue
        #print(f"{record.rank} {g} {r2}")
        print(g)
        n += 1
        if n >= 100: break


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
