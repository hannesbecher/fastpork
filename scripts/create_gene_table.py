"""
create_gene_table.py:

For a chromosome or interval or position of interest,
load the zarr dataset produced by genomewide_analysis.py.
Find the variants in the region of interest and output details on screen.

Sven Rahmann, 2022

Example for JAK1, JAK2, STAT3:
-r 6:147320288-147567188 1:216849744-217002310 12:20407316-20471091
"""


from argparse import ArgumentParser
from sys import stdout, stderr

import numpy as np
from numba import njit, uint64, int64, uint8
from zarrutils import save_to_zgroup, load_from_zgroup, get_zdataset

from rich.console import Console
_console = Console()
rprint = _console.print



_DNA = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T',
    4: 'N',
}



def load_regions(fname, read_exons=False):
    """
    Load regions strings from a text file; one per line.
    Each such region string `rs` must be parsed by `parse_region(rs)`.
    """
    regions = []
    genes = []
    exons = []
    with open(fname, "rt") as f:
        for i, line in enumerate(f):
            try:
                rs, genename, *myexons = line.strip().split()
            except ValueError:
                print(f"Line {i+1}: {line}", file=stderr)
                raise
            regions.append(parse_region(rs))
            genes.append(genename)
            if not read_exons: continue

            # TODO: read exons
            nexons = int(myexons[0])
            assert len(myexons) == nexons + 1
            exons.append( [parse_region(x) for x in myexons[1:]] )
    return regions, genes, exons


def parse_region(regionstr):
    """
    Given a genomic a region string (chr:start-end),
    return the tuple (chr:str, start:int, end:int).
    The fields start or end
    (or both, together with the dash '-') may be omitted.

    Instead of specifying end, one can specify a length,
    using '+' instead of '-', so for example
    '1:1234567+200' == '1:1234567:1234766'.
    """
    start, end = 1, 0
    colon = regionstr.find(':')
    if colon < 0:
        ch = regionstr
        return (ch, start, end)
    ch = regionstr[:colon]
    if len(ch) == 0:
        raise ValueError(f"Invalid region format: '{regionstr}'")
    intervalstr = regionstr[colon+1:].strip()
    if len(intervalstr) == 0:
        return (ch, start, end)
    dash = intervalstr.find("-")
    if dash >= 0:
        startstr = intervalstr[:dash]
        endstr = intervalstr[dash+1:]
        if len(startstr):
            start = int(startstr)
        if len(endstr):
            end = int(endstr)
        return (ch, start, end)
    plus = intervalstr.find("+")
    if plus >= 0:
        startstr = intervalstr[:plus]
        lenstr = intervalstr[plus+1:]
        if len(startstr):
            start = int(startstr)
        if len(lenstr):
            end = start + int(lenstr) - 1
        return (ch, start, end)
    # single position after colon
    start = end = int(intervalstr)
    return (ch, start, end)



@njit
def find_minima(k, co, cm, start, end, mins):
    n = co.size
    M = mins.shape[0]
    m = 0
    minstart = n+1
    mid = False
    for i in range(start, end):
        left = (co[i-1] > co[i]) if i > 0 else False
        if left:
            minstart = i
            mid = False
        else:
            leftx = (co[i-1] < co[i]) if i > 0 else True
            if leftx:
                minstart = n+1
                mid = False
        #mid |= co[i] < cm[i]
        mid |= cm[i] == k
        right = (co[i+1] > co[i]) if i < n-1 else False
        if minstart <= i and right and mid:
            # found a local minimum:
            # minstart .. i, value co[i]
            if m >= M:
                ##print("recalloc:", M, 2*M, "Mbytes:", 2*M*3*8/1e6)
                mins2 = np.zeros((2*M, 3), dtype=mins.dtype)
                mins2[:M,:] = mins[:M,:]
                mins = mins2
                M *= 2
            mins[m,0] = minstart
            mins[m,1] = i
            mins[m,2] = co[i]
            m += 1
    return mins, m



def show_variants(region, gene, k, seq, co, cm, pc, DNA=_DNA):
    ch, start, end = region
    if end == 0: end = len(seq)
    start0, end0 = start - 1, end
    ##maxell = 0
    #rprint(f"# Looking for locally interesting positions in chromosome {ch}: {start}-{end} (length {end0-start0}, gene {gene})...")
    #rprint(f"# Looking for SNPs in chromosome {ch}: {start}-{end} (length {end0-start0})...")
    ##print("# Finding minima")
    mins = np.zeros((2**16, 3), dtype=np.int64)
    mins, nv = find_minima(k, co, cm, start0, end0, mins)
    ##print(f"# mins: {mins.size}")
    vmbp = int(round(nv/(end0-start0) * 1_000_000))
    nloss = 0
    for i in range(nv):
        a, b, v = mins[i,:]
        if v != 0: continue
        ell = b - a + 1
        nloss += ell
    lmbp = int(round(nloss/(end0-start0) * 1_000_000))
    #    if ell >= 100:
    #        #rprint(f"    at (0-based) {a}, length {ell}: value {v}")
    #        maxell = max(maxell, ell)
    #rprint(f"    Maximum length event: {maxell}")
    #    rprint(co[max(0,a-27):b+27])
    #    rprint(cm[max(0,a-27):b+27])
    #    rprint(seq[max(0,a-27):b+27] & 3)


    #for i in snps[:msnp]:
    #    # position i is 0-based, but we output 1-based positions.
    #    #rprint(f"{i+1}: {DNA[seq[i]]} SNP {status[i-1:i+2]}")
    #rprint(f"==> {msnp} SNP candidates found.")

    #inss = snps
    #inss, mins = find_inss(status, start0, end0, inss)
    #rprint(f"==> {mins} INS candidates found.")
    return vmbp, nv, lmbp, nloss


@njit(locals=dict(c=uint8))
def fill_bitvector(mins, nv, mvec):
    for i in range(nv):
        a, b, v = mins[i,:]
        mvec[a] |= uint8(4)  # variant starts
        c = uint8(3) if v==0 else uint8(2)  # loss+present / present
        mvec[a:(b+1)] |= c


def accumulate_exons(region, genes, exons, byloss,
        k, seq, co, cm, paircount):
    # we have '-r all' AND a filter!
    # Extend L by (genes, exons) matching ch, where:
    # genes is a list of gene names, exons is a list of lists of exon regions
    ##print(f"# accumulate_exons: {region=}, {k=}, {byloss=}")

    # Get bit vector of local minima
    ch, start, end = region
    if end == 0: end = len(seq)
    start0, end0 = start - 1, end
    mins = np.zeros((2**19, 3), dtype=np.int64)
    mins, nv = find_minima(k, co, cm, start0, end0, mins)
    ##print(f"# local minima: {nv}")
    mvec = np.zeros(len(seq), dtype=np.uint8)
    fill_bitvector(mins, nv, mvec)

    # Go through all the genes + their exons
    ##gc = ec = 0
    for genename, gex in zip(genes, exons):
        if not len(gex): continue
        if gex[0][0] != ch: continue
        nvar = nloss = length = 0  # number of exonic variants, exonic losses
        for _, exstart, exend in gex:
             exstart0, exend0 = exstart - 1, exend
             nvar += (mvec[exstart0] != 0) + np.sum((mvec[(exstart0+1):exend]&4) != 0)
             nloss += int(np.sum(mvec[exstart0:exend] & 1))
             length += exend0 - exstart0
        lmbp = int(round(nloss/length * 1_000_000))
        vmbp = int(round(nvar/length * 1_000_000))
        genereg = (ch, gex[0][1], gex[-1][2])
        entry = (lmbp, nloss, vmbp, nvar, genename, genereg, length) if byloss \
            else (vmbp, nvar, lmbp, nloss, genename, genereg, length)
        yield entry
        ##gc += 1; ec += len(gex)
    ##print(f"# genes on chromosome: {gc}, exons: {ec}")




def get_argument_parser():
    p = ArgumentParser(description="Create a gene table to inspect variants")
    p.add_argument("--genomewide", "-g", metavar="GENOMEWIDE_ZARR", required=True,
        help="input: zarr file from genomewide_analysis.py")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--regions", "--intervals", "-r", "-i", 
        nargs="+",
        help="genomic region, like '1:12_345_678-17_987_432', or 'all'")
    g.add_argument("--regionfile", "-f",
        help="file with genomic regions and optional gene names")
    p.add_argument("--filter", "-F",
        help="for '--regions all' during analysis, apply an output filter of exon regions")
    p.add_argument("-k", type=int, required=True,
        help="k-mer size (should be stored in ZARR, TODO!)")
    p.add_argument("--sortby", "-s",
        choices=("variants", "losses"), default="losses",
        help="how to sort the results")
    p.add_argument("--swap-pigs", action="store_true",
        help="swap roles of Ossabaw and Minipigs (i.e. find genes for Minipigs)")
    return p


def main(args):
    #rprint(f'# Hello, this is [bold green]inspect_variants.py.')
    zarrfile = args.genomewide
    k = args.k
    regs = args.regions

    filtered = False
    if args.filter:
        if not (regs==['all']):
            raise ValueError("ERROR: '--filter' only allowed with '-r all'")
        _, xgenes, exons = load_regions(args.filter, read_exons=True)
        filtered = True
    if regs is not None:
        if len(regs) == 1 and regs[0] == 'all':
            regs = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 X Y MT".split()
        regions = [parse_region(regionstr) for regionstr in regs]
        genes = ['?'] * len(regions)
    else:  # regions are in a file
        regions, genes, _ = load_regions(args.regionfile)
    #rprint(f'# Using {len(regions)} regions')
    assert len(regions) == len(genes)

    # process all regions
    C_OSS = 'coverage_ossabaw' if not args.swap_pigs else 'coverage_minipig'
    C_MIN = 'coverage_minipig' if not args.swap_pigs else 'coverage_ossabaw'
    ch_loaded = cz = seq = None
    coverage_minipig = coverage_ossabaw = paircount = None
    L = []  # list of regions
    byloss = args.sortby == "losses"
    for tr, (region, gene) in enumerate(zip(regions, genes)):
        # process region, a tuple (chr:str, start:int, end:int)
        ch, start, end = region
        if ch != ch_loaded:
            del cz, seq, coverage_ossabaw, coverage_minipig, paircount
            ##rprint(f'# Loading data for chromosome {ch}...')
            ##stdout.flush()
            cz = load_from_zgroup(zarrfile, f'/{ch}')
            seq = cz['seq'][:]
            coverage_ossabaw = cz[C_OSS][:]
            coverage_minipig = cz[C_MIN][:]
            paircount = cz['paircount'][:]
            ch_loaded = ch
        if end == 0:
            regions[tr] = region = (ch, start, len(seq))
        if not args.filter:
            vmbp, nv, lmbp, nloss = show_variants(region, gene, k, seq, coverage_ossabaw, coverage_minipig, paircount)
            length = region[2] - region[1] + 1
            entry = (lmbp, nloss, vmbp, nv, gene, region, length) if byloss \
                else (vmbp, nv, lmbp, nloss, gene, region, length)
            L.append(entry)
        else:
            # we have '-r all' AND a filter!
            # Extend L by (genes, exons) matching ch, where:
            # genes is a list of gene names, exons is a list of lists of exon regions
            ###print(xgenes[:20], exons[:20], sep='\n')
            L.extend(entry for entry in accumulate_exons(
                region, xgenes, exons, byloss,
                k, seq, coverage_ossabaw, coverage_minipig, paircount))
    L.sort(reverse=True)

    # output
    # TODO: include length in entry (different for exons!)
    if byloss:
        for rank, entry in enumerate(L):
            (lmbp, nloss, vmbp, nv, gene, region, length) = entry
            ch, start, end = region
            print(f"{rank+1}\t{lmbp:6d} lost/Mbp\t{vmbp:5d} var/Mbp\t{gene:15}\t{ch}:{start}-{end}\tlength={length}\t{nloss=}\t{nv=}")
    else:
        for rank, entry in enumerate(L):
            (vmbp, nv, lmbp, nloss, gene, region, length) = entry
            ch, start, end = region
            print(f"{rank+1}\t{vmbp:5d} var/Mbp\t{lmbp:6d} lost/Mbp\t{gene:15}\t{ch}:{start}-{end}\tlength={length}\t{nloss=}\t{nv=}")


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
