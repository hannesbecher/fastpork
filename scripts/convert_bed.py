"""
This annotation should be referred to as NCBI Sus scrofa Annotation Release 106.
https://www.ncbi.nlm.nih.gov/genome/annotation_euk/Sus_scrofa/106/

Sus scrofa page:
https://www.ncbi.nlm.nih.gov/genome/?term=Sus+scrofa

GTF file:
https://ftp.ncbi.nlm.nih.gov/genomes/all/annotation_releases/9823/106/GCF_000003025.6_Sscrofa11.1/

BED file:
gtf2bed < GCF_000003025.6_Sscrofa11.1_genomic.gtf > Sscrofa11.1.bed
"""

import sys
from collections import Counter

use_exons = True

def get_chrname(extra):
    fields = extra.split(";")
    for field in fields:
        try:
            name, value = field.split("=")
        except ValueError:
            print(f"ERROR in get_chrname, {field=}")
            raise
        if name == "Name":
            return value
    return '?'


def enhance_genename(name, extra):
    """if the gene is a miRNA, enhance its name by adding information"""
    infos = [name]
    fields = extra.split(";")
    for field in fields:
        try:
            name, value = field.split("=")
        except ValueError:
            print(f"ERROR in enhance_genename, {field=}")
            raise
        if name == "gene_synonym":
            infos.append(value)
            continue
        if name != "Dbxref":
            continue
        dbentries = value.split(",")
        for dbentry in dbentries:
            dname, dvalue = dbentry.split(":")
            if dname != "miRBase": continue
            infos.append(dvalue)
    return ",".join(infos).replace(" ", "_")

def get_genename(extra):
    """extract the gene name for an exon feature"""
    fields = extra.split(";")
    for field in fields:
        try:
            name, value = field.split("=")
        except ValueError:
            print(f"ERROR in get_chrname, {field=}")
            raise
        if name == "gene":
            return value
    return None


def merge(L, ch, start, stop):
    for i, ex in enumerate(L):
        ech, estart, estop = ex
        if ch != ech: continue
        if (start <= estart <= stop or estart <= start <= estop
            or start <= estop <= stop or estart <= stop <= estop):
                L[i] = (ch, min(start,estart), max(stop,estop))
                break
    else: # no break: not found
        L.append((ch, start, stop))


def aggregate_overlapping(exonlist):
    L = []
    for ch, start, stop in exonlist:
        merge(L, ch, start, stop)
    return L


########### MAIN SCRIPT ###############################


# Step 1: Read chromosomes ("regions")
chrdict = dict()
with open('resources/GCF_000003025.6_Sscrofa11.1_genomic.bed') as fbed:
    for row in fbed:
        fields = row.strip().split('\t')
        acc, start0, end0, what, dot, strand, refseq, region, dot2, extra = fields
        start0 = int(start0); end0 = int(end0)
        if region != 'region': continue

        assert acc not in chrdict
        name = get_chrname(extra)
        chrdict[acc] = name
        ##print(f"NEW CHROMOSOME: {acc} -> {name}")

# Step 2: Read genes
out = dict()
exons = dict()
seen = Counter()
with open('resources/GCF_000003025.6_Sscrofa11.1_genomic.bed') as fbed:
    for row in fbed:
        fields = row.strip().split('\t')
        acc, start0, end0, what, dot, strand, refseq, region, dot2, extra = fields
        start0 = int(start0); end0 = int(end0)
        if region != 'gene': continue
        #ch = chrdict.get(acc, "Unknown")
        ch = chrdict[acc]
        if ch == "Unknown": continue

        assert what.startswith('gene-')
        oname = what[5:]
        name = enhance_genename(oname, extra)
        if oname in seen:
            seen[oname] += 1
            name = f"{name}___{seen[name]}"
            assert False, f"repeated {name}"
        else:
            seen[oname] += 1
        out[oname] = f"{ch}:{start0+1}-{end0}  {name}"
        exons[oname] = []

if not use_exons:
    for _, output in out.items():
        print(output)
    sys.exit(0)


# Step 3a: Read exons, gene names are in dict 'exons'
del seen
with open('resources/GCF_000003025.6_Sscrofa11.1_genomic.bed') as fbed:
    for row in fbed:
        fields = row.strip().split('\t')
        acc, start0, end0, what, dot, strand, refseq, region, dot2, extra = fields
        start0 = int(start0); end0 = int(end0)
        if region != 'exon': continue
        ch = chrdict[acc]
        if ch == "Unknown": continue
        if not what.startswith('exon-'):
            if what.startswith('id-'): continue
            print(f"WHAT ERROR: {what=}")
        genename = get_genename(extra)
        if genename is None: continue
        if genename not in exons:
            # was defined but probably not a gene, but pseudogene
            continue
        og = out[genename]
        och = og[:og.find(":")]
        if och != ch: continue  # second gene version, or tRNA
        exons[genename].append((ch,start0,end0))


# Step 4: Reduce exons and print output
for oname, output in out.items():
    ex = exons[oname]
    reduced_exons = aggregate_overlapping(ex)
    formatted_exons = " ".join([ f'{ch}:{start0+1}-{end0}' for ch,start0,end0 in reduced_exons])
    print(f"{output}  {len(reduced_exons)}  {formatted_exons}")

