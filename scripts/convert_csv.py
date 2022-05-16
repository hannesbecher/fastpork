import csv
from collections import Counter


def merge(L, ch, start, stop):
    for i, ex in enumerate(L):
        ech, estart, estop = ex['ch'], ex['min'], ex['max']
        if ch != ech: continue
        if (start <= estart <= stop or estart <= start <= estop
            or start <= estop <= stop or estart <= stop <= estop):
                L[i] = dict(ch=ch, min=min(start,estart), max=max(stop,estop))
                break
    else: # no break: not found
        L.append(dict(ch=ch, min=start, max=stop))


skipped = ok = 0
result = dict()
C = Counter()
lastlocus = ''
with open('resources/proteins_84_317145.csv') as fcsv:
    reader = csv.reader(fcsv, dialect='unix')
    for row in reader:
        if row[0].startswith("#"): continue
        if not row[0].startswith("chromosome") and not row[0].startswith("mitochondrion"):
            skipped += 1
            if not row[0].startswith("Un"):
                print(row[0])
            continue
        ch, acc, start, stop, strand, geneid, locus, tag, product, length, pname = row
        if locus != lastlocus:
            C[locus] += 1
            lastlocus = locus
        start = int(start)
        stop = int(stop)
        assert start <= stop, f"{start=} > {stop=}"
        cs = ch.split()[1]
        if locus not in result:
            result[locus] = [dict(ch=cs, min=start, max=stop)]
        else:
            L = result[locus]
            merge(L, cs, start, stop)
        ok += 1
for k in result.keys():
    #if len(result[k]) == 1: continue
    for ex in result[k]:
        ch, start, stop = ex['ch'], ex['min'], ex['max']
        print(f"{ch}:{start}-{stop}  {k}")
