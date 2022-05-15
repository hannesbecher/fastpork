

from .seqio import FormatError, _universal_reads
from ..dnaencode import dna_to_2bits


# FASTA/gz handling ######################################

# Generator that yields all sequences and values from FASTA files

def all_fasta_seqs(fastas, value_from_name, both, skipvalue, *, progress=False):
    """
    Yield a (sq, v1, v2) triple for each sequence in given fastas, where:
    - sq is the two-bit-encoded sequence,
    - v1 is the first value derived from the header using value_from_name,
    - v2 is the second value derived from the header using value_from_name,
      or identical to v1 if both==False.
    Sequences whose v1 evaluates to skipvalue are skipped.
    Progress is printed to stdout if progress=True.
    """
    for fasta in fastas:
        print(f"# Processing '{fasta}':")
        for header, seq in fasta_reads(fasta):
            name = header.split()[0]
            v1 = value_from_name(name, 1)
            if v1 == skipvalue:
                if progress:
                    print(f"#   sequence '{name.decode()}': length {len(seq)}, skipping")
                continue
            v2 = value_from_name(name, 2) if both else v1
            if progress:
                print(f"#   sequence '{name.decode()}': length {len(seq)}, values {v1}, {v2}")
            sq = dna_to_2bits(seq)
            yield (name, sq, v1, v2)


def fasta_reads(files, sequences_only=False):
    """
    For the given
    - list or tuple of FASTA paths,
    - single FASTA path (string),
    - open binary FASTA file-like object f,
    yield a pair of bytes (header, sequence) for each entry (of each file).
    If sequences_only=True, yield only the sequence of each entry.
    This function operatates at the bytes (not string) level.
    The header DOES NOT contain the initial b'>' character.
    If f == "-", the stdin buffer is used.
    Automatic gzip decompression is provided,
    if f is a string and ends with .gz or .gzip.
    """
    func = _fasta_reads_from_filelike if not sequences_only else _fasta_seqs_from_filelike
    if type(files) == list or type(files) == tuple:
        # multiple files
        for f in files:
            yield from _universal_reads(f, func)
    else:
        # single file
        yield from _universal_reads(files, func)


def _fasta_reads_from_filelike(f, COMMENT=b';'[0], HEADER=b'>'[0]):
    strip = bytes.strip
    header = seq = None
    for line in f:
        line = strip(line)
        if len(line) == 0:
            continue
        if line[0] == COMMENT:
            continue
        if line[0] == HEADER:
            if header is not None:
                yield (header, seq)
            header = line[1:]
            seq = bytearray()
            continue
        seq.extend(line)
    if header is not None:
        yield (header, seq)


def _fasta_seqs_from_filelike(f, COMMENT=b';'[0], HEADER=b'>'[0]):
    strip = bytes.strip
    header = seq = False
    for line in f:
        line = strip(line)
        if len(line) == 0:
            continue
        if line[0] == COMMENT:
            continue
        if line[0] == HEADER:
            if header:
                yield seq
            header = True
            seq = bytearray()
            continue
        seq.extend(line)
    yield seq


# FASTA header extraction ###########################################

_SEPARATORS = {'TAB': '\t', 'SPACE': ' '}

def fastaextract(args):
    """extract information from FASTA headers and write in tabular form to stdout"""
    files = args.files
    items = args.items
    seps = args.separators
    sfx = [args.suffix] if args.suffix else []
    seps = [_SEPARATORS.get(sep.upper(), sep) for sep in seps]
    if items is None: 
        items = list()
        seps = list()
    if len(seps) == 1:
        seps = seps * len(items)
    seps = [""] + seps
    head = ['transcript_id'] + items

    first = [x for t in zip(seps, head) for x in t] + sfx
    print("".join(first))
    for f in files:
        for (header, _) in fasta_reads(f):
            infolist = get_header_info(header, items, ":", seps) + sfx
            print("".join(infolist))


def get_header_info(header, items, assigner, seps):
    fields = header.decode("ascii").split()
    assigners = [i for (i,field) in enumerate(fields) if assigner in field]
    if 0 in assigners: 
        assigners.remove(0)
    D = dict()
    if items is None: items = list()
    for j, i in enumerate(assigners):
        field = fields[i]
        pos = field.find(assigner)
        assert pos >= 0
        name = field[:pos]
        nexti = assigners[j+1] if j+1 < len(assigners) else len(fields)
        suffix = "_".join(fields[i+1:nexti])
        if len(suffix)==0:
            D[name] = field[pos+1:]
        else:
            D[name] = field[pos+1:] + '_' + suffix
    # dictionary D now has values for all fields
    L = [seps[0], fields[0]]
    for i, item in enumerate(items):
        if item in D:
            L.append(seps[i+1])
            L.append(D[item])
    return L
