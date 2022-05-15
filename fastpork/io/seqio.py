import sys
import io
import gzip
from collections import Counter
from subprocess import check_output
from os import path as ospath

import numpy as np
from numba import njit, uint8, int32


class FormatError(RuntimeError):
    pass


def _universal_reads(f, func):
    """
    yield each read from file f, where
    f is a filename (string), possibly ending with .gz/.gzip,
      or a file-like object,
    func describes the logic of obtaining reads from the file,
    and can be one of
      _fastq_reads_from_filelike: yields triples (header, sequence, qualities)
      _fastq_seqs_from_filelike: yields sequences only
      _fasta_reads_from_filelike: yields pairs (header, sequence)
      _fasta_seqs_from_filelike: yields sequences only
      _text_reads_from_filelike: yields each line as a read
    All objects are bytes/bytearray objects (can be decoded using ASCII encoding).
    Headers are yielded WITHOUT the initial character (> for FASTA, @ for FASTQ).
    """
    if not isinstance(f, str) and not isinstance(f, bytes):
        yield from func(f)
    elif f == "-" or f == b"-":
        yield from func(sys.stdin.buffer)
    elif f.endswith((".gz", ".gzip")):
        with gzip.open(f, "rb") as file:
            reader = io.BufferedReader(file, 4*1024*1024)
            yield from func(reader)
    else:
        with open(f, "rb", buffering=-1) as file:
            yield from func(file)

####################################################################

# Grouping
#
# A group_by_* function is a function that returns a group name, given
# - a filename (full absolute path or relative path)
# - a sequence name (header string)


def group_by_all(path, header):
    return "all"


def group_by_basename(path, header):
    fname = ospath.basename(ospath.abspath(path))
    while True:
        fname, ext = ospath.splitext(fname)
        if not ext:
            break
    return fname


def group_by_seqname(path, header):
    fields = header[1:].split()
    if not fields:
        return "default"
    return fields[0]


def group_by_seqname_strict(path, header):
    return header[1:].split()[0]  # raises IndexError if header is empty


def group_by_dict_factory(d, default="default"):
    """
    return a groupby function that looks up group in given dict,
    using the first word of the sequence header
    """
    def groupby_d(path, header):
        fields = header[1:].split()
        name = fields[0] if fields else ''
        if default:
            return d.get(name, default)
        return d[name]
    return groupby_d


def get_grouper(groupspec):
    """
    groupspec is singleton list or pair list:
    (method[, specifier]) with the following possibilities:
    ('const', constant): a constant group name for all sequences
    """
    method = groupspec[0]
    spec = groupspec[1] if len(groupspec) > 1 else None
    if method == 'const':
        if spec is None:
            raise ValueError('groupby "const" needs an argument (the constant)')
        return lambda path, header: spec
    if method == 'all':
        return None
    raise NotImplementedError('this groupby functionality is not yet implemented: {groupspec}')


def get_group_sizes(files, groupmap, offset=0, override=None):
    lengths = Counter()
    if files is None:
        return lengths
    if override is not None:
        if groupmap is None:
            lengths["all"] = override
            return lengths
        for (_, _, group) in grouped_sequences(files, groupmap):
            lengths[group] = override
        return lengths
    # count total lengths of sequences
    for (_, seq, group) in grouped_sequences(files, groupmap):
        lengths[group] += len(seq) + offset
    return lengths


def grouped_sequences(files, groupby=None, format=None):
    """
    For each sequence in the given list/tuple of files or (single) file path,
    yield a triple (header, sequence, group),
    according to the given groupby function.

    The file format (.fast[aq][.gz]) is recognized automatically,
    but can be explicitly given by format="fasta" or format="fastq".
    """
    if type(files) == list or type(files) == tuple:
        for f in files:
            yield from _grouped_sequences_from_a_file(f, groupby, format=format)
    else:
        yield from _grouped_sequences_from_a_file(files, groupby, format=format)


def _grouped_sequences_from_a_file(fname, groupby=None, format=None):
    few = fname.lower().endswith
    if format is not None:
        format = format.lower()
    if format == "fasta" or few((".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")):
        if groupby is not None:
            reads = _fasta_reads_from_filelike
            for (h, s) in _universal_reads(fname, reads):
                g = groupby(fname, h)
                yield (h, s, g)
        else:
            reads = _fasta_seqs_from_filelike
            for s in _universal_reads(fname, reads):
                yield (True, s, "all")
    elif format == "fastq" or few((".fq", ".fastq", ".fq.gz", ".fastq.gz")):
        if groupby is not None:
            reads = _fastq_reads_from_filelike
            for (h, s, q) in _universal_reads(fname, reads):
                g = groupby(fname, h)
                yield (h, s, g)
        else:
            reads = _fastq_seqs_from_filelike
            for s in _universal_reads(fname, reads):
                yield (True, s, "all")
    else:
        raise FormatError("format of file '{fname}' not recognized")


####################################################################

def get_sizebounds(files):
    """
    return a pair (sumbound, maxbound), where
    sumbound is an upper bound on the sum of the number of q-grams in the given 'files',
    maxbound is an upper bound on the maximum of the number of q-grams in one entry in 'files'.
    """
    if files is None:
        return (0, 0)
    sb = mb = 0
    for (_, seq, _) in grouped_sequences(files):
        ls = len(seq)
        sb += ls
        if ls > mb:  mb = ls
    return (sb, mb)


def number_of_sequences_in(fname):
    # TODO: this only works with Linux / Mac
    few = fname.lower().endswith  # few = "filename ends with"
    if few((".fa", ".fasta")):
        x = check_output(["grep", "-c", "'^>'", fname])
        return int(x)
    if few((".fa.gz", ".fasta.gz")):
        x = check_output(["gzcat", fname, "|", "grep", "-c", "'^>'"])
        return int(x)
    if few((".fq", ".fastq")):
        x = check_output(["wc", "-l", fname])
        n = int(x.strip().split()[0])
        return n//4
    if few((".fq.gz", ".fastq.gz")):
        x = check_output(["gzcat", fname, "|", "wc", "-l"])
        n = int(x.strip().split()[0])
        return n//4

