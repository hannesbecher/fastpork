# fastcash.textio.py
# Text file input/output

from .seqio import FormatError, _universal_reads


def text_reads(f):
    """
    For the given text file path or open binary file-like object f,
    yield a each line as a bytes object, without the newline.
    If f == "-", the stdin buffer is used.
    Automatic gzip decompression is provided
    if f is a string and ends with .gz or .gzip.
    """
    yield from _universal_reads(f, _text_reads_from_filelike)


def _text_reads_from_filelike(f):
    strip = bytes.strip
    while f:
        yield strip(next(f))

