import sys
import io

import numpy as np
from numba import njit, uint8, int32

from .seqio import FormatError, _universal_reads


# Python FASTQ/gz handling ######################################

def fastq_reads(files, sequences_only=False, dirty=False):
    """
    For the given 'files':
    - list or tuple of FASTQ paths,
    - single FASTQ path (string; "-" for stdin)
    - open binary FASTQ file-like object f,
    yield a triple of bytes (header, sequence, qualities) for each read.
    If sequences_only=True, yield only the sequence of each read.

    This function operatates at the bytes (not string) level.
    The header contains the initial b'@' character.

    Automatic gzip decompression is provided
    if a file is a string and ends with .gz or .gzip.
    """
    func = _fastq_reads_from_filelike
    if sequences_only:
        func = _fastq_seqs_from_filelike
        if dirty:
            func = _fastq_seqs_dirty_from_filelike
    if type(files) == list or type(files) == tuple:
        # multiple files
        for f in files:
            yield from _universal_reads(f, func)
    else:
        # single file
        yield from _universal_reads(files, func)


def _fastq_reads_from_filelike(f, HEADER=b'@'[0], PLUS=b'+'[0]):
    strip = bytes.strip
    entry = 0
    try:
        while f:
            header = strip(next(f))
            if not header: continue
            entry += 1
            seq = strip(next(f))
            plus = strip(next(f))
            qual = strip(next(f))
            if header[0] != HEADER:
                raise FormatError(f"ERROR: Illegal FASTQ header: '{header.decode()}', entry {entry}")
            if plus[0] != PLUS:
                raise FormatError(f"ERROR: Illegal FASTQ plus line: '{plus.decode()}',\nheader '{header.decode()}',\nsequence '{seq.decode()}',\nentry {entry}")
            if len(plus) > 1 and plus[1:] != header[1:]:
                raise FormatError(f"ERROR: FASTQ Header/plus mismatch: '{header.decode()}' vs. '{plus.decode()}', entry {entry}")
            yield (header[1:], seq, qual)
    except StopIteration:
        pass

def _fastq_seqs_from_filelike(f, HEADER=b'@'[0], PLUS=b'+'[0]):
    strip = bytes.strip
    try:
        while f:
            header = strip(next(f))
            if not header: continue
            seq = strip(next(f))
            plus = next(f)
            next(f)  # ignore quality value
            if header[0] != HEADER:
                raise FormatError(f"ERROR: Illegal FASTQ header: '{header.decode()}'")
            if plus[0] != PLUS:
                raise FormatError(f"ERROR: Illegal FASTQ plus line: {plus.decode()}'")
            yield seq
    except StopIteration:
        pass


def _fastq_seqs_dirty_from_filelike(f):
    strip = bytes.strip
    while f:
        next(f)
        seq = strip(next(f))
        next(f)
        next(f)  # ignore quality value
        yield seq


# numba FASTQ handling ######################################

@njit(nogil=True)
def _find_fastq_seqmarks(buf, linemarks):
    """
    Find start and end positions of lines in byte buffer 'buf'.
    Store the information in 'linemarks', such that
    linemarks[i, 0:4] has information about record number i.

    Return pair (m, nxt), where:
        m: number of sequences processed
        nxt: position at which to continue processing 
             in the next iteration (start of new entry, '@')
    linemarks[i,0] = start of sequence line
    linemarks[i,1] = end of sequence line
    linemarks[i,2] = start of record (all lines)
    linemarks[i,3] = end of record (all lines)
    """
    n = buf.size
    if n == 0: return (0, 0)
    M = linemarks.shape[0]
    i = 0
    m = -1  # number of current record we're in
    nxt = 0  # byte of beginning of last record
    line = 0  # current line in FASTQ record (0,1,2,3)
    # find start of current line
    while True:
        if buf[i] == 10:
            i += 1
            if line == 0:
                linemarks[m, 3] = i
            if i >= n:
                # we did not find valid record start
                if line == 0:
                    m += 1
                    nxt = n
                return (m, nxt)
        if line == 0:
            m += 1
            nxt = i
            linemarks[m, 2] = i
            if m >= M: return M, nxt
        elif line == 1:
            linemarks[m, 0] = i
        # find end of current line
        while buf[i] != 10:
            i += 1
            if i >= n:
                # we did not find the end of the line before the buffer was exhausted
                # we cannot set linemarks[m][1]
                return m, nxt
        if line == 1:
            linemarks[m, 1] = i
        line = (line + 1) % 4


def fastq_chunks(files, bufsize=2**23, maxreads=(2**23)//200, subsample=1):
    """
    Yield all chunks from a list or tuple of FASTQ 'files'.
    A chunk is a pair (buffer, linemarks), where
    - buffer is readable/writable byte buffer
    - buffer[linemarks[i,0]:linemarks[i,1]] contains the i-th sequence 
      of the chunk; other line marks are not stored.
    - The part of linemarks that is returned is such that
      linemarks.shape[0] is at most maxreads and equals the number of reads.
    CAUTION: If bufsize is very small, such that not even a single FASTQ entry
      fits into the buffer, it will appear that the buffer is empty.
    """
    # defaults are good for single-threaded runs; multiply by #threads.
    if not (isinstance(files, tuple) or isinstance(files, list)):
        files = (files,)  # single file?
    if bufsize/2**30 > 1.5:
        raise ValueError(f"ERROR: bufsize={bufsize/2**30} GiB must be <= 1.5 GiB (2**30 bytes); use smaller chunksize of fewer threads.")
    linemarks = np.empty((maxreads, 4), dtype=np.int32)
    buf = np.empty(bufsize, dtype=np.uint8)
    for filename in files:
        with io.BufferedReader(io.FileIO(filename, 'r'), buffer_size=bufsize) as f:
            prev = 0
            while True:
                read = f.readinto(buf[prev:])
                if read == 0: break
                available = prev + read  # number of bytes available
                m, cont = _find_fastq_seqmarks(buf[:available], linemarks)
                if m <= 0: 
                    # TODO: check for FASTQ (buffer starts with @)
                    raise RuntimeError(f"no complete records for bufsize {bufsize}")
                    break
                chunk = (buf, linemarks[0:m:subsample,:])
                yield chunk
                #cont = linemarks[m-1,1] + 1
                prev = available - cont
                if prev > 0:
                    buf[:prev] = buf[cont:available]
                    #print(f"  moving buf[{cont}:{available}] to buf[0:{prev}]")
                assert prev < cont


def fastq_chunks_paired(pair, bufsize=2**23, maxreads=2**23//200):
    """
    Yield all chunks from a list or tuple of FASTQ files.
    A chunk is a pair (buffer, linemarks), where
    - buffer is readable/writable byte buffer
    - buffer[linemarks[i,0]:linemarks[i,1]] contains the i-th sequence 
      of the chunk; other line marks are not stored.
    - The part of linemarks that is returned is such that
      linemarks.shape[0] is at most maxreads and equals the number of reads.
    CAUTION: If bufsize is very small, such that not even a single FASTQ entry
      fits into the buffer, it will appear that  the buffer is empty.
    """
    # defaults are good for single-threaded runs; multiply by #threads.
    (files1, files2) = pair
    if not (isinstance(files1, tuple) or isinstance(files1, list)):
        files1 = (files1,)
    if not (isinstance(files2, tuple) or isinstance(files2, list)):
        files2 = (files2,)
    linemarks1 = np.empty((maxreads, 4), dtype=np.int32)
    linemarks2 = np.empty((maxreads, 4), dtype=np.int32)
    buf1 = np.empty(bufsize, dtype=np.uint8)
    buf2 = np.empty(bufsize, dtype=np.uint8)
    if len(files1) != len(files2):
        raise RuntimeError(f"different numbers of fastq files")
    for i in range(len(files1)):
        with io.BufferedReader(io.FileIO(files1[i], 'r'), buffer_size=bufsize) as f1, \
             io.BufferedReader(io.FileIO(files2[i], 'r'), buffer_size=bufsize) as f2:
            prev1 = 0
            prev2 = 0
            while True:
                read1 = f1.readinto(buf1[prev1:])
                read2 = f2.readinto(buf2[prev2:])
                if read1 == 0 and read2 == 0: break
                available1 = prev1 + read1
                available2 = prev2 + read2
                m1, cont1 = _find_fastq_seqmarks(buf1[:available1], linemarks1)
                m2, cont2 = _find_fastq_seqmarks(buf2[:available2], linemarks2)
                if m1 <= 0 or m2 <= 0: 
                    # TODO: check for FASTQ (buffer starts with @)
                    raise RuntimeError(f"no complete records for bufsize {bufsize}")
                    break
                if m1 == m2:
                    chunk = (buf1, linemarks1[:m1], buf2, linemarks2[:m2])
                elif m1 < m2:
                    chunk = (buf1, linemarks1[:m1], buf2, linemarks2[:m1])
                    cont2 = linemarks2[m1][2] 
                else:
                    chunk = (buf1, linemarks1[:m2], buf2, linemarks2[:m2])  
                    cont1 = linemarks1[m2][2] 
                yield chunk
                prev1 = available1 - cont1
                prev2 = available2 - cont2
                if prev1 > 0:
                    buf1[:prev1] = buf1[cont1:available1]
                    buf1[prev1:] = 0
                if prev2 > 0:
                    buf2[:prev2] = buf2[cont2:available2]
                    buf2[prev2:] = 0
                #assert prev1 < cont1
                #assert prev2 < cont2


# FASTQ checking ####################################################

def fastqcheck(args):
    files = args.sequences
    if args.paired:
        success = fastqcheck_paired(list(zip(*[iter(files)]*2)), args)
    else:
        success = fastqcheck_single(files, args)
    return success


def fastqcheck_paired(filepairs, args):
    success = True
    for (f1, f2) in filepairs:
        print(f"Checking {f1}, {f2}...")
        msg = "OK"
        try:
            for entry1, entry2 in zip(fastq_reads(f1), fastq_reads(f2)):
                c1 = entry1[0].split()[0]
                c2 = entry2[0].split()[0]
                if c1 != c2:
                    raise FormatError(f"headers {c1.decode()} and {c2.decode()} do not match")
        except FormatError as err:
            success = False
            msg = "FAILED: " + str(err)
        print(f"{f1}, {f2}: {msg}")
    return success


def fastqcheck_single(files, args):
    success = True
    for f in files:
        print(f"Checking {f}...")
        msg = "OK"
        try:
            for entry in fastq_reads(f):
                pass
        except FormatError as err:
            success = False
            msg = "FAILED: " + str(err)
        print(f"{f}: {msg}")
    return success



# FASTQ name guessing and argument parsing   ##########################

def guess_pairs(filenames, replacer):
    """guess_pairs(filenames, replacer):
    Return a list of file names where the read number (1/2) 
    has been replaced by the 'replacer'.
    Example: guess_pairs(['abc_R1.fq', 'xyz_R1.fq'], 2)
        returns ['abc_R2.fq', 'xyz_R2.fq'].
    """
    if replacer != 1 and replace != 2:
        raise ValueError("Eguess_pairs: Parameter 'replacer' must be 1 or 2")
    # replace=1: Replace rightmost occurrence of '_R2' by '_R1'
    # replace=2: Replace rightmost occurrence of '_R1' by '_R2'
    orig = str(3 - replacer)
    o = "_R" + orig
    r = "_R" + str(replacer)
    pairnames = list()
    for name in filenames:
        start = name.rfind(o)
        if start < 0:
            raise ValueError(f"guess_pairs: FASTQ file name '{name}' does not contain '{o}'")
        pairnames.append(name[:start] + r + name[start+len(o):])
    return pairnames


def parse_fastq_args(args):
    """P
    arse args.{first, second, single, guess_pairs}
    Return (paired, fastq), where
    - paired is a bool that is True iff args.first or args.second is provided
    - fastq is a 1-tuple or a 2-tuple of lists of filenames.
      It is a 2-tuple iff paired == True.
    """
    argerror = ArgumentParser.error
    paired = False
    if args.first or args.second:
        paired = True
        if args.guess_pairs and args.first:
            if args.second:
                argerror("ERROR: With given --first, cannot specify --second together with --guess-pairs")
            args.second = guess_pairs(args.first, 2)
        elif args.guess_pairs and args.second:
            args.first = guess_pairs(args.second, 1)
        if len(args.first) != len(args.second):
            argerror(f"ERROR: --first and --second must specify the same number of files ({len(args.first)} vs {len(args.second)})")
        if args.first is None or args.second is None:
            argerror(f"ERROR: not enough information given for paired reads")
    if args.single:
        if paired:
            argerror(f"ERROR: cannot use --single together with --first/--second")
    fastq = (args.single,) if single else (args.first, args.second)
    return paired, fastq


