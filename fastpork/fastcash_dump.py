"""
dump.py
Dump a hash table of k-mer codes with choices and values.
"""

import sys
import datetime
from math import ceil, log2
from os.path import splitext
import io

import numpy as np
from numba import njit, uint64, int64, int8

from .lowlevel.intbitarray import intbitarray
from .h5utils import save_to_h5group
from .hashio import load_hash
from .mathutils import bitsfor
from .dnaencode import qcode_to_dnastr



def parse_selected_values(args_values):
    if args_values is None:
        return None
    # Otherwise, args_values is a list of strings representing ints or ranges.
    # Currently, we can only deal with a single int value...
    if len(args_values) != 1:
        raise ValueError("ERROR: Currently, only selection of a single value is implemented; "
            "selecting {args_values} is not possible. Sorry.")
    return int(args_values[0])
    # TODO: parse list of values / ranges, such as values=['1', '2', '5-7']
    # into a bitarray, but make sure the largest selected value is < 512 for this.

"""
Input:
    h: hashtable,
    codes: np.array
    choices: np.array
    values: np.array
    myvalue: int; only dump keys with this value; default None
"""
def make_dump_function_single(h, codes, choices, values, myvalue=None):
    """compile the dump function"""
    assert h.subtables == 0
    npages = h.npages
    pagesize = h.pagesize
    is_slot_empty_at = h.private.is_slot_empty_at
    get_signature_at = h.private.get_signature_at
    get_key_choice_from_signature = h.private.get_subkey_choice_from_page_signature
    get_value_at = h.private.get_value_at
    setcode = codes.set
    setchoice = choices.set
    setvalue = values.set

    if myvalue is None:
        @njit(nogil=True,
            locals=dict(page=int64, slot=int64, i=int64))
        def dump(ht, acodes, achoices, avalues):
            """
            Fill given arrays acodes[:], choices[:], avalues[:] with
            k-mer codes, choices and values in the hash table.
            Return number of kmers inserted.
            """
            i = 0
            for page in range(npages):
                for slot in range(pagesize):
                    if is_slot_empty_at(ht, page, slot): continue
                    sig = get_signature_at(ht, page, slot)
                    (key, c) = get_key_choice_from_signature(page, sig)  # 1, 2, 3
                    value = get_value_at(ht, page, slot)
                    setcode(acodes, i, key)
                    setchoice(achoices, i, c)  # 1, 2, 3
                    setvalue(avalues, i, value)
                    i += 1
            return i
        return dump

    # myvalue is a given int
    @njit(nogil=True,
        locals=dict(page=int64, slot=int64, i=int64))
    def dump(ht, acodes, achoices, avalues):
        """
        Fill given arrays acodes[:], choices[:], avalues[:] with
        k-mer codes, choices and values in the hash table.
        Return number of kmers inserted.
        """
        i = 0
        for page in range(npages):
            for slot in range(pagesize):
                if is_slot_empty_at(ht, page, slot): continue
                value = get_value_at(ht, page, slot)
                if value != myvalue: continue
                setvalue(avalues, i, value)
                sig = get_signature_at(ht, page, slot)
                (key, c) = get_key_choice_from_signature(page, sig)  # 1, 2, 3
                setcode(acodes, i, key)
                setchoice(achoices, i, c)  # 1, 2, 3
                i += 1
        return i
    return dump

def make_dump_function(h, codes, subtables, choices, values, myvalue=None):
    """compile the dump function"""
    nsubtables = h.subtables
    assert nsubtables > 0
    npages = h.npages
    pagesize = h.pagesize
    is_slot_empty_at = h.private.is_slot_empty_at
    get_signature_at = h.private.get_signature_at
    get_subkey_choice_from_page_signature = h.private.get_subkey_choice_from_page_signature
    get_key_from_sub_subkey = h.private.get_key_from_subtable_subkey
    get_value_at = h.private.get_value_at
    setsubtable = subtables.set
    setcode = codes.set
    setchoice = choices.set
    setvalue = values.set

    if myvalue is None:
        @njit(nogil=True,
            locals=dict(subtable=int64, page=int64, slot=int64, i=int64))
        def dump(ht, acodes, asubtables, achoices, avalues):
            """
            Fill given arrays acodes[:], asubtables[:], choices[:], avalues[:] with
            k-mer codes, subtables, choices and values in the hash table.
            Return number of kmers inserted.
            """
            i = 0
            for subtable in range(nsubtables):
                for page in range(npages):
                    for slot in range(pagesize):
                        if is_slot_empty_at(ht, subtable, page, slot): continue
                        sig = get_signature_at(ht, subtable, page, slot)
                        (subkey, c) = get_subkey_choice_from_page_signature(page, sig)
                        key = get_key_from_sub_subkey(subtable, subkey)
                        value = get_value_at(ht, subtable, page, slot)
                        setsubtable(asubtables, i , subtable)
                        setcode(acodes, i, key)
                        setchoice(achoices, i, c)  # 1, 2, 3
                        setvalue(avalues, i, value)
                        i += 1
            return i
        return dump

    # myvalue is a given int
    @njit(nogil=True,
        locals=dict(page=int64, slot=int64, i=int64))
    def dump(ht, acodes, achoices, avalues):
        """
        Fill given arrays acodes[:], asubtables[:], choices[:], avalues[:] with
        k-mer codes, choisubtables, ces and values in the hash table.
        Return number of kmers inserted.
        """
        i = 0
        for subtable in range(nsubtables):
            for page in range(npages):
                for slot in range(pagesize):
                    if is_slot_empty_at(ht, subtable, page, slot): continue
                    value = get_value_at(ht, subtable, page, slot)
                    if value != myvalue: continue
                    sig = get_signature_at(ht, subtable, page, slot)
                    (subkey, c) = get_subkey_choice_from_page_signature(page, sig)
                    key = get_key_from_sub_subkey(subtable, subkey)
                    setsubtable(asubtables, i, subtable)
                    setcode(acodes, i, key)
                    setchoice(achoices, i, c)  # 1, 2, 3
                    setvalue(avalues, i, value)
                    i += 1
        return i
    return dump


# main #########################################

def main(args):
    """main method for dmuping k-mer codes, choicesand values"""
    # needs: args.index (str), args.dump (str)
    # optional: args.value (int)

    starttime = datetime.datetime.now()
    if args.dump is None and args.text is None:
        base, ext = splitext(args.index)
        args.dump = base + '.dump.h5'

    if args.dump:
        print(f"# {starttime:%Y-%m-%d %H:%M:%S}: Dumping index '{args.index}' -> '{args.dump}'")
    if args.text:
        print(f"# {starttime:%Y-%m-%d %H:%M:%S}: Dumping index '{args.index}' -> '{args.text}'")
    myvalues = parse_selected_values(args.values)  # currently: int, later: bitarray

    # Load index
    h, valueinfo, info = load_hash(args.index)
    print(f"# Hash loaded. Size is {(h.mem_bytes/2**30):.3f} GB (2**30 bytes).")

    k = int(info['k'])
    rcmode = info.get('rcmode', valueinfo.RCMODE)
    if isinstance(rcmode, bytes): rcmode = rcmode.decode("ASCII")
    assert type(rcmode) == str
    valuebits = int(ceil(log2(valueinfo.NVALUES)))
    info['valuebits'] = valuebits
    choicebits = bitsfor(h.choices)
    info['choicebits'] = choicebits
    subtablebits = max(1,bitsfor(h.subtables))
    info['subtablebits'] = subtablebits
    info['subtables'] = h.subtables


    ht = h.hashtable
    if args.fast or h.subtables == 0:
        print(f"# Fast mode: skipping counting, over-allocating.")
        tablesize = h.pagesize * h.npages
        kmers = tablesize
    else:
        tablesize = h.pagesize * h.npages * h.subtables
        print(f"# Counting slots and allocating arrays...")
        if myvalues is None:
            kmers = h.slots_nonempty(ht)
        else:
            kmers = h.slots_with_value(ht, myvalue)

    codes = intbitarray(kmers, 2*k)
    print(f"# Codes array: Size is {(codes.capacity_bytes/2**30):.3f} GB (2**30 bytes).")
    subtables = intbitarray(kmers, subtablebits)
    print(f"# Subtables array: Size is {(subtables.capacity_bytes/2**30):.3f} GB (2**30 bytes).")
    choices = intbitarray(kmers, choicebits)
    print(f"# Choices array: Size is {(choices.capacity_bytes/2**30):.3f} GB (2**30 bytes).")
    values = intbitarray(kmers, valuebits)
    print(f"# Values array: Size is {(values.capacity_bytes/2**30):.3f} GB (2**30 bytes).")
    print()
    print(f"# Collecting dump data from hash table...")
    if h.subtables == 0:
        dump = make_dump_function_single(h, codes, choices, values, myvalue=myvalues)
        dkmers = dump(ht, codes.array, choices.array, values.array)
    else:
        dump = make_dump_function(h, codes, subtables, choices, values, myvalue=myvalues)
        dkmers = dump(ht, codes.array, subtables.array, choices.array, values.array)
    info['kmers'] = dkmers
    r = dkmers/tablesize if kmers > 0 else 1.0
    info['fillrate'] = r
    print(f"# Selected k-mers: {dkmers} of {kmers} allocated:  ratio {r:.3f}.")

    # save to HDF5
    if args.dump:
        now = datetime.datetime.now()
        outfile = args.dump
        print(f"# {now:%Y-%m-%d %H:%M:%S}: writing dump h5-file '{outfile}'")
        save_to_h5group(outfile, "info", info=info)
        save_to_h5group(outfile, "data", kmercodes=codes.array, choices=choices.array, values=values.array, subtables=subtables.array)
    if args.text:
        now = datetime.datetime.now()
        outfile = args.text
        print(f"# {now:%Y-%m-%d %H:%M:%S}: writing dump csv-file '{outfile}'")
        with io.BufferedWriter(io.FileIO(outfile, 'w')) as out:
            out.write(b"code,subtable,choice,value\n")
            for i in range(dkmers):
                code = qcode_to_dnastr(codes.get(codes.array, i),k)
                st = subtables.get(subtables.array, i)
                c = choices.get(choices.array, i)
                v = values.get(values.array, i)
                line = f"{code},{st},{c},{v}\n"
                out.write(line.encode())
            out.flush()

    endtime = datetime.datetime.now()
    elapsed = (endtime - starttime).total_seconds()
    print(f"# time sec: {elapsed:.1f}")
    print(f"# {endtime:%Y-%m-%d %H:%M:%S}: Done.")
