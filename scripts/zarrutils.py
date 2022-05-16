"""
zarrutils.py:
utilities for dealing with ZARR files
that store a k-mer index and additional information.
"""

from collections import defaultdict
import pickle
import numpy as np
import zarr


# saving to and loading from ZARR groups
# with auto-pickling (names with suffix '!p')

def save_to_zgroup(path, group, **kwargs):
    with zarr.open(path, mode='a') as f:
        g = f.require_group(group)
        for name, data in kwargs.items():
            if isinstance(data, (list, tuple, dict)):
                name = name + '!p'
                data = np.frombuffer(pickle.dumps(data), dtype=np.uint8)
            g.create_dataset(name, overwrite=True, data=data)


def load_from_zgroup(path, group, names=None):
    """
    Load all datasets from a group in a ZARR file.
    Return them as a dict {name: data}.
    If names is not None, but an iterable of strings,
    ensure that all given names exist as keys;
    otherwise raise a KeyError.
    """
    results = dict()
    with zarr.open(path, mode='r') as f:
        g = f[group]
        for name, data in g.items():
            if (names is not None) and (name not in names):
                continue
            if name.endswith('!p'):
                results[name[:-2]] = pickle.loads(data[:])
            else:
                dims = len(data.shape)
                if dims > 0:
                    results[name] = data[:]
                else:
                    results[name] = data[()]
    if names is not None:
        for name in names:
            if name.endswith('!p'):
                name = name[:-2]
            if name not in results:
                raise KeyError(f"did not get dataset '{name}'' from {path}/{group}")
    return results


def get_zdataset(path, dataset):
    with zarr.open(path, mode='r') as f:
        d = f[dataset]
    return d
