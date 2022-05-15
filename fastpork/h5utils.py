
from collections import defaultdict
import pickle
import numpy as np
import h5py


def guess_q(path, q=None):
    """
    for the hdf5 file at the given path,
    find the group with maximum q and return that value of q.
    Return None if nothing was found.
    """
    if q is not None:
        return q
    with h5py.File(path, 'r') as f:
        for q in range(31, 0, -1):
            testname = "x"*q + '-'
            if testname in f:
                return q
    return None


def guess_group(path, q, group=None):
    """
    for the hdf5 file at the given path and the given q,
    find the given or first named group with counts.
    Return the hdf5 group object.
    Return None if nothing was found.
    """
    if group is not None:
        return group
    shape = "x"*q + '-'  # the final '-' is for canonical codes
    with h5py.File(path, 'r') as f:
        for grp in f[shape]:
            return grp
    return None


def guess_only_group(path, suffix=None, error=True):
    """
    for the hdf5 file at the given path,
    check whether a single group (with the given suffix) exists
    and return the group name.
    If no group is found return None, or throw an error if error=True.
    """
    with h5py.File(path, 'r') as f:
        groups = list()
        for grp in f:
            if suffix is not None:
                if not grp.endswith(suffix): continue
            groups.append(grp)
    if len(groups) == 1:
        return groups[0]
    if error:
        raise RuntimeError("found {} groups in {}".format(len(groups), path))
    return None


def common_datasets(paths):
    """
    Given a list of HDF5 file paths,
    return a list of datasets that occur in all of these files
    """
    datasets = defaultdict(list)
    for inputfile in paths:
        def _collect_datasets(name, node):
            if isinstance(node, h5py.Dataset):
                datasets[name].append(inputfile)
        with h5py.File(inputfile, 'r') as fin:
            fin.visititems(_collect_datasets)
    ninput = len(paths)
    D = [name for name, inputfiles in datasets.items() if len(inputfiles)==ninput]
    return D


## saving and loading to HDF5 groups with auto-pickling (names with suffix '!p') #

def save_to_h5group(path, group, __useattrs=False, **kwargs):
    with h5py.File(path, 'a', libver='latest') as f:
        g = f.require_group(group)
        if __useattrs:
            gx = g.attrs
            creator = g.attrs.create
        else:
            gx = g
            creator = g.create_dataset
        for name, data in kwargs.items():
            if isinstance(data, (list, tuple, dict)):
                name = name + '!p'
                data = np.frombuffer(pickle.dumps(data), dtype=np.uint8)
            elif isinstance(data, str):
                name = name + '!s'  # encode UTF-8 explicitly, mark as string
                data = data.encode("UTF-8")
            # delete if existing and create new
            if name in gx: del gx[name]
            creator(name, data=data)


def save_to_h5attrs(path, group, **kwargs):
    save_to_h5group(path, group, __useattrs=True, **kwargs)


def load_from_h5group(path, group, names=None):
    results = dict()
    with h5py.File(path, 'r') as f:
        g = f[group]
        for name, data in g.items():
            if (names is not None) and (name not in names):
                continue
            if name.endswith('!p'):
                results[name[:-2]] = pickle.loads(data[:])
            elif name.endswith('!s'):  # string was saved as scalar bytes
                results[name[:-2]] = data[()].decode("UTF-8")
            else:
                dims = len(data.shape)
                if dims > 0:
                    results[name] = data[:]
                else:
                    results[name] = data[()]
    if names is not None:
        for name in names:
            if name.endswith('!p') or name.endswith('!s'):
                name = name[:-2]
            if name not in results:
                raise KeyError("Could not get dataset {} from {}/{}".format(name, path, group))
    return results


def get_h5_dataset(path, dataset):
    with h5py.File(path, 'r') as f:
        d = f[dataset]
    return d
