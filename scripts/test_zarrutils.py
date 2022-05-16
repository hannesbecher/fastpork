"""
test_zarrutils.py:
test for zarrutils
"""

import numpy as np
from zarrutils import save_to_zgroup, load_from_zgroup, get_zdataset


def test_write_read():
    A = np.arange(0, 2**33, 2**16-1, dtype=np.uint64)
    info = dict(
        hashtype='something',
        hashfunc='linear1:affine2+3:special4',
        rcmode='maxmin',
        somenumber=17,
        afloat=11.11,
        )
    save_to_zgroup('test1.zarr', '/', data=A)
    save_to_zgroup('test1.zarr', '/', info=info)
    contents = load_from_zgroup('test1.zarr' ,'/')
    A2 = contents['data']
    print(f"A: {type(A)}: {A.shape}")
    print(f"A2: {type(A2)}: {A2.shape}")
    info2 = contents['info']
    assert np.all(A == A2)
    assert info == info2
    A3 = get_zdataset('test1.zarr', '/data')[:]
    print(f"A3: {type(A3)}: {A3.shape}")
    assert np.all(A == A3)
    print(f"info: {type(info)}: {info}")
    print(f"info2: {type(info2)}: {info2}")
