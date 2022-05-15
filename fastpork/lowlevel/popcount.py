"""
pause:
Module that implements a numba extension:

pause():
    run the SSE2 pause assembly instruction
    to keep CPU power consumption low during spin locks

To access this function in your code, you need to run:

```
from pause import compile_pause
pause = compile_pause()
```
"""

import numba
from llvmlite import ir

ir_types_str = {
    "uint8" : "i8",
    "int8" : "i8",
    "uint16" : "i16",
    "int16" : "i16",
    "uint32" : "i32",
    "int32" : "i32",
    "uint64" : "i64",
    "int64" : "i64",
}

ir_types = {
    "uint8" : ir.IntType(8),
    "int8" : ir.IntType(8),
    "uint16" : ir.IntType(16),
    "int16" : ir.IntType(16),
    "uint32" : ir.IntType(32),
    "int32" : ir.IntType(32),
    "uint64" : ir.IntType(64),
    "int64" : ir.IntType(64),
}

numba_types = {
    "int8" : numba.int8,
    "uint8" : numba.uint8,
    "int16" : numba.int16,
    "uint16" : numba.uint16,
    "int32" : numba.int32,
    "uint32" : numba.uint32,
    "int64" : numba.int64,
    "uint64" : numba.uint64,
}

def compile_popcount(dtype):
    """
    Compile and return the ctpop function.
    """
    assert isinstance(dtype, str)
    assert dtype in ir_types

    @numba.extending.intrinsic
    def blob(typingctx, value):
        def codegen(context, builder, sig, args):
            fty = ir.FunctionType(ir_types[dtype], (ir_types[dtype],))
            _popcount = builder.module.declare_intrinsic(
                f"llvm.ctpop.{ir_types_str[dtype]}", fnty=fty)
            return builder.call(_popcount, (args[0],))
        sig = numba_types[dtype](numba_types[dtype])
        return sig, codegen

    return blob


def test_popcount():
    popct = compile_popcount("uint64")

    @numba.njit
    def ctpop(a):
        return popct(a)

    t = numba.uint64(12297829382473034410)
    assert ctpop(t) == 32


if __name__ == "__main__":
    test_popcount()
