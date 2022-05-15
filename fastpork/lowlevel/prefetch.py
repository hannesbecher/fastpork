"""
prefetch:
Module that implements a numba extension:

prefetch_array_element(a, i):
    For a numpy array a and an index i,
    prefetch a[i] from RAM

To access this function in your code, you need to run:

```
from prefetch import compile_prefetch_array_element
prefetch_array_element = compile_prefetch_array_element()
```
"""

import numba
from llvmlite import ir

# Factory function that will compile and return
# prefetch_array_element(a, i)


def compile_prefetch_array_element():
    """Compile and return the prefetch_array_element function"""
    @numba.extending.intrinsic
    def prefetch_address(typingctx, address):
        """prefetch given memory address (uint64 or int64)"""

        if isinstance(address, numba.types.Integer):

            def codegen(context, builder, sig, args):
                int32_t = ir.IntType(32)
                int8_p = ir.IntType(8).as_pointer()
                const0 = ir.Constant(int32_t, 0)
                const1 = ir.Constant(int32_t, 1)
                prefetch = builder.module.declare_intrinsic(
                    "llvm.prefetch",
                    fnty=ir.FunctionType(
                        ir.VoidType(), (int8_p, int32_t, int32_t, int32_t)
                    ),
                )
                ptr = builder.inttoptr(args[0], int8_p)
                builder.call(prefetch, (ptr, const0, const0, const1))

            sig = numba.void(numba.types.uintp)
            return sig, codegen

    @numba.njit
    def prefetch_array_element(a, i):
        """prefetch array element a[i]"""
        return prefetch_address(a.ctypes.data + a.itemsize * i)

    return prefetch_array_element

    # @numba.extending.overload_method(numba.types.Array, 'prefetch')
    # def array_prefetch(arr, index):
    #    if isinstance(index, numba.types.Integer):
    #        def prefetch_impl(arr, index):
    #            return prefetch_array_element(arr, index)
    #        return prefetch_impl
