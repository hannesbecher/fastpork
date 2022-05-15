# https://github.com/numba/numba/issues/2795#issuecomment-471820906

import numba
import llvmlite
import numpy as np
import inspect

def insert_ir_call(ir_func_text, ir_func_name, context, builder, sig, args):
    # Get the context library to add the IR to.
    active_library = context.active_code_library
    # Test if the IR function was already added; a NameError if raised
    # if it is not found.
    try:
        active_library.get_function(ir_func_name)
    except NameError:
        # Parse and add the IR.
        ll_module = llvmlite.binding.parse_assembly(ir_func_text)
        ll_module.verify()
        active_library.add_llvm_module(ll_module)

    # Insert, or look up, the function in the builder module.
    # Code is similar to numba.cgutils.insert_pure_function, but doesn't
    # set the "readonly" flag since the intrinsic may change
    # pointed-to values.
    function = numba.core.cgutils.get_or_insert_function(
        builder.module,
        fnty = llvmlite.ir.FunctionType(
            return_type = context.get_argument_type(sig.return_type),
            args = [context.get_argument_type(aty) for aty in sig.args]
            ),
        name = ir_func_name)

    # Add 'nounwind' attribute; no exception handling is wanted.
    function.attributes.add('nounwind')

    # Add a Call node to the IR function.
    retval = context.call_external_function(builder, function, sig.args, args)
    return retval

ir_types = {
    "uint8" : "i8",
    "int8" : "i8",
    "uint16" : "i16",
    "int16" : "i16",
    "uint32" : "i32",
    "int32" : "i32",
    "uint64" : "i64",
    "int64" : "i64",
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
def generate_volatile_load(dtype):

    assert isinstance(dtype, str)
    assert dtype in ir_types

    load_volatile_ir = f"define {ir_types[dtype]} @load_volatile_{ir_types[dtype]}({ir_types[dtype]}* %address) {{\n" \
    f"    %res = load volatile {ir_types[dtype]}, {ir_types[dtype]}* %address\n" \
    f"    ret {ir_types[dtype]} %res\n" \
    f"}}"

    @numba.extending.intrinsic
    def volatile_load(typingctx, address):
        if isinstance(address, numba.types.Integer):
            def codegen(context, builder, sig, args):
                return insert_ir_call(
                    load_volatile_ir,
                    f'load_volatile_{ir_types[dtype]}',
                    context, builder, sig, args)
            signature = numba_types[dtype](numba.types.intp)
            return signature, codegen

    @numba.njit(nogil=True)
    def volatile_load_array_pos(a,i):
        return volatile_load(a.ctypes.data + a.itemsize * i)

    return volatile_load_array_pos

def generate_volatile_store(dtype):

    assert isinstance(dtype, str)
    assert dtype in ir_types

    store_volatile_ir = f"define void @store_volatile_{ir_types[dtype]}({ir_types[dtype]}* %address, {ir_types[dtype]} %value) {{\n"\
    f"    store volatile {ir_types[dtype]} %value, {ir_types[dtype]}* %address\n"\
    f"    ret void"\
    f"}}"

    @numba.extending.intrinsic
    def volatile_store(typingctx, address, value):
        if isinstance(address, numba.types.Integer) and isinstance(value, numba.types.Integer):
            def codegen(context, builder, sig, args):
                return insert_ir_call(
                    store_volatile_ir,
                    f'store_volatile_{ir_types[dtype]}',
                    context, builder, sig, args)
            signature = numba.void(numba.types.intp, numba_types[dtype]) # (int64, intx) -> none
            return signature, codegen

    @numba.njit(nogil=True)
    def volatile_store_array_pos(a,i,v):
        assert a.dtype is np.dtype(dtype)
        return volatile_store(a.ctypes.data + a.itemsize * i, v)

    return volatile_store_array_pos