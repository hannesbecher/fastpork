import numba
import llvmlite
import numpy as np

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
def generate_compare_xchange(dtype):

    assert isinstance(dtype, str)
    assert dtype in ir_types

    load_volatile_ir = f"define i1 @compare_xchange{ir_types[dtype]}({ir_types[dtype]}* %address, {ir_types[dtype]} %cmp, {ir_types[dtype]} %new) {{\n" \
    f"    %val_success = cmpxchg volatile {ir_types[dtype]}* %address, {ir_types[dtype]} %cmp, {ir_types[dtype]} %new acq_rel monotonic\n" \
    f"    %value_loaded = extractvalue {{ {ir_types[dtype]}, i1 }} %val_success, 0\n"\
    f"    %success = extractvalue {{ {ir_types[dtype]}, i1 }} %val_success, 1\n"\
    f"    ret i1 %success\n" \
    f"}}"

    @numba.extending.intrinsic
    def build_compare_xchange(typingctx, address, cmp, new):
        if isinstance(address, numba.types.Integer):
            def codegen(context, builder, sig, args):
                return insert_ir_call(
                    load_volatile_ir,
                    f'compare_xchange{ir_types[dtype]}',
                    context, builder, sig, args)
            signature = numba.uint8(numba.types.intp, numba_types[dtype], numba_types[dtype])
            return signature, codegen

    @numba.njit(nogil=True)
    def compare_xchange_array_pos(a, i, comp, new):
        return build_compare_xchange(a.ctypes.data + a.itemsize * i, comp, new)

    return compare_xchange_array_pos


def test_cmpxchg():
    array = np.zeros(1, dtype=np.uint64)
    print(array)
    cmpxchg = generate_compare_xchange("uint64")

    numba.njit()
    def test(a):
        print(cmpxchg(a,0,0,115))
        print(cmpxchg(a,0,0,115))
    test(array)
    print(array)


if __name__ == "__main__":
    test_cmpxchg()