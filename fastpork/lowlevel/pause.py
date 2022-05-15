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

from platform import machine
from warnings import warn

import numba
from llvmlite import ir


MACHINE = machine().lower()  # 'arm64',  'x86_64', 'amd64'


def compile_pause():
    """
    Compile and return the pause function.
    This may fail badly wtih an LLVM error
    if we don't check the machine's capabilities correctly.
    """
    if MACHINE in ('x86_64', 'amd64'):
        ##print(f"compiling for {MACHINE}")
        @numba.extending.intrinsic
        def pause(typingctx):
            """do nothing for a while (pause)"""
            def codegen(context, builder, sig, args):
                void_t = ir.VoidType()
                fty = ir.FunctionType(void_t, [])
                _pause = builder.module.declare_intrinsic(
                    "llvm.x86.sse2.pause", fnty=fty)
                builder.call(_pause, [])

            sig = numba.void()
            return sig, codegen

    elif MACHINE == 'arm64':
        ##print(f"compiling for {MACHINE}")
        @numba.extending.intrinsic
        def pause(typingctx):
            """do nothing for a while (pause)"""
            def codegen(context, builder, sig, args):
                void_t = ir.VoidType()
                int32_t = ir.IntType(32)
                const1 = ir.Constant(int32_t, 1)
                fty = ir.FunctionType(void_t, (int32_t,))
                _hint = builder.module.declare_intrinsic(
                    "llvm.aarch64.hint", fnty=fty)
                builder.call(_hint, (const1,))

            sig = numba.void()
            return sig, codegen

    else:  # unknown MACHINE
        warn(f"Unsupported Machine '{MACHINE}' for pause(); using busy waiting")
        @numba.njit
        def pause():
            return None

    return pause


def test_pause():
    pause = compile_pause()
    @numba.njit
    def use_pause():
        pause()
    result = use_pause()  # run it once to auto-compile
    ##use_pause.inspect_types()  # DEBUG
    asm = use_pause.inspect_asm()
    for sig, code in asm.items():  print(sig, '\n=========\n', code, '\n')
    assert result is None


if __name__ == "__main__":
    test_pause()
