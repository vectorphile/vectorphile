@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set PYDIR=C:\Python314
set REPOROOT=%~dp0..
set SRCDIR=%REPOROOT%\vectorphile\_core\csrc
set ISPCDIR=%REPOROOT%\vectorphile\_core\ispc
set OUTDIR=%REPOROOT%\vectorphile\_core
set ISPC_EXE=C:\ispc\ispc-v1.25.3-windows\bin\ispc.exe

set ISPC_OBJ=
set ISPC_FLAGS=

if exist "%ISPC_EXE%" (
    echo Compiling ISPC kernels...
    "%ISPC_EXE%" "%ISPCDIR%\vp_kernels.ispc" --target=avx2-i32x8 --arch=x86-64 --math-lib=fast --opt=fast-math -o "%ISPCDIR%\vp_kernels.obj" -h "%ISPCDIR%\vp_kernels_ispc.h"
    if %ERRORLEVEL% neq 0 (
        echo ISPC svml failed, retrying with default math...
        "%ISPC_EXE%" "%ISPCDIR%\vp_kernels.ispc" --target=avx2-i32x8 --arch=x86-64 --opt=fast-math -o "%ISPCDIR%\vp_kernels.obj" -h "%ISPCDIR%\vp_kernels_ispc.h"
    )
    if %ERRORLEVEL% equ 0 (
        set ISPC_OBJ="%ISPCDIR%\vp_kernels.obj"
        set ISPC_FLAGS=/DVP_HAVE_ISPC /I"%ISPCDIR%"
        echo ISPC kernels OK.
    ) else (
        echo WARNING: ISPC failed, building AVX2-only.
    )
) else (
    echo ISPC not found, building AVX2-only.
)

rem Build with AVX2 (safe default for all modern x86-64 CPUs including 12th/13th Gen Intel
rem which disable AVX-512 in microcode). Use /arch:AVX512 only on Skylake-X/Ice Lake/Zen 4+.
cl.exe /nologo /O2 /arch:AVX2 /openmp /fp:fast /GL /W3 /I"%PYDIR%\include" %ISPC_FLAGS% /LD /MD "%SRCDIR%\vp_core.c" %ISPC_OBJ% /link /LIBPATH:"%PYDIR%\libs" python314.lib /OUT:"%OUTDIR%\_vp_core.cp314-win_amd64.pyd" /IMPLIB:"%OUTDIR%\_vp_core.lib" /LTCG

echo Build exit code: %ERRORLEVEL%
