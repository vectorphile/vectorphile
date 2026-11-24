@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set REPOROOT=%~dp0..
set SRCDIR=%REPOROOT%\vectorphile\_core\cuda_src
set OUTDIR=%REPOROOT%\vectorphile\_core
set COMPAT=%SRCDIR%\cuda_compat

echo Building PTX kernels (sm_89)...
nvcc -ptx -O3 -arch=compute_89 -allow-unsupported-compiler -I"%COMPAT%" ^
     -D_ENABLE_EXTENDED_ALIGNED_STORAGE -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS ^
     -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ^
     -Xcompiler "/wd4996 /wd4100" ^
     "%SRCDIR%\vp_kernels.cu" -o "%OUTDIR%\vp_kernels_sm89.ptx"
if %ERRORLEVEL% neq 0 (
    echo sm_89 failed, trying sm_86...
    nvcc -ptx -O3 -arch=compute_86 -allow-unsupported-compiler -I"%COMPAT%" ^
         -D_ENABLE_EXTENDED_ALIGNED_STORAGE -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS ^
         -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ^
         -Xcompiler "/wd4996 /wd4100" ^
         "%SRCDIR%\vp_kernels.cu" -o "%OUTDIR%\vp_kernels_sm86.ptx"
)
echo PTX build exit code: %ERRORLEVEL%
