@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set PYDIR=C:\Python314
set REPOROOT=%~dp0..
set SRCDIR=%REPOROOT%\vectorphile\_core\cuda_src
set OUTDIR=%REPOROOT%\vectorphile\_core

echo Building vp_cuda_ext.pyd with nvcc (sm_89)...
nvcc -O3 -arch=sm_89 --shared -allow-unsupported-compiler -std=c++17 ^
     -Xcompiler "/MD /O2 /EHsc /wd4819 /std:c++17" ^
     -I"%PYDIR%\include" ^
     "%SRCDIR%\vp_cuda_kernels.cu" ^
     -L"%PYDIR%\libs" -lpython314 ^
     -o "%OUTDIR%\vp_cuda_ext.cp314-win_amd64.pyd"

if %ERRORLEVEL% neq 0 (
    echo sm_89 failed, retrying with sm_86...
    nvcc -O3 -arch=sm_86 --shared -allow-unsupported-compiler -std=c++17 ^
         -Xcompiler "/MD /O2 /EHsc /wd4819 /std:c++17" ^
         -I"%PYDIR%\include" ^
         "%SRCDIR%\vp_cuda_kernels.cu" ^
         -L"%PYDIR%\libs" -lpython314 ^
         -o "%OUTDIR%\vp_cuda_ext.cp314-win_amd64.pyd"
)

echo CUDA ext build exit code: %ERRORLEVEL%
