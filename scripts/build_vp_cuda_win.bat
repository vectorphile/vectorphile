@echo off
REM Build script for vectorphile CUDA extension (_vp_cuda.pyd)
REM Requires: CUDA Toolkit (nvcc), Python dev headers, MSVC
REM Usage: scripts\build_vp_cuda_win.bat

setlocal

REM ---- Configuration ---------------------------------------------------------
REM Adjust these paths as needed:
set PYTHON_INCLUDE=%LOCALAPPDATA%\Programs\Python\Python312\include
set PYTHON_LIBS=%LOCALAPPDATA%\Programs\Python\Python312\libs

REM Auto-detect Python if above is wrong:
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path(\"include\"))"') do set PYTHON_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var(\"installed_base\") + \"\\libs\")"') do set PYTHON_LIBS=%%i
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version[:3].replace(\".\",\"\"))"') do set PYVER=%%i

set SRC=vectorphile\_core\csrc\vp_cuda.cu
set OUT=vectorphile\_core\_vp_cuda.pyd

REM ---- Architecture flags (Ampere/Turing/Ada — adjust for your GPU) ----------
REM sm_86 = RTX 30xx (Ampere)   sm_89 = RTX 40xx (Ada)   sm_75 = RTX 20xx (Turing)
REM sm_70+ enables Tensor Core WMMA API (kernel_gemm_tc_f32)
set ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89

REM ---- Compile ----------------------------------------------------------------
echo Building %OUT% ...
nvcc %ARCH% ^
     -O3 ^
     --use_fast_math ^
     --expt-relaxed-constexpr ^
     -DVP_TENSOR_CORE_AVAILABLE ^
     -Xcompiler "/MD /O2 /W3" ^
     -I "%PYTHON_INCLUDE%" ^
     -L "%PYTHON_LIBS%" ^
     -lpython%PYVER% ^
     --shared ^
     -o "%OUT%" ^
     "%SRC%"

if %ERRORLEVEL% == 0 (
    echo.
    echo SUCCESS: %OUT% built.
) else (
    echo.
    echo FAILED — check nvcc/CUDA toolkit installation.
    exit /b 1
)
endlocal
