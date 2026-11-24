@echo off
REM ============================================================
REM  setup_mojo_wsl.bat
REM  Installs Mojo/Magic inside WSL2 Ubuntu and builds
REM  the vectorphile Mojo backend (_vp_mojo.so).
REM
REM  Prerequisites: WSL2 with Ubuntu distribution installed.
REM  Usage: scripts\setup_mojo_wsl.bat
REM ============================================================
setlocal

echo === vectorphile Phase 6: Mojo/MAX backend setup ===
echo.

REM -- Check WSL is available ---
wsl --status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: WSL2 not found. Install from Microsoft Store or:
    echo   wsl --install
    exit /b 1
)

REM -- Map Windows repo path to WSL path ---
REM  Convert S:\My Data\Github\vectorphile -> /mnt/s/My Data/Github/vectorphile
set WIN_REPO=%~dp0..
for /f "delims=" %%i in ('wsl wslpath -u "%WIN_REPO%"') do set WSL_REPO=%%i
echo Repo in WSL: %WSL_REPO%

REM -- Install Magic + Mojo if not present, then build ---
wsl -d Ubuntu -- bash -lc "
    set -e

    # Install magic if needed
    if ! command -v magic &>/dev/null; then
        echo 'Installing Modular magic CLI...'
        curl -ssL https://magic.modular.com/install.sh | bash
        export PATH=\"$HOME/.modular/bin:$PATH\"
    fi

    # Install mojo if needed
    if ! command -v mojo &>/dev/null; then
        echo 'Installing Mojo...'
        magic install mojo
        export PATH=\"$HOME/.modular/bin:$PATH\"
    fi

    echo 'Mojo version:' $(mojo --version)

    # Install clang if needed (for linking .so)
    if ! command -v clang &>/dev/null; then
        echo 'Installing clang...'
        sudo apt-get install -y clang 2>/dev/null || true
    fi

    # Build the backend
    echo 'Building vectorphile Mojo backend...'
    bash '%WSL_REPO%/scripts/build_vp_mojo_linux.sh'
"

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED. Check output above.
    exit /b 1
)

echo.
echo SUCCESS: _vp_mojo.so built.
echo Run Python in WSL to use the Mojo backend:
echo   wsl -d Ubuntu -- python3 -c "import vectorphile as vp; print(vp.mojo.is_available())"
endlocal
