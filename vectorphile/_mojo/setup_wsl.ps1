# vectorphile Mojo backend setup
# Run this script in an ADMIN PowerShell once, then restart when prompted.
# After restart, run it again — it will skip the WSL install and go straight to Mojo.

$ErrorActionPreference = "Stop"

Write-Host "=== vectorphile Mojo Backend Setup ===" -ForegroundColor Cyan

# Step 1: Check/install WSL2
$wslInstalled = $false
try {
    $wslInfo = wsl --status 2>&1
    if ($LASTEXITCODE -eq 0) { $wslInstalled = $true }
} catch {}

if (-not $wslInstalled) {
    Write-Host "`nWSL2 not found. Installing..." -ForegroundColor Yellow
    Write-Host "This will enable WSL2 and install Ubuntu. A RESTART may be required." -ForegroundColor Yellow
    $confirm = Read-Host "Proceed? (y/n)"
    if ($confirm -ne "y") { exit 0 }

    wsl --install --distribution Ubuntu
    Write-Host "`nWSL2 installed. PLEASE RESTART your computer, then run this script again." -ForegroundColor Green
    exit 0
}

Write-Host "WSL2 found." -ForegroundColor Green

# Step 2: Install Mojo inside WSL
Write-Host "`nInstalling Mojo (MAX SDK) inside WSL Ubuntu..." -ForegroundColor Cyan

$mojoInstall = @'
set -e
echo "--- Installing Mojo / MAX SDK ---"
if ! command -v mojo &> /dev/null; then
    curl -ssL https://magic.modular.com | bash
    export PATH="$HOME/.modular/bin:$PATH"
    echo 'export PATH="$HOME/.modular/bin:$PATH"' >> ~/.bashrc
    magic install max
    echo "Mojo installed successfully."
else
    echo "Mojo already installed: $(mojo --version)"
fi
'@

wsl -d Ubuntu -- bash -c $mojoInstall

# Step 3: Build the kernels
Write-Host "`nBuilding vectorphile Mojo kernels..." -ForegroundColor Cyan

# Convert Windows path to WSL path
$projPath = (Get-Location).Path -replace '\\', '/' -replace '^([A-Z]):', '/mnt/$1'.ToLower()
# Fix drive letter casing
$projPath = $projPath -replace '^/mnt/([A-Z])', { '/mnt/' + $_.Groups[1].Value.ToLower() }

$buildCmd = "cd '$projPath' && export PATH=`$HOME/.modular/bin:`$PATH && python vectorphile/_mojo/build.py"

wsl -d Ubuntu -- bash -c $buildCmd

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "vectorphile will now use Mojo-accelerated kernels automatically." -ForegroundColor Green
Write-Host "Mojo path: vectorphile/_mojo/libvpkernels.so" -ForegroundColor Gray
