# CUDA Password Cracker build script (Windows)
# Requires: NVIDIA CUDA Toolkit (nvcc) + MSVC toolchain
# Usage: .\build.ps1 [arch]   (default arch = 120, e.g. .\build.ps1 90 for RTX 30 series)
param([string]$Arch = "120")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $root "build"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host "==> building SHA-256 cracker (sm_$Arch)"
nvcc -O3 -arch=sm_$Arch (Join-Path $root "src\brute_sha256.cu") -o (Join-Path $out "brute_sha256.exe")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> building ZIP cracker (sm_$Arch)"
nvcc -O3 -arch=sm_$Arch (Join-Path $root "src\zipcrack.cu") -o (Join-Path $out "zipcrack.exe")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> building CPU reference cracker"
nvcc -O2 (Join-Path $root "src\zipcrack_cpu.cpp") -o (Join-Path $out "zipcrack_cpu.exe")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> done. binaries in $out"
Write-Host "    tip: ensure CUDA bin dir is in PATH before running, e.g.:"
Write-Host '    $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;" + $env:PATH'
