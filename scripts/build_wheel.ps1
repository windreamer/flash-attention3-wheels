<#
.SYNOPSIS
    Build Flash-Attention 3 Python wheel(PowerShell)

.PARAMETER CudaVersion
    CUDA version, default 12.8

.PARAMETER TorchVersion
    PyTorch version, default 2.8.0

.PARAMETER MaxJobs
    Max jobs，default 4
#>
param(
    [string]$CudaVersion = "12.8",
    [string]$TorchVersion = "2.8.0",
    [string]$MaxJobs = "4"
)

$ErrorActionPreference = "Stop"

Write-Host "Building Flash-Attention 3 wheel:"
Write-Host "CUDA Version: $CudaVersion"
Write-Host "PyTorch Version: $TorchVersion"
Write-Host "Max Jobs: $MaxJobs"

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion"
$env:PATH = "$($env:CUDA_HOME)\bin;$($env:PATH)"
$env:LD_LIBRARY_PATH = "$($env:CUDA_HOME)\lib64;$($env:LD_LIBRARY_PATH)"
$env:MAX_JOBS = $MaxJobs
$env:FLASH_ATTENTION_FORCE_BUILD = "TRUE"
$env:CUDAFLAGS = "-t 2"

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install ninja packaging wheel setuptools numpy change-wheel-version
$cuShort = $CudaVersion.Replace(".", "")
pip install torch==$TorchVersion --index-url "https://download.pytorch.org/whl/cu$cuShort"

$workDir = New-TemporaryFile | %{ Remove-Item $_; New-Item -ItemType Directory -Path $_.FullName }
Set-Location $workDir
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
Set-Location flash-attention/hopper

$gitHash = (git rev-parse --short=6 HEAD).Trim()
Write-Host "Current git hash: $gitHash"

Write-Host "Building Flash-Attention 3 wheel..."
python setup.py bdist_wheel

$originalWheel = Get-ChildItem -Path dist -Filter *.whl | Select-Object -First 1 -ExpandProperty FullName
if (-not $originalWheel) {
    Write-Error "Error: Wheel file not found"
    exit 1
}
Write-Host "Original wheel built: $originalWheel"

$buildDate = (Get-Date -Format "yyyyMMdd")
Write-Host "Build date: $buildDate"

$cxx11Abi = "FALSE"
try {
    $abiOut = python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" 2>$null
    if ($abiOut -match "True") { $cxx11Abi = "TRUE" }
} catch {}

$cudaClean = $CudaVersion.Replace(".", "")
$torchClean = $TorchVersion.Replace(".", "")
$localVersion = "${buildDate}.cu${cudaClean}torch${torchClean}cxx11abi${cxx11Abi}.${gitHash}"

Write-Host "Local version identifier: $localVersion"

Write-Host "Modifying wheel with local version..."
$modifiedWheel = python -m change_wheel_version $originalWheel --local-version $localVersion --delete-old-wheel
if (-not $modifiedWheel -or -not (Test-Path $modifiedWheel)) {
    Write-Error "Error: Failed to modify wheel version"
    exit 1
}
Write-Host "Modified wheel created: $modifiedWheel"

$outputDir = "C:\tmp\wheels"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$wheelName = Split-Path -Leaf $modifiedWheel
Copy-Item $modifiedWheel -Destination $outputDir -Force
Write-Host "Wheel saved to: $outputDir\$wheelName"
