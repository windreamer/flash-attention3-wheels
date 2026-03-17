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
$env:CL = "/wd4996"
$env:NVCC_PREPEND_FLAGS = "-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=221 -Xcudafe --diag_suppress=186 -Xcudafe --diag_suppress=550"
$env:DISTUTILS_USE_SDK = 1
$env:PYTHONUNBUFFERED = 1

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install ninja packaging wheel setuptools numpy change-wheel-version
$cuShort = "130"
pip install torch==$TorchVersion --index-url "https://download.pytorch.org/whl/cu$cuShort"

$workDir = New-TemporaryFile | %{ Remove-Item $_; New-Item -ItemType Directory -Path $_.FullName }
Set-Location $workDir
git -c core.autocrlf=false clone --recursive https://github.com/Dao-AILab/flash-attention.git
Set-Location flash-attention

# Apply Windows build fix patch from the repository's scripts directory.
$patchFile = Join-Path $PSScriptRoot "windows_fix.patch"

Write-Host "Applying Windows build fix patch..."
git apply --ignore-whitespace $patchFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to apply patch"
    exit 1
}
Write-Host "Patch applied successfully"

# For CUDA 13.0+, apply Windows C2719 alignment workaround.
# MSVC does not support function-parameter alignment > 64 bytes, which causes
# "error C2719: formal parameter with requested alignment of 128 won't be aligned"
# when compiling NVCC-generated stub code.  The fix caps host-side alignment at
# 64 bytes in cutlass and in the CUDA toolkit header.
# Reference: https://github.com/SystemPanic/vllm-windows/issues/41
$isCuda13 = $false
try {
    $cudaVer = [Version]$CudaVersion
    $isCuda13 = ($cudaVer -ge [Version]"13.0")
} catch {
    Write-Warning "Could not parse CUDA version '$CudaVersion' for alignment-fix check; skipping."
}
if ($isCuda13) {
    Write-Host "CUDA 13.0+ detected, applying C2719 alignment fix for Windows..."

    # --- Patch 1: cutlass cute/container/alignment.hpp ---
    # Adds CUTE_ALIGNAS_HOST_SAFE macro that caps host-side alignment at 64 bytes
    # on MSVC, applied with git apply inside the cutlass submodule git repo.
    $cutlassRoot = Join-Path (Get-Location) "csrc\cutlass"
    if (Test-Path $cutlassRoot) {
        Write-Host "Applying cutlass alignment fix..."
        $cutlassPatch = Join-Path $PSScriptRoot "cutlass_alignment_fix.patch"
        Push-Location $cutlassRoot
        try {
            git apply --ignore-whitespace $cutlassPatch
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to apply cutlass alignment fix patch"
                exit 1
            }
        } finally {
            Pop-Location
        }
        Write-Host "Cutlass alignment fix applied successfully"
    } else {
        Write-Warning "Cutlass submodule not found at $cutlassRoot, skipping patch"
    }

    # --- Patch 2: CUDA toolkit cuda.h (CUtensorMap alignment) ---
    # Lowers CUtensorMap_st alignment from 128 to 64 bytes so that the type can
    # be used as a function parameter without triggering C2719.
    # Applied with patch.exe bundled with Git for Windows (in Git\usr\bin\).
    Write-Host "Applying cuda.h alignment fix..."
    $cudaHPatch = Join-Path $PSScriptRoot "cuda_h_alignment_fix.patch"
    # Locate patch.exe that ships with Git for Windows. It lives in the usr\bin
    # sub-directory relative to the Git installation root. Derive that root from
    # the path of git.exe so that non-standard Git installations are handled.
    $gitExe = (Get-Command git -ErrorAction SilentlyContinue).Source
    if (-not $gitExe) {
        Write-Error "git.exe not found on PATH; cannot locate patch.exe"
        exit 1
    }
    # git.exe is typically at <GitRoot>\cmd\git.exe or <GitRoot>\bin\git.exe
    $gitRoot = Split-Path (Split-Path $gitExe -Parent) -Parent
    $patchExe = Join-Path $gitRoot "usr\bin\patch.exe"
    if (-not (Test-Path $patchExe)) {
        Write-Error "patch.exe not found at expected path: $patchExe"
        exit 1
    }
    & $patchExe --fuzz 2 -p1 --directory $env:CUDA_HOME -i $cudaHPatch
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to apply cuda.h alignment fix patch"
        exit 1
    }
    Write-Host "cuda.h alignment fix applied successfully"
}

Set-Location hopper

$gitHash = (git rev-parse --short=6 HEAD).Trim()
Write-Host "Current git hash: $gitHash"

function Find-VcVarsAll {
    $possiblePaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    )

    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

$vcvarsallPath = if ($VsPath) {
    Join-Path $VsPath "VC\Auxiliary\Build\vcvarsall.bat"
} else {
    Find-VcVarsAll
}

if (-not (Test-Path $vcvarsallPath)) {
    Write-Error "vcvarsall.bat not found at: $vcvarsallPath"
    Write-Error "Please ensure Visual Studio 2019 or later with C++ build tools is installed."
    Write-Error "You can specify the path using -VsPath parameter."
    exit 1
}

Write-Host "Initializing Visual Studio build environment..."
Write-Host "Using vcvarsall.bat: $vcvarsallPath"

Write-Host "Building Flash-Attention 3 wheel..."
$buildCmd = "`"$vcvarsallPath`" x64 && python setup.py bdist_wheel 2>&1"
cmd /c $buildCmd | Select-String -Pattern 'ptxas info|bytes stack frame,' -NotMatch

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
