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

    # MSVC maximum supported alignment for function parameters is 64 bytes.
    $maxMsvcAlignment = 64

    # --- Patch 1: cutlass cute/container/alignment.hpp ---
    # Add CUTE_ALIGNAS_HOST_SAFE macro that caps host-side alignment at
    # $maxMsvcAlignment on MSVC, keeping the original value elsewhere.
    # Try known submodule locations first to avoid a slow full-tree search.
    $alignmentHpp = $null
    $cutlassSearchRoots = @(
        "csrc\cutlass\include\cute\container\alignment.hpp",
        "third_party\cutlass\include\cute\container\alignment.hpp"
    )
    foreach ($rel in $cutlassSearchRoots) {
        $candidate = Join-Path (Get-Location) $rel
        if (Test-Path $candidate) {
            $alignmentHpp = $candidate
            break
        }
    }
    if (-not $alignmentHpp) {
        # Fall back to a bounded recursive search if the submodule is in an unexpected location.
        $alignmentHpp = Get-ChildItem -Path . -Recurse -Depth 10 -Filter "alignment.hpp" |
            Where-Object { ($_.DirectoryName -replace '\\', '/') -match 'cute/container' } |
            Select-Object -First 1 -ExpandProperty FullName
    }

    if ($alignmentHpp) {
        Write-Host "Patching cutlass alignment.hpp: $alignmentHpp"
        $src = [System.IO.File]::ReadAllText($alignmentHpp)

        # Insert CUTE_ALIGNAS_HOST_SAFE definition after CUTE_ALIGNAS in the __CUDACC__ branch.
        $src = $src -replace `
            '(?m)^(#\s*define CUTE_ALIGNAS\(n\) __align__\(n\))([ \t]*)$', `
            "`$1`$2`n#  if defined(_MSC_VER)`n#    define CUTE_ALIGNAS_HOST_SAFE(n) __align__($maxMsvcAlignment)`n#  else`n#    define CUTE_ALIGNAS_HOST_SAFE(n) __align__(n)`n#  endif"

        # Insert CUTE_ALIGNAS_HOST_SAFE definition after CUTE_ALIGNAS in the non-CUDACC branch.
        $src = $src -replace `
            '(?m)^(#\s*define CUTE_ALIGNAS\(n\) alignas\(n\))([ \t]*)$', `
            "`$1`$2`n#  define CUTE_ALIGNAS_HOST_SAFE(n) alignas(n)"

        # Use CUTE_ALIGNAS_HOST_SAFE for the 128- and 256-byte aligned_struct specialisations.
        $src = $src -replace 'CUTE_ALIGNAS\(128\)(\s+aligned_struct<128)', 'CUTE_ALIGNAS_HOST_SAFE(128)$1'
        $src = $src -replace 'CUTE_ALIGNAS\(256\)(\s+aligned_struct<256)', 'CUTE_ALIGNAS_HOST_SAFE(256)$1'

        [System.IO.File]::WriteAllText($alignmentHpp, $src)
        Write-Host "Patched cutlass alignment.hpp successfully"
    } else {
        Write-Warning "cutlass alignment.hpp not found, skipping patch"
    }

    # --- Patch 2: CUDA toolkit cuda.h (CUtensorMap alignment) ---
    # MSVC rejects types aligned beyond $maxMsvcAlignment bytes when used as
    # function parameters.  Lower the CUtensorMap_st alignment accordingly.
    # The struct uses two preprocessor branches (C++ alignas and C11 _Alignas),
    # so each is replaced separately via targeted non-greedy patterns.
    $cudaH = Join-Path $env:CUDA_HOME "include\cuda.h"
    if (Test-Path $cudaH) {
        Write-Host "Patching cuda.h: $cudaH"
        $src = [System.IO.File]::ReadAllText($cudaH)

        $src = [regex]::Replace($src,
            '(?s)(typedef struct CUtensorMap_st \{.*?)alignas\(128\)',
            "`${1}alignas($maxMsvcAlignment)")
        $src = [regex]::Replace($src,
            '(?s)(typedef struct CUtensorMap_st \{.*?)_Alignas\(128\)',
            "`${1}_Alignas($maxMsvcAlignment)")

        [System.IO.File]::WriteAllText($cudaH, $src)
        Write-Host "Patched cuda.h successfully"
    } else {
        Write-Warning "cuda.h not found at $cudaH, skipping patch"
    }
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
