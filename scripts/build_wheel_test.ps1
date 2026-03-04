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

# Apply Windows build fix patch.
# Use a single-quoted here-string to avoid PowerShell escaping issues with $, ", and backticks.
# The Replace call ensures LF-only line endings regardless of how this script was saved,
# because git apply requires LF line endings in patch files on Windows.
$patchContent = @'
diff --git a/hopper/setup.py b/hopper/setup.py
index 87f6f45..7d9dbbb 100755
--- a/hopper/setup.py
+++ b/hopper/setup.py
@@ -22,7 +22,8 @@ import urllib.error
 from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
 
 import torch
-from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
+from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME
+from torch.utils.cpp_extension import BuildExtension as _BuildExtension
 
 
 # with open("../README.md", "r", encoding="utf-8") as fh:
@@ -451,7 +452,7 @@ if not SKIP_CUDA_BUILD:
     # We want to use the nvcc front end from 12.6 however, since if we use nvcc 12.8
     # Cutlass 3.8 will expect the new data types in cuda.h from CTK 12.8, which we don't have.
     # For CUDA 13.0+, use system nvcc instead of downloading CUDA 12.x toolchain
-    if bare_metal_version >= Version("12.3") and bare_metal_version < Version("13.0") and bare_metal_version != Version("12.8"):
+    if not IS_WINDOWS and bare_metal_version >= Version("12.3") and bare_metal_version < Version("13.0") and bare_metal_version != Version("12.8"):
         download_and_copy(
             name="nvcc",
             src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin",
@@ -574,12 +575,12 @@ if not SKIP_CUDA_BUILD:
     if DISABLE_BACKWARD:
         sources_bwd_sm90 = []
         sources_bwd_sm80 = []
-    
+
     # Choose between flash_api.cpp and flash_api_stable.cpp based on torch version
     torch_version = parse(torch.__version__)
     target_version = parse("2.9.0.dev20250830")
     stable_args = []
-      
+
     if torch_version >= target_version:
         flash_api_source = "flash_api_stable.cpp"
         stable_args = ["-DTORCH_TARGET_VERSION=0x0209000000000000"]  # Targets minimum runtime version torch 2.9.0
@@ -703,6 +704,80 @@ class CachedWheelsCommand(_bdist_wheel):
             # If the wheel could not be downloaded, build from source
             super().run()
 
+class BuildExtension(_BuildExtension):
+    def build_extensions(self) -> None:
+        original_link_shared_object = self.compiler.link_shared_object
+
+        def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
+            lib_opts = [compiler.library_dir_option(dir) for dir in library_dirs]
+
+            for dir in runtime_library_dirs:
+                lib_opts.extend(compiler.runtime_library_dir_option(dir))
+
+            for lib in libraries:
+                (lib_dir, lib_name) = os.path.split(lib)
+                if lib_dir:
+                    lib_file = compiler.find_library_file([lib_dir], lib_name)
+                    if lib_file:
+                        lib_opts.append(lib_file)
+                    else:
+                        compiler.warn(
+                            f"no library file corresponding to '{lib}' found (skipping)"
+                        )
+                else:
+                    lib_opts.append(compiler.library_option(lib))
+            return lib_opts
+
+        def _link_shared_object(
+                objects, output_filename, output_dir = None, libraries = None, library_dirs = None, runtime_library_dirs = None,
+                export_symbols = None, debug = None, extra_preargs = None, extra_postargs = None, build_temp = None, target_lang = None):
+            if not self.compiler.initialized:
+                self.compiler.initialize()
+
+            library_target = os.path.abspath(output_filename if output_dir is None else os.path.join(output_dir, output_filename))
+            output_dir = os.path.dirname(library_target)
+            objs = [f'{o.replace(":", "$:")}' for o in objects]
+            libraries, library_dirs, runtime_library_dirs = self.compiler._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
+
+            with open(os.path.join(self.build_temp, 'build.ninja'), 'a') as f:
+                library_dirs = [f'"{lib}"' for lib in library_dirs]
+                runtime_library_dirs = [f'"{lib}"' for lib in runtime_library_dirs]
+
+                lib_opts = gen_lib_options(self.compiler, library_dirs, runtime_library_dirs, libraries)
+                export_opts = [f'/EXPORT:{sym}' for sym in (export_symbols or [])]
+
+                ld_args = (
+                    lib_opts + export_opts
+                )
+
+                if export_symbols is not None:
+                    (dll_name, dll_ext) = os.path.splitext(
+                        os.path.basename(output_filename)
+                    )
+                    implib_file = os.path.abspath(os.path.join(self.build_temp, self.compiler.library_filename(dll_name)))
+                    ld_args.append(f'/IMPLIB:"{implib_file}"')
+
+                f.write('\n'.join([
+                    'rule link',
+                    f'  command = "{self.compiler.linker}" /nologo /INCREMENTAL:NO /LTCG /DLL /MANIFEST:EMBED,ID=2 /MANIFESTUAC:NO {" ".join(ld_args)} /out:$out @$out.rsp',
+                    '  rspfile = $out.rsp',
+                    '  rspfile_content = $in_newline',
+                    '',
+                    f'build {library_target.replace(":", "$:")}: link {" ".join(objs)}',
+                ])+'\n')
+
+            self.compiler.mkpath(output_dir)
+            subprocess.check_call(['ninja', '-C', os.path.abspath(self.build_temp), library_target])
+
+        if IS_WINDOWS:
+            self.compiler.link_shared_object = _link_shared_object
+
+        try:
+            super().build_extensions()
+        finally:
+            if IS_WINDOWS:
+                self.compiler.link_shared_object = original_link_shared_object
+
 setup(
     name=PACKAGE_NAME,
     version=get_package_version(),
'@

# Write patch with LF line endings only (replace any CRLF from the script's own line endings)
$patchBytes = [System.Text.Encoding]::UTF8.GetBytes($patchContent.Replace("`r`n", "`n"))
$patchFile = Join-Path $workDir "windows_fix.patch"
[System.IO.File]::WriteAllBytes($patchFile, $patchBytes)

Write-Host "Applying Windows build fix patch..."
git apply --ignore-whitespace $patchFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to apply patch"
    exit 1
}
Write-Host "Patch applied successfully"

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
