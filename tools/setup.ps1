param(
    [Parameter(Mandatory=$true)][string]$OnnxRuntimeRoot,
    [string]$Compiler = 'C:\msys64\ucrt64\bin\g++.exe',
    [switch]$Test
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
function Invoke-Checked {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $File" }
}
if (-not (Test-Path -LiteralPath $Compiler)) { throw "Install the MSYS2 UCRT64 GCC toolchain or provide -Compiler." }
$sdk = (Resolve-Path -LiteralPath $OnnxRuntimeRoot).Path
foreach ($relative in @('include/onnxruntime_cxx_api.h', 'lib/onnxruntime.lib', 'lib/onnxruntime.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $sdk $relative))) { throw "Incomplete Windows x64 ORT SDK: $relative" }
}
$compilerDirectory = Split-Path $Compiler -Parent
$toolchainRoot = Split-Path $compilerDirectory -Parent
foreach ($relative in @('include/zmq.h', 'include/nlohmann/json.hpp')) {
    if (-not (Test-Path -LiteralPath (Join-Path $toolchainRoot $relative))) { throw "Missing MSYS2 dependency: $relative" }
}
$env:PATH = "$compilerDirectory;$projectRoot\.venv\Scripts;" + $env:PATH
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Invoke-Checked -File 'python' -Arguments @('-m', 'venv', '.venv')
}
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
Invoke-Checked -File $python -Arguments @('-m', 'pip', 'install', '-r', 'requirements-phase2.txt', '-r', 'requirements-phase4.txt')
$cmake = Join-Path $projectRoot '.venv/Scripts/cmake.exe'
$ninja = Join-Path $projectRoot '.venv/Scripts/ninja.exe'
Invoke-Checked -File $cmake -Arguments @('-S', '.', '-B', 'cmake-build-phase5e', '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_CXX_COMPILER=$Compiler", "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DONNXRUNTIME_ROOT=$sdk", "-DPython3_EXECUTABLE=$python")
Invoke-Checked -File $cmake -Arguments @('--build', 'cmake-build-phase5e', '--parallel', '4')
if ($Test) {
    Invoke-Checked -File (Join-Path $projectRoot '.venv/Scripts/ctest.exe') -Arguments @('--test-dir', 'cmake-build-phase5e', '--output-on-failure')
}
if (Test-Path -LiteralPath 'model/artifacts/phase5/staged_export_manifest.json') {
    Invoke-Checked -File $python -Arguments @('-m', 'tools.evidence', 'doctor')
} else {
    Write-Host 'Build complete. Transfer a verified stage bundle or export your checkpoint before running the image workflow.'
}
