$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir
$workspaceRoot = Split-Path -Parent $projectRoot
$buildDir = Join-Path $projectRoot 'build'

$cmakeExe = 'cmake'
$vsCmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (Test-Path $vsCmake) {
    $cmakeExe = $vsCmake
}

function Test-TensorRTRoot([string]$root) {
    if (-not $root) { return $false }
    $includeOk = Test-Path (Join-Path $root 'include\NvInfer.h')
    $libOk = (Test-Path (Join-Path $root 'lib\nvinfer.lib')) -or (Test-Path (Join-Path $root 'lib64\nvinfer.lib'))
    return ($includeOk -and $libOk)
}

function Find-TensorRTRoot {
    $candidates = @()
    if ($env:TENSORRT_ROOT) { $candidates += $env:TENSORRT_ROOT }

    $candidates += @(
        'C:\TensorRT',
        'C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT',
        'C:\tools\TensorRT'
    )

    $downloads = Join-Path $env:USERPROFILE 'Downloads'
    if (Test-Path $downloads) {
        $candidates += Get-ChildItem -Path $downloads -Directory -Filter 'TensorRT*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    }

    $candidates += Get-ChildItem -Path 'C:\' -Directory -Filter 'TensorRT*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-TensorRTRoot $candidate) {
            return $candidate
        }
    }

    return $null
}

function Find-OpenCVDir {
    $candidates = @()
    if ($env:OpenCV_DIR) { $candidates += $env:OpenCV_DIR }
    if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX 'Library\cmake') }

    $condaPkgs = Join-Path $env:USERPROFILE '.conda\pkgs'
    if (Test-Path $condaPkgs) {
        $pkgDirs = Get-ChildItem -Path $condaPkgs -Directory -Filter 'opencv-*' -ErrorAction SilentlyContinue
        foreach ($pkg in $pkgDirs) {
            $candidates += (Join-Path $pkg.FullName 'Library\cmake')
            $candidates += (Join-Path $pkg.FullName 'Library\cmake\x64\vc16\lib')
        }
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path (Join-Path $candidate 'OpenCVConfig.cmake')) {
            return $candidate
        }
    }

    return $null
}

$tensorrtRoot = Find-TensorRTRoot
if (-not $tensorrtRoot) {
    Write-Host ''
    Write-Host '[ERROR] TensorRT C++ SDK not found: include\NvInfer.h + lib\nvinfer.lib are required.' -ForegroundColor Red
    Write-Host 'Only Python tensorrt package is installed; it cannot compile C++ code.' -ForegroundColor Yellow
    Write-Host 'Please extract TensorRT SDK to a folder like C:\TensorRT-10.x.x.x, then run this script again.' -ForegroundColor Yellow
    exit 1
}

$opencvDir = Find-OpenCVDir
if (-not $opencvDir) {
    Write-Host '[WARN] OpenCVConfig.cmake not found automatically; CMake may fail.' -ForegroundColor Yellow
} else {
    Write-Host "[OK] OpenCV_DIR = $opencvDir" -ForegroundColor Green
}

Write-Host "[OK] TENSORRT_ROOT = $tensorrtRoot" -ForegroundColor Green

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

$cmakeArgs = @(
    '-S', $projectRoot,
    '-B', $buildDir,
    '-G', 'Visual Studio 16 2019',
    '-A', 'x64',
    "-DTENSORRT_ROOT=$tensorrtRoot"
)

if ($opencvDir) {
    $cmakeArgs += "-DOpenCV_DIR=$opencvDir"
}

Write-Host ''
Write-Host '== CMake Configure ==' -ForegroundColor Cyan
& $cmakeExe @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '[ERROR] CMake configure failed.' -ForegroundColor Red
    Write-Host 'If you see "No CUDA toolset found", install/repair VS C++ workload and CUDA Visual Studio Integration.' -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host ''
Write-Host '== CMake Build (Release) ==' -ForegroundColor Cyan
& $cmakeExe --build $buildDir --config Release --target wafer_infer
if ($LASTEXITCODE -ne 0) {
    Write-Host '[ERROR] Build failed. Check logs above.' -ForegroundColor Red
    exit $LASTEXITCODE
}

$exe = Join-Path $buildDir 'Release\wafer_infer.exe'
if (-not (Test-Path $exe)) {
    Write-Host '[WARN] Build succeeded but executable not found at Release\wafer_infer.exe' -ForegroundColor Yellow
    exit 0
}

$engineCandidates = @(
    (Join-Path $projectRoot 'mobilenetv4.engine'),
    (Join-Path $workspaceRoot 'mobilenetv4.engine'),
    (Join-Path $buildDir 'mobilenetv4.engine')
)
$enginePath = $null
foreach ($candidate in ($engineCandidates | Select-Object -Unique)) {
    if (Test-Path $candidate) {
        $enginePath = $candidate
        break
    }
}

$runtimePathCandidates = @(
    (Join-Path $tensorrtRoot 'lib'),
    (Join-Path $tensorrtRoot 'lib64'),
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin'
)

if ($env:CONDA_PREFIX) {
    $runtimePathCandidates += (Join-Path $env:CONDA_PREFIX 'Library\bin')
}

if ($opencvDir) {
    if ($opencvDir -match '^(.*\\Library)\\cmake(\\.*)?$') {
        $runtimePathCandidates += (Join-Path $Matches[1] 'bin')
    }
    $runtimePathCandidates += (Join-Path $opencvDir 'bin')
}

$runtimePath = ($runtimePathCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique) -join ';'
if ($runtimePath) {
    $env:PATH = "$runtimePath;$env:PATH"
}

Write-Host ''
Write-Host '== Run ==' -ForegroundColor Cyan
Push-Location $buildDir
try {
    if ($enginePath) {
        Write-Host "[OK] Engine = $enginePath" -ForegroundColor Green
        & $exe $enginePath
    } else {
        Write-Host '[WARN] Engine path not provided; executable will auto-search.' -ForegroundColor Yellow
        & $exe
    }
    $runExitCode = $LASTEXITCODE
    Write-Host "[INFO] wafer_infer exit code: $runExitCode"
    if ($runExitCode -ne 0) {
        Write-Host '[ERROR] Runtime failed. Check TensorRT/OpenCV/CUDA DLL paths and engine file.' -ForegroundColor Red
        exit $runExitCode
    }
} finally {
    Pop-Location
}
