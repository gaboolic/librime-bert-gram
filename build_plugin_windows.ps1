param(
    [string] $LibrimeRoot = "",
    [string] $BuildDir = "",
    [string] $Config = "Release",
    [string] $OnnxRuntimeRoot = "",
    [string] $MSBuildPath = "",
    [switch] $Clean
)

$ErrorActionPreference = "Stop"
$IsWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT

function Ensure-Directory {
    param([string] $Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Resolve-FullPath {
    param([string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    return [System.IO.Path]::GetFullPath($Path)
}

function Invoke-Step {
    param(
        [string] $Title,
        [scriptblock] $Action
    )

    Write-Host ""
    Write-Host "============================================"
    Write-Host $Title
    Write-Host "============================================"
    & $Action
}

function Invoke-External {
    param(
        [string] $FilePath,
        [string[]] $ArgumentList,
        [switch] $IgnoreExitCode
    )

    Write-Host ">> $FilePath $($ArgumentList -join ' ')"
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if (-not $IgnoreExitCode -and $exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath"
    }
    return $exitCode
}

function Test-LibrimeRoot {
    param([string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    return (Test-Path -LiteralPath (Join-Path $Path "CMakeLists.txt"))
}

function Resolve-LibrimeRoot {
    param(
        [string] $ExplicitPath,
        [string] $PluginRoot
    )

    $candidates = @(
        $ExplicitPath,
        [Environment]::GetEnvironmentVariable("LIBRIME_ROOT"),
        (Join-Path (Split-Path $PluginRoot -Parent) "librime")
    )
    foreach ($candidate in $candidates) {
        $resolved = Resolve-FullPath $candidate
        if (Test-LibrimeRoot $resolved) {
            return $resolved
        }
    }
    return ""
}

function Get-BoostRoot {
    param([string] $LibrimeRoot)
    foreach ($candidate in @(
        (Join-Path $LibrimeRoot "deps\boost-1.89.0"),
        (Join-Path $LibrimeRoot "deps\boost")
    )) {
        if (Test-Path -LiteralPath (Join-Path $candidate "boost")) {
            return $candidate
        }
    }
    return ""
}

function Test-OnnxRuntimeSdkRoot {
    param([string] $Root)
    if ([string]::IsNullOrWhiteSpace($Root)) {
        return $false
    }
    return (
        (Test-Path -LiteralPath (Join-Path $Root "include\onnxruntime_cxx_api.h")) -and
        (Test-Path -LiteralPath (Join-Path $Root "lib\onnxruntime.lib"))
    )
}

function Get-OnnxRuntimeNuGetRoot {
    param([string] $Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return ""
    }

    $resolved = Resolve-FullPath $Candidate
    foreach ($root in @(
        $resolved,
        (Join-Path $resolved "pkg")
    )) {
        if (
            (Test-Path -LiteralPath (Join-Path $root "build\native\include\onnxruntime_cxx_api.h")) -and
            (Test-Path -LiteralPath (Join-Path $root "runtimes\win-x64\native\onnxruntime.lib"))
        ) {
            return $root
        }
    }
    return ""
}

function Prepare-OnnxRuntimeSdkRoot {
    param(
        [string] $SourceRoot,
        [string] $PluginRoot
    )

    $sdkRoot = Join-Path $PluginRoot ".deps\onnxruntime\win-x64"
    $includeDir = Join-Path $sdkRoot "include"
    $libDir = Join-Path $sdkRoot "lib"
    $binDir = Join-Path $sdkRoot "bin"

    Ensure-Directory -Path $includeDir
    Ensure-Directory -Path $libDir
    Ensure-Directory -Path $binDir

    Copy-Item -Path (Join-Path $SourceRoot "build\native\include\*") -Destination $includeDir -Recurse -Force
    foreach ($name in @(
        "onnxruntime.lib",
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_shared.lib"
    )) {
        $source = Join-Path $SourceRoot "runtimes\win-x64\native\$name"
        if (-not (Test-Path -LiteralPath $source)) {
            continue
        }
        if ($name -like "*.lib") {
            Copy-Item -LiteralPath $source -Destination (Join-Path $libDir $name) -Force
        }
        else {
            Copy-Item -LiteralPath $source -Destination (Join-Path $binDir $name) -Force
        }
    }

    if (-not (Test-OnnxRuntimeSdkRoot $sdkRoot)) {
        throw "Prepared ONNX Runtime SDK layout is incomplete: `"$sdkRoot`""
    }
    return $sdkRoot
}

function Resolve-OnnxRuntimeRoot {
    param(
        [string] $ExplicitPath,
        [string] $PluginRoot
    )

    $candidates = @(
        $ExplicitPath,
        [Environment]::GetEnvironmentVariable("ONNXRUNTIME_ROOT_DIR"),
        (Join-Path $PluginRoot ".deps\onnxruntime\win-x64"),
        (Join-Path $PluginRoot ".deps\onnxruntime\current")
    )

    foreach ($candidate in $candidates) {
        $resolved = Resolve-FullPath $candidate
        if (Test-OnnxRuntimeSdkRoot $resolved) {
            return $resolved
        }

        $nugetRoot = Get-OnnxRuntimeNuGetRoot $resolved
        if ($nugetRoot) {
            return (Prepare-OnnxRuntimeSdkRoot -SourceRoot $nugetRoot -PluginRoot $PluginRoot)
        }
    }

    return ""
}

function Get-VsWherePath {
    $path = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $path) {
        return $path
    }
    throw "vswhere.exe was not found. Install Visual Studio Build Tools."
}

function Resolve-MSBuildPath {
    param([string] $ExplicitPath)

    $resolved = Resolve-FullPath $ExplicitPath
    if ($resolved -and (Test-Path -LiteralPath $resolved)) {
        return $resolved
    }

    $vswhere = Get-VsWherePath
    $found = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($found)) {
        throw "MSBuild.exe was not found via vswhere."
    }
    return (Resolve-FullPath $found)
}

function Ensure-RimeImportLibrary {
    param(
        [string] $MSBuildExe,
        [string] $BuildDir,
        [string] $Config
    )

    $rimeImportLib = Join-Path $BuildDir ("src\{0}\rime.lib" -f $Config)
    if (Test-Path -LiteralPath $rimeImportLib) {
        return $rimeImportLib
    }

    $rimeProject = Join-Path $BuildDir "src\rime.vcxproj"
    if (-not (Test-Path -LiteralPath $rimeProject)) {
        throw "Missing generated rime project: `"$rimeProject`""
    }

    Write-Warning "rime.lib is missing; building librime import library first."
    $null = Invoke-External -FilePath $MSBuildExe -ArgumentList @(
        $rimeProject,
        "/p:Configuration=$Config",
        "/p:BuildProjectReferences=false"
    ) -IgnoreExitCode

    if (-not (Test-Path -LiteralPath $rimeImportLib)) {
        throw "Failed to produce librime import library: `"$rimeImportLib`""
    }

    return $rimeImportLib
}

function Get-PluginOutputCandidates {
    param(
        [string] $BuildDir,
        [string] $Config
    )

    return @(
        (Join-Path $BuildDir "bin\rime-plugins\rime-bert-grammar.dll"),
        (Join-Path $BuildDir "lib\rime-plugins\rime-bert-grammar.dll"),
        (Join-Path $BuildDir "bin\rime-plugins\$Config\rime-bert-grammar.dll"),
        (Join-Path $BuildDir "lib\rime-plugins\$Config\rime-bert-grammar.dll")
    )
}

$PluginRoot = Resolve-FullPath (Join-Path $PSScriptRoot ".")
$LibrimeRoot = Resolve-LibrimeRoot -ExplicitPath $LibrimeRoot -PluginRoot $PluginRoot
if (-not $LibrimeRoot) {
    throw "Unable to resolve librime root. Pass -LibrimeRoot or set LIBRIME_ROOT."
}
if (-not $BuildDir) {
    $BuildDir = Join-Path $LibrimeRoot "build-bert-grammar"
}

$BuildDir = Resolve-FullPath $BuildDir
$BoostRoot = Get-BoostRoot -LibrimeRoot $LibrimeRoot
$ResolvedOnnxRuntimeRoot = Resolve-OnnxRuntimeRoot -ExplicitPath $OnnxRuntimeRoot -PluginRoot $PluginRoot
$PluginLinkDir = Join-Path $LibrimeRoot "plugins\bert_grammar"
$ResolvedMSBuildPath = ""

Invoke-Step -Title "Step 1: Prepare plugin link" -Action {
    if (Test-Path -LiteralPath $PluginLinkDir) {
        $existing = Get-Item -LiteralPath $PluginLinkDir -Force
        $isLink = ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        if (-not $isLink) {
            throw "Existing path is not a junction/symlink: `"$PluginLinkDir`""
        }
        $targets = @($existing.Target | ForEach-Object { Resolve-FullPath $_ })
        if ($targets -contains $PluginRoot) {
            Write-Host "[INFO] Plugin junction already points at `"$PluginRoot`""
            return
        }
        $null = Invoke-External -FilePath "cmd.exe" -ArgumentList @("/c", "rmdir", $PluginLinkDir)
    }
    New-Item -ItemType Junction -Path $PluginLinkDir -Target $PluginRoot | Out-Null
    Write-Host "[INFO] Linked plugin source: `"$PluginLinkDir`" -> `"$PluginRoot`""
}

Invoke-Step -Title "Step 2: Configure librime build" -Action {
    if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
    Ensure-Directory -Path $BuildDir

    $cmakeArgs = @(
        "-S", $LibrimeRoot,
        "-B", $BuildDir,
        "-DBUILD_SHARED_LIBS=ON",
        "-DENABLE_EXTERNAL_PLUGINS=ON",
        "-DBUILD_MERGED_PLUGINS=OFF",
        "-DBUILD_SEPARATE_LIBS=OFF",
        "-DRIME_ROOT_DIR=$LibrimeRoot",
        "-DRIME_BUILD_DIR_VAR=$BuildDir"
    )
    if ($BoostRoot) {
        $cmakeArgs += "-DBOOST_ROOT=$BoostRoot"
        $cmakeArgs += "-DBoost_INCLUDE_DIR=$BoostRoot"
    }
    if ($ResolvedOnnxRuntimeRoot) {
        $cmakeArgs += "-DONNXRUNTIME_ROOT_DIR=$ResolvedOnnxRuntimeRoot"
        $cmakeArgs += "-DENABLE_ONNXRUNTIME=ON"
    }
    else {
        Write-Warning "ONNX Runtime SDK was not found; plugin will be built without ONNX support."
    }

    $null = Invoke-External -FilePath "cmake" -ArgumentList $cmakeArgs
}

Invoke-Step -Title "Step 3: Build plugin DLL" -Action {
    if ($IsWindowsHost) {
        $ResolvedMSBuildPath = Resolve-MSBuildPath -ExplicitPath $MSBuildPath
        $null = Ensure-RimeImportLibrary -MSBuildExe $ResolvedMSBuildPath -BuildDir $BuildDir -Config $Config

        $pluginProject = Join-Path $BuildDir "plugins\rime-bert-grammar.vcxproj"
        if (-not (Test-Path -LiteralPath $pluginProject)) {
            throw "Missing generated plugin project: `"$pluginProject`""
        }
        $null = Invoke-External -FilePath $ResolvedMSBuildPath -ArgumentList @(
            $pluginProject,
            "/p:Configuration=$Config",
            "/p:BuildProjectReferences=false"
        )
    }
    else {
        $null = Invoke-External -FilePath "cmake" -ArgumentList @(
            "--build", $BuildDir,
            "--config", $Config,
            "--target", "rime-bert-grammar"
        )
    }
}

Invoke-Step -Title "Step 4: Report outputs" -Action {
    $found = Get-PluginOutputCandidates -BuildDir $BuildDir -Config $Config | Where-Object {
        Test-Path -LiteralPath $_
    }
    if (-not $found) {
        throw "Plugin DLL was not found in the expected output directories."
    }

    if ($ResolvedOnnxRuntimeRoot) {
        Write-Host "[INFO] ONNX Runtime SDK root: `"$ResolvedOnnxRuntimeRoot`""
    }
    if ($ResolvedMSBuildPath) {
        Write-Host "[INFO] MSBuild: `"$ResolvedMSBuildPath`""
    }
    $found | ForEach-Object {
        Write-Host "[INFO] Built plugin DLL: `"$($_)`""
    }
}
