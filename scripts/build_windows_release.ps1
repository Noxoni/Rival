[CmdletBinding()]
param(
    [string]$PythonPath = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$OutputRoot = (Join-Path $PSScriptRoot "..\dist"),
    [switch]$InstallBuildDependencies
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$PythonPath = [System.IO.Path]::GetFullPath($PythonPath)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$buildRoot = Join-Path $repositoryRoot "build\pyinstaller"
$pyinstallerDist = Join-Path $buildRoot "dist"
$pyinstallerWork = Join-Path $buildRoot "work"
$releaseRoot = Join-Path $OutputRoot "Rival-Dev-Windows-x64"
$archivePath = Join-Path $OutputRoot "Rival-Dev-Windows-x64.zip"
$archiveHashPath = "$archivePath.sha256"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Assert-SafeChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd("\", "/")
    $fullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd("\", "/")
    $prefix = $fullParent + [System.IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing destructive operation outside '$fullParent': $fullPath"
    }
    if ($fullPath -eq $fullParent) {
        throw "Refusing destructive operation against parent directory: $fullPath"
    }
}

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python runtime not found: $PythonPath. Create the repository .venv with CPython 3.12 first."
}

$pythonVersion = (& $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not query Python version from $PythonPath"
}
if (-not $pythonVersion.StartsWith("3.12.")) {
    throw "The tested Wisp build requires CPython 3.12; found $pythonVersion at $PythonPath"
}

if ($InstallBuildDependencies) {
    Invoke-CheckedNative $PythonPath -m pip install -r (Join-Path $repositoryRoot "requirements-build.txt")
}

$pyinstallerVersion = (& $PythonPath -c "import PyInstaller; print(PyInstaller.__version__)" 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or -not $pyinstallerVersion) {
    throw "PyInstaller is unavailable. Re-run with -InstallBuildDependencies or install requirements-build.txt."
}
if ($pyinstallerVersion -ne "6.22.2") {
    throw "Expected tested PyInstaller 6.22.2, found $pyinstallerVersion"
}

Assert-SafeChildPath -Path $buildRoot -Parent (Join-Path $repositoryRoot "build")
Assert-SafeChildPath -Path $releaseRoot -Parent $OutputRoot

foreach ($directory in @($buildRoot, $releaseRoot)) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -LiteralPath $directory -Recurse -Force
    }
}
foreach ($file in @($archivePath, $archiveHashPath)) {
    if (Test-Path -LiteralPath $file) {
        Remove-Item -LiteralPath $file -Force
    }
}

New-Item -ItemType Directory -Force -Path $buildRoot, $pyinstallerDist, $pyinstallerWork, $OutputRoot | Out-Null

Push-Location $repositoryRoot
try {
    Invoke-CheckedNative $PythonPath -m PyInstaller --noconfirm --clean --distpath $pyinstallerDist --workpath $pyinstallerWork (Join-Path $repositoryRoot "packaging\rival.spec")
}
finally {
    Pop-Location
}

$builtBundle = Join-Path $pyinstallerDist "RivalDev"
$builtExecutable = Join-Path $builtBundle "RivalDev.exe"
if (-not (Test-Path -LiteralPath $builtExecutable -PathType Leaf)) {
    throw "PyInstaller did not produce the expected executable: $builtExecutable"
}

New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
Get-ChildItem -LiteralPath $builtBundle -Force | Copy-Item -Destination $releaseRoot -Recurse -Force

Copy-Item -LiteralPath (Join-Path $repositoryRoot "bot\loadout.toml") -Destination (Join-Path $releaseRoot "loadout.toml")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "packaging\RELEASE_README.md") -Destination (Join-Path $releaseRoot "README.md")
New-Item -ItemType Directory -Force -Path (Join-Path $releaseRoot "third_party") | Out-Null
Copy-Item -LiteralPath (Join-Path $repositoryRoot "third_party\wisp") -Destination (Join-Path $releaseRoot "third_party\wisp") -Recurse

$releaseToml = Get-Content -LiteralPath (Join-Path $repositoryRoot "bot\rival.bot.toml") -Raw
$releaseToml = $releaseToml -replace "(?m)^run_command\s*=.*$", 'run_command = "RivalDev.exe"'
$releaseToml = $releaseToml -replace "(?m)^run_command_linux\s*=.*(?:\r?\n)?", ""
[System.IO.File]::WriteAllText((Join-Path $releaseRoot "rival.bot.toml"), $releaseToml, $utf8NoBom)

$gitCommit = (& git -C $repositoryRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not resolve repository commit"
}
$trackedChanges = @(& git -C $repositoryRoot status --porcelain --untracked-files=no)
$modelHashes = [ordered]@{
    "POLICY.lt" = (Get-FileHash -LiteralPath (Join-Path $repositoryRoot "bot\models\POLICY.lt") -Algorithm SHA256).Hash.ToLowerInvariant()
    "SHARED_HEAD.lt" = (Get-FileHash -LiteralPath (Join-Path $repositoryRoot "bot\models\SHARED_HEAD.lt") -Algorithm SHA256).Hash.ToLowerInvariant()
}
$buildInfo = [ordered]@{
    format_version = 1
    product = "Rival Dev"
    platform = "windows-x64"
    agent_id = "noxoni/rival/dev-v1"
    source_repository = "https://github.com/Noxoni/Rival"
    source_commit = $gitCommit
    source_had_tracked_changes = ($trackedChanges.Count -gt 0)
    built_utc = [DateTime]::UtcNow.ToString("o")
    python = $pythonVersion
    pyinstaller = $pyinstallerVersion
    models = $modelHashes
}
[System.IO.File]::WriteAllText(
    (Join-Path $releaseRoot "BUILD_INFO.json"),
    (($buildInfo | ConvertTo-Json -Depth 5) + "`n"),
    $utf8NoBom
)

Invoke-CheckedNative (Join-Path $releaseRoot "RivalDev.exe") --self-test

$manifestPath = Join-Path $releaseRoot "MANIFEST.sha256"
$manifestLines = Get-ChildItem -LiteralPath $releaseRoot -File -Recurse | Where-Object {
    $_.FullName -ne $manifestPath
} | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($releaseRoot.Length + 1).Replace("\", "/")
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $relative"
}
[System.IO.File]::WriteAllLines($manifestPath, $manifestLines, $utf8NoBom)

Compress-Archive -Path (Join-Path $releaseRoot "*") -DestinationPath $archivePath -CompressionLevel Optimal
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText($archiveHashPath, "$archiveHash  $([System.IO.Path]::GetFileName($archivePath))`n", $utf8NoBom)

[pscustomobject]@{
    status = "pass"
    release_root = $releaseRoot
    archive = $archivePath
    archive_bytes = (Get-Item -LiteralPath $archivePath).Length
    archive_sha256 = $archiveHash
    source_commit = $gitCommit
    source_had_tracked_changes = ($trackedChanges.Count -gt 0)
    python = $pythonVersion
    pyinstaller = $pyinstallerVersion
} | ConvertTo-Json -Depth 4
