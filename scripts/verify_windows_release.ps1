[CmdletBinding()]
param(
    [string]$ArchivePath = (Join-Path $PSScriptRoot "..\dist\Rival-Dev-Windows-x64.zip"),
    [switch]$KeepExtracted
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ArchivePath = [System.IO.Path]::GetFullPath($ArchivePath)
$verificationParent = Join-Path $repositoryRoot "dist"
$verificationRoot = Join-Path $verificationParent "verify-Rival-Dev-Windows-x64"

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

if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
    throw "Release archive not found: $ArchivePath"
}

Assert-SafeChildPath -Path $verificationRoot -Parent $verificationParent
if (Test-Path -LiteralPath $verificationRoot) {
    Remove-Item -LiteralPath $verificationRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $verificationRoot | Out-Null

try {
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $verificationRoot

    $requiredPaths = @(
        "RivalDev.exe",
        "_internal",
        "rival.bot.toml",
        "loadout.toml",
        "README.md",
        "BUILD_INFO.json",
        "MANIFEST.sha256",
        "third_party\wisp\LICENSE",
        "third_party\wisp\NOTICE.md",
        "third_party\wisp\UPSTREAM_README.md"
    )
    foreach ($relativePath in $requiredPaths) {
        if (-not (Test-Path -LiteralPath (Join-Path $verificationRoot $relativePath))) {
            throw "Release is missing required path: $relativePath"
        }
    }

    $releaseToml = Get-Content -LiteralPath (Join-Path $verificationRoot "rival.bot.toml") -Raw
    if ($releaseToml -notmatch '(?m)^run_command\s*=\s*"RivalDev\.exe"\s*$') {
        throw "Release TOML does not launch RivalDev.exe"
    }
    if ($releaseToml -match '\.venv|run_command_linux') {
        throw "Release TOML still contains a development-only launch command"
    }
    if ($releaseToml -notmatch 'noxoni/rival/dev-v1') {
        throw "Release TOML does not contain the expected Rival agent id"
    }

    $manifestPath = Join-Path $verificationRoot "MANIFEST.sha256"
    $manifestEntries = @{}
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
            throw "Malformed manifest line: $line"
        }
        $manifestEntries[$Matches[2]] = $Matches[1]
    }

    $actualFiles = Get-ChildItem -LiteralPath $verificationRoot -File -Recurse | Where-Object {
        $_.FullName -ne $manifestPath
    }
    foreach ($file in $actualFiles) {
        $relative = $file.FullName.Substring($verificationRoot.Length + 1).Replace("\", "/")
        if (-not $manifestEntries.ContainsKey($relative)) {
            throw "File is not charged to MANIFEST.sha256: $relative"
        }
        $actualHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $manifestEntries[$relative]) {
            throw "SHA-256 mismatch for $relative"
        }
    }
    if ($actualFiles.Count -ne $manifestEntries.Count) {
        throw "Manifest/file count mismatch: manifest=$($manifestEntries.Count) files=$($actualFiles.Count)"
    }

    $selfTestOutput = & (Join-Path $verificationRoot "RivalDev.exe") --self-test 2>&1
    $selfTestExitCode = $LASTEXITCODE
    $selfTestOutput | ForEach-Object { Write-Host $_ }
    if ($selfTestExitCode -ne 0) {
        throw "Packaged self-test failed with exit code $selfTestExitCode"
    }
    if (($selfTestOutput -join "`n") -notmatch '"status"\s*:\s*"pass"') {
        throw "Packaged self-test did not report pass"
    }
    if (($selfTestOutput -join "`n") -notmatch '"frozen"\s*:\s*true') {
        throw "Packaged self-test did not run from a frozen executable"
    }

    $buildInfo = Get-Content -LiteralPath (Join-Path $verificationRoot "BUILD_INFO.json") -Raw | ConvertFrom-Json
    $archiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    [pscustomobject]@{
        status = "pass"
        archive = $ArchivePath
        archive_bytes = (Get-Item -LiteralPath $ArchivePath).Length
        archive_sha256 = $archiveHash
        manifest_files = $manifestEntries.Count
        source_commit = $buildInfo.source_commit
        source_had_tracked_changes = $buildInfo.source_had_tracked_changes
        python = $buildInfo.python
        pyinstaller = $buildInfo.pyinstaller
    } | ConvertTo-Json -Depth 4
}
finally {
    if (-not $KeepExtracted -and (Test-Path -LiteralPath $verificationRoot)) {
        Remove-Item -LiteralPath $verificationRoot -Recurse -Force
    }
}
