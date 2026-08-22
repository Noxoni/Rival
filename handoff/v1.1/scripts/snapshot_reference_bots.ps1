param(
    [string]$BotpackRoot = "C:\Users\patri\AppData\Local\RLBot5\bots",
    [string]$DestinationRoot = (Join-Path $PSScriptRoot "..\..\..\.local_reference_sources"),
    [string]$Version = "v1"
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

$BotpackRoot = Resolve-FullPath $BotpackRoot
$DestinationRoot = Resolve-FullPath $DestinationRoot
$VersionRoot = Join-Path $DestinationRoot $Version

if (-not (Test-Path -LiteralPath $BotpackRoot)) {
    throw "BotPack root does not exist: $BotpackRoot"
}

if (Test-Path -LiteralPath $VersionRoot) {
    throw "Reference snapshot $Version already exists at $VersionRoot. Refusing to overwrite. Use -Version v2 (or later)."
}

New-Item -ItemType Directory -Path $VersionRoot -Force | Out-Null

Write-Host "Scanning RLBot bot directory: $BotpackRoot"

# Find likely bot configuration files first.
$configFiles = Get-ChildItem -LiteralPath $BotpackRoot -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -eq "bot.toml" -or
        $_.Name -like "*.bot.toml" -or
        $_.Name -eq "bot.cfg"
    }

$wispCandidates = @()
$nextoCandidates = @()

foreach ($cfg in $configFiles) {
    $content = ""
    try {
        $content = Get-Content -LiteralPath $cfg.FullName -Raw -ErrorAction Stop
    } catch {
        continue
    }

    if (
        $content -match 'eastvillage/wisp/v2-75B' -or
        $content -match '(?i)name\s*=\s*["'']?Wisp v2-75B' -or
        ($content -match '(?i)\bWisp\b' -and $cfg.DirectoryName -match '(?i)wisp')
    ) {
        $wispCandidates += $cfg
    }

    if (
        $content -match '(?i)agent_id\s*=\s*["''][^"'']*nexto' -or
        $content -match '(?i)name\s*=\s*["'']?Nexto\b' -or
        ($content -match '(?i)\bNexto\b' -and $cfg.DirectoryName -match '(?i)nexto')
    ) {
        $nextoCandidates += $cfg
    }
}

# Fallbacks for packaged layouts where metadata may not be easy to parse.
if ($wispCandidates.Count -eq 0) {
    $wispFiles = Get-ChildItem -LiteralPath $BotpackRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '(?i)wisp' }
    if ($wispFiles.Count -gt 0) {
        $wispCandidates += $wispFiles[0]
    }
}

if ($nextoCandidates.Count -eq 0) {
    $nextoModel = Get-ChildItem -LiteralPath $BotpackRoot -Recurse -File -Filter "nexto-model.pt" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $nextoModel) {
        $nextoCandidates += $nextoModel
    } else {
        $nextoFiles = Get-ChildItem -LiteralPath $BotpackRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '(?i)nexto' }
        if ($nextoFiles.Count -gt 0) {
            $nextoCandidates += $nextoFiles[0]
        }
    }
}

if ($wispCandidates.Count -eq 0) {
    throw "Could not locate Wisp v2-75B under $BotpackRoot. Inspect the installed layout manually and rerun with an adjusted script."
}
if ($nextoCandidates.Count -eq 0) {
    throw "Could not locate Nexto under $BotpackRoot. Inspect the installed layout manually and rerun with an adjusted script."
}

function Select-SourceRoot($Candidates, [string]$BotName) {
    # Prefer a config file's directory. If multiple configs exist, show what was found
    # and select the shortest path as the likely packaged bot root.
    $dirs = $Candidates |
        ForEach-Object { $_.Directory.FullName } |
        Sort-Object -Unique

    Write-Host "$BotName candidates:"
    $dirs | ForEach-Object { Write-Host "  $_" }

    return $dirs | Sort-Object { $_.Length } | Select-Object -First 1
}

$wispRoot = Select-SourceRoot $wispCandidates "Wisp"
$nextoRoot = Select-SourceRoot $nextoCandidates "Nexto"

# If a selected root is a nested src/nexto folder but its immediately adjacent parent
# clearly contains the dependency/build context, preserve the smallest self-contained
# tree Codex can reason about. We intentionally do not walk arbitrarily high.
function Expand-UsefulRoot([string]$Root, [string]$BotName) {
    $parent = Split-Path $Root -Parent
    if ([string]::IsNullOrWhiteSpace($parent)) { return $Root }

    if ($BotName -eq "Wisp") {
        if (
            (Test-Path (Join-Path $parent "requirements.txt")) -or
            (Test-Path (Join-Path $parent "bob.toml"))
        ) {
            return $parent
        }
    }

    if ($BotName -eq "Nexto") {
        if (
            (Test-Path (Join-Path $parent "requirements.txt")) -and
            ((Test-Path (Join-Path $parent "nexto")) -or (Test-Path (Join-Path $parent "bob.toml")))
        ) {
            return $parent
        }
    }

    return $Root
}

$wispRoot = Expand-UsefulRoot $wispRoot "Wisp"
$nextoRoot = Expand-UsefulRoot $nextoRoot "Nexto"

Write-Host "Selected Wisp source root: $wispRoot"
Write-Host "Selected Nexto source root: $nextoRoot"

$excludeDirs = @(".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache")
$excludeFiles = @("*.pyc", "*.pyo")

function Copy-Tree([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    # robocopy gives us a reliable recursive copy on Windows and supports directory exclusions.
    $args = @(
        $Source,
        $Destination,
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    )

    if ($excludeDirs.Count -gt 0) {
        $args += "/XD"
        $args += $excludeDirs
    }
    if ($excludeFiles.Count -gt 0) {
        $args += "/XF"
        $args += $excludeFiles
    }

    & robocopy @args
    $code = $LASTEXITCODE
    if ($code -gt 7) {
        throw "robocopy failed copying '$Source' to '$Destination' with exit code $code"
    }
}

$wispDest = Join-Path $VersionRoot "wisp"
$nextoDest = Join-Path $VersionRoot "nexto"

Copy-Tree $wispRoot $wispDest
Copy-Tree $nextoRoot $nextoDest

function Get-TreeManifest([string]$BasePath) {
    $base = Resolve-FullPath $BasePath
    $items = Get-ChildItem -LiteralPath $base -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
            [PSCustomObject]@{
                path = $_.FullName.Substring($base.Length).TrimStart('\','/')
                size = $_.Length
                sha256 = $hash.Hash.ToLowerInvariant()
            }
        }
    return @($items)
}

$manifest = [ordered]@{
    version = $Version
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_botpack_root = $BotpackRoot
    selected_sources = [ordered]@{
        wisp = $wispRoot
        nexto = $nextoRoot
    }
    snapshots = [ordered]@{
        wisp = Get-TreeManifest $wispDest
        nexto = Get-TreeManifest $nextoDest
    }
}

$manifestPath = Join-Path $VersionRoot "MANIFEST.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "Reference snapshot created:"
Write-Host "  $VersionRoot"
Write-Host "Manifest:"
Write-Host "  $manifestPath"
Write-Host ""
Write-Host "Installed BotPack was not modified."
