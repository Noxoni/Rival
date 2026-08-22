param(
    [string]$SnapshotRoot = (Join-Path $PSScriptRoot "..\..\..\.local_reference_sources\v1")
)

$ErrorActionPreference = "Stop"
$SnapshotRoot = [System.IO.Path]::GetFullPath($SnapshotRoot)
$ManifestPath = Join-Path $SnapshotRoot "MANIFEST.json"

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$failures = New-Object System.Collections.Generic.List[string]

foreach ($botName in @("wisp", "nexto")) {
    $botRoot = Join-Path $SnapshotRoot $botName
    $entries = $manifest.snapshots.$botName

    foreach ($entry in $entries) {
        $path = Join-Path $botRoot $entry.path
        if (-not (Test-Path -LiteralPath $path)) {
            $failures.Add("$botName missing: $($entry.path)")
            continue
        }

        $file = Get-Item -LiteralPath $path
        if ($file.Length -ne [long]$entry.size) {
            $failures.Add("$botName size mismatch: $($entry.path)")
            continue
        }

        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne [string]$entry.sha256) {
            $failures.Add("$botName hash mismatch: $($entry.path)")
        }
    }
}

if ($failures.Count -gt 0) {
    Write-Host "Reference verification FAILED:"
    $failures | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Write-Host "Reference verification PASSED for $SnapshotRoot"
