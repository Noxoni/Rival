param(
    [string]$BotRoot = (Join-Path $PSScriptRoot "..\bot")
)

$ErrorActionPreference = "Stop"
$BotRoot = [System.IO.Path]::GetFullPath($BotRoot)

$expected = [ordered]@{
    "models\POLICY.lt" = "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7"
    "models\SHARED_HEAD.lt" = "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42"
}

$failures = New-Object System.Collections.Generic.List[string]
foreach ($entry in $expected.GetEnumerator()) {
    $path = Join-Path $BotRoot $entry.Key
    if (-not (Test-Path -LiteralPath $path)) {
        $failures.Add("missing: $($entry.Key)")
        continue
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) {
        $failures.Add("hash mismatch: $($entry.Key) expected=$($entry.Value) actual=$actual")
    }
}

if ($failures.Count -gt 0) {
    Write-Host "Wisp model verification FAILED:"
    $failures | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Write-Host "Wisp model verification PASSED for $BotRoot"
