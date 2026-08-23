param()

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $TrainingRoot "requirements-rlviser.txt"
$BinaryDirectory = Join-Path $TrainingRoot "tools\rlviser"
$Binary = Join-Path $BinaryDirectory "rlviser.exe"
$BinaryDownload = "https://github.com/VirxEC/rlviser/releases/download/v0.8.2/rlviser.exe"
$ExpectedBinarySha256 = "518a04f711c68de81008a51cb90a61808847d11a0ce8a102a87017cf6f94f8ad"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Install the isolated training environment first: ./training/install_training_env.ps1"
}

& $Python -m pip install --no-deps --requirement $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Optional RLViser installation failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force -Path $BinaryDirectory | Out-Null
$NeedsDownload = -not (Test-Path -LiteralPath $Binary -PathType Leaf)
if (-not $NeedsDownload) {
    $ActualHash = (Get-FileHash -LiteralPath $Binary -Algorithm SHA256).Hash.ToLowerInvariant()
    $NeedsDownload = $ActualHash -ne $ExpectedBinarySha256
}
if ($NeedsDownload) {
    $TemporaryBinary = Join-Path $BinaryDirectory "rlviser.exe.download"
    Invoke-WebRequest -Uri $BinaryDownload -OutFile $TemporaryBinary
    $DownloadedHash = (Get-FileHash -LiteralPath $TemporaryBinary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $ExpectedBinarySha256) {
        Remove-Item -LiteralPath $TemporaryBinary -Force
        throw "Downloaded RLViser v0.8.2 hash mismatch: $DownloadedHash"
    }
    Move-Item -LiteralPath $TemporaryBinary -Destination $Binary -Force
}

& $Python -c "import numpy, rlviser_py; from rlgym.rocket_league.rlviser import RLViserRenderer; assert numpy.__version__ == '1.26.4'; assert rlviser_py.__version__ == '0.6.13'; print('RLViser spectator dependency ready; locked headless NumPy remains', numpy.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Optional RLViser compatibility check failed with exit code $LASTEXITCODE"
}

$VerifiedHash = (Get-FileHash -LiteralPath $Binary -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Output "RLViser v0.8.2 executable ready: $Binary ($VerifiedHash)"
