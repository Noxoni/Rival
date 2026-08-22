param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $TrainingRoot
$EnvironmentRoot = Join-Path $TrainingRoot ".venv"

if (-not $PythonExecutable) {
    $ProductionPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $ProductionPython) {
        $PythonExecutable = $ProductionPython
    } else {
        $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
    }
}

$Version = & $PythonExecutable -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($Version -ne "3.12") {
    throw "Rival training requires Python 3.12; selected interpreter reports $Version"
}

if (-not (Test-Path -LiteralPath $EnvironmentRoot)) {
    & $PythonExecutable -m venv $EnvironmentRoot
}

$TrainingPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
& $TrainingPython -m pip install --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0
& $TrainingPython -m pip install -r (Join-Path $TrainingRoot "requirements-lock.txt")
& $TrainingPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

Write-Host "Rival training environment is ready at $EnvironmentRoot"
