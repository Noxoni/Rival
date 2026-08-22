param(
    [switch]$IncludeMeasuredRuns
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $TrainingRoot
$TrainingPython = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$ProductionPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$PreviousPythonPath = $env:PYTHONPATH

if (-not (Test-Path -LiteralPath $TrainingPython)) {
    throw "Missing training environment. Run training/install_training_env.ps1 first."
}

Push-Location $RepositoryRoot
try {
    $env:PYTHONPATH = $TrainingRoot
    New-Item -ItemType Directory -Force training/.pytest_tmp | Out-Null
    & $ProductionPython training/scripts/verify_action_prefix.py
    & $TrainingPython training/scripts/verify_environment.py
    & $TrainingPython training/scripts/verify_bootstrap.py
    & $TrainingPython training/scripts/stress_env.py --decisions 5000
    & $TrainingPython -m pytest training/tests -q --basetemp=training/.pytest_tmp/verification
    & $TrainingPython -m ruff check training
    & $TrainingPython -m compileall -q training/rival_training training/scripts training/tests

    if ($IncludeMeasuredRuns) {
        & $TrainingPython training/scripts/benchmark_throughput.py
        & $TrainingPython training/scripts/run_ppo_smoke.py
        & $TrainingPython training/scripts/smoke_deployment.py
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
