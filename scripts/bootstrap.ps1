param([string]$Python)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $Python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $Python = $command.Source }
    else { $Python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' }
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python not found. Pass -Python with a Python 3.12 executable.' }
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
$venv = Join-Path $repo '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
$runner = Join-Path $venv 'Scripts\python.exe'
& $runner -m pip install --only-binary=:all: -r (Join-Path $repo 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $runner -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
$env:PYTHONUTF8 = '1'
& $runner (Join-Path $PSScriptRoot 'asset_job.py') self-test
if ($LASTEXITCODE -ne 0) { throw 'Self-test failed' }
& $runner (Join-Path $PSScriptRoot 'test_workflow.py')
if ($LASTEXITCODE -ne 0) { throw 'Workflow test failed' }
Write-Output "READY: $runner"
