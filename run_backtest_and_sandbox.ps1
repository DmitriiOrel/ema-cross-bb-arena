param(
    [int]$DaysBack = 730,
    [int]$FastStart = 10,
    [int]$FastEnd = 40,
    [int]$FastStep = 5,
    [int]$SlowStart = 45,
    [int]$SlowEnd = 120,
    [int]$SlowStep = 5,
    [int]$Backcandles = 15,
    [double]$InitialCapital = 100000,
    [switch]$SkipSandboxRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$StepName
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed (exit code $LASTEXITCODE)."
    }
}

function Clear-NetworkEnv {
    $vars = @(
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY",
        "PIP_NO_INDEX", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"
    )
    foreach ($name in $vars) {
        Remove-Item -Path ("Env:{0}" -f $name) -ErrorAction SilentlyContinue
    }
}

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found. Run .\quickstart.ps1 -Token <YOUR_TOKEN> first."
}
if (-not (Test-Path ".env")) {
    throw ".env not found. Run .\quickstart.ps1 -Token <YOUR_TOKEN> first."
}

$env:PYTHONPATH = "."
$env:PYTHONDONTWRITEBYTECODE = "1"
Clear-NetworkEnv

Invoke-External -Executable $pythonExe -Arguments @(
    ".\tools\optimize_scalpel_report.py",
    "--days-back", "$DaysBack",
    "--fast-start", "$FastStart",
    "--fast-end", "$FastEnd",
    "--fast-step", "$FastStep",
    "--slow-start", "$SlowStart",
    "--slow-end", "$SlowEnd",
    "--slow-step", "$SlowStep",
    "--backcandles", "$Backcandles",
    "--initial-capital", "$InitialCapital",
    "--write-live-config"
) -StepName "Run EMA sweep backtest and build report"

if (-not $SkipSandboxRun) {
    & ".\run_sandbox.ps1"
}
