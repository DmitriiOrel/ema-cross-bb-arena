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
    [switch]$SkipSandboxRun,
    [switch]$SkipReportOpen
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

Write-Host "[1/2] Running backtest + EMA sweep. This may take several minutes for 2 years of data..."
Invoke-External -Executable $pythonExe -Arguments @(
    "-u",
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

if (-not $SkipReportOpen) {
    $plotPath = Join-Path $PSScriptRoot "reports\scalpel_backtest_plot.png"
    if (Test-Path $plotPath) {
        Write-Host "[report] Opening backtest chart..."
        Start-Process $plotPath | Out-Null
    }
    else {
        Write-Host "[report] Chart file was not found: $plotPath"
    }
}

if (-not $SkipSandboxRun) {
    Write-Host "[2/2] Starting sandbox bot (continuous run). Press Ctrl+C to stop."
    & ".\run_sandbox.ps1"
}
