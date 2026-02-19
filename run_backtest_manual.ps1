param(
    [Parameter(Position = 0)]
    [int]$EmaFast,
    [Parameter(Position = 1)]
    [int]$EmaSlow,
    [Parameter(Position = 2)]
    [int]$BbWindow,
    [Parameter(Position = 3)]
    [double]$BbDev,
    [Parameter(Position = 4)]
    [int]$TimeframeMin,
    [string]$Name,
    [string]$GitHubOwner = "DmitriiOrel",
    [string]$GitHubRepo = "winter_school_project",
    [string]$GitHubPath = "reports/leaderboard.json",
    [string]$GitHubToken = "",
    [switch]$NoChartOpen
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

function Prompt-IfMissing {
    param(
        [string]$Value,
        [string]$Prompt
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return (Read-Host $Prompt)
    }
    return $Value
}

if (-not $PSBoundParameters.ContainsKey("EmaFast")) { $EmaFast = [int](Read-Host "ema_fast (8..30)") }
if (-not $PSBoundParameters.ContainsKey("EmaSlow")) { $EmaSlow = [int](Read-Host "ema_slow (35..120)") }
if (-not $PSBoundParameters.ContainsKey("BbWindow")) { $BbWindow = [int](Read-Host "bb_window (10..40)") }
if (-not $PSBoundParameters.ContainsKey("BbDev")) { $BbDev = [double](Read-Host "bb_dev (1.0..3.5, step 0.25)") }
if (-not $PSBoundParameters.ContainsKey("TimeframeMin")) { $TimeframeMin = [int](Read-Host "timeframe_min (5,15,30,60,120,240,720,1440)") }
$Name = Prompt-IfMissing -Value $Name -Prompt "participant name for leaderboard"

if ($EmaFast -lt 8 -or $EmaFast -gt 30) { throw "ema_fast must be in range 8..30" }
if ($EmaSlow -lt 35 -or $EmaSlow -gt 120) { throw "ema_slow must be in range 35..120" }
if ($EmaFast -ge $EmaSlow) { throw "Constraint failed: ema_fast < ema_slow" }
if ($BbWindow -lt 10 -or $BbWindow -gt 40) { throw "bb_window must be in range 10..40" }
if ($BbDev -lt 1.0 -or $BbDev -gt 3.5) { throw "bb_dev must be in range 1.0..3.5" }
$scaledDev = [int][Math]::Round($BbDev * 100)
if (($scaledDev % 25) -ne 0) { throw "bb_dev step must be 0.25" }
$allowedTf = @(5, 15, 30, 60, 120, 240, 720, 1440)
if ($TimeframeMin -notin $allowedTf) { throw "timeframe_min must be one of: 5, 15, 30, 60, 120, 240, 720, 1440" }

if ([string]::IsNullOrWhiteSpace($GitHubToken) -and -not [string]::IsNullOrWhiteSpace($env:GITHUB_TOKEN)) {
    $GitHubToken = $env:GITHUB_TOKEN
}
if ([string]::IsNullOrWhiteSpace($GitHubToken)) {
    throw "GitHub PAT is required. Pass -GitHubToken <PAT> or set GITHUB_TOKEN."
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

$argsList = @(
    "-u",
    ".\tools\manual_backtest_leaderboard.py",
    "--name", "$Name",
    "--ema-fast", "$EmaFast",
    "--ema-slow", "$EmaSlow",
    "--bb-window", "$BbWindow",
    "--bb-dev", "$BbDev",
    "--timeframe-min", "$TimeframeMin",
    "--days-back", "1095",
    "--write-live-config",
    "--require-github",
    "--github-owner", "$GitHubOwner",
    "--github-repo", "$GitHubRepo",
    "--github-path", "$GitHubPath",
    "--github-token", "$GitHubToken"
)

Write-Host "Running backtest (3 years) and updating GitHub leaderboard..."
Invoke-External -Executable $pythonExe -Arguments $argsList -StepName "Manual backtest + leaderboard update"

if (-not $NoChartOpen) {
    $plotPath = Join-Path $PSScriptRoot "reports\scalpel_backtest_plot.png"
    if (Test-Path $plotPath) {
        Start-Process $plotPath | Out-Null
    }
}

Write-Host "Starting sandbox bot with selected parameters. Stop: Ctrl+C."
Invoke-External -Executable $pythonExe -Arguments @(".\app\main.py") -StepName "Sandbox bot run"
