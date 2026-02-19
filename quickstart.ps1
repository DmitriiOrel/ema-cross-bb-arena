param(
    [Parameter(Mandatory = $false)]
    [string]$Token,
    [string]$PythonVersion = "3.11",
    [switch]$Run
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

if ([string]::IsNullOrWhiteSpace($Token)) {
    $Token = Read-Host "Enter T-Invest API token (starts with t.)"
}
if ([string]::IsNullOrWhiteSpace($Token)) {
    throw "Token is empty. Run again with -Token."
}
$Token = $Token.Trim()
if ($Token.StartsWith("<") -and $Token.EndsWith(">")) {
    $Token = $Token.Substring(1, $Token.Length - 2).Trim()
}
if ($Token.StartsWith("t.<") -and $Token.EndsWith(">")) {
    $inner = $Token.Substring(3, $Token.Length - 4).Trim()
    if ($inner.StartsWith("t.")) {
        $Token = $inner
    }
    else {
        $Token = "t.$inner"
    }
}
if (-not $Token.StartsWith("t.")) {
    throw "Token format is invalid. Use raw token like t.xxxxx (without < >)."
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found. Install Python 3.11+ first."
}

$pyArgs = @("-$PythonVersion")
try {
    Invoke-External -Executable "py" -Arguments ($pyArgs + @("--version")) -StepName "Validate Python $PythonVersion"
}
catch {
    Write-Host "Python $PythonVersion not found, using default 'py'."
    $pyArgs = @()
}

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    Invoke-External -Executable "py" -Arguments ($pyArgs + @("-m", "venv", ".venv")) -StepName "Create virtual environment"
}

$pythonExe = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"

Clear-NetworkEnv
Invoke-External -Executable $pythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade",
    "--index-url", "https://pypi.org/simple",
    "pip", "setuptools", "wheel"
) -StepName "Upgrade pip tooling"
Invoke-External -Executable $pythonExe -Arguments @(
    "-m", "pip", "install",
    "--index-url", "https://pypi.org/simple",
    "-r", "requirements.txt"
) -StepName "Install project dependencies"
Invoke-External -Executable $pythonExe -Arguments @(
    "-m", "pip", "install",
    "--index-url", "https://pypi.org/simple",
    "--no-deps",
    "git+https://github.com/RussianInvestments/invest-python.git@0.2.0-beta97"
) -StepName "Install T-Invest SDK"
Invoke-External -Executable $pythonExe -Arguments @("-c", "import tinkoff.invest") -StepName "Validate tinkoff SDK import"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.template" ".env"
}

$env:SETUP_TOKEN = $Token
$env:PYTHONPATH = "."
$env:PYTHONDONTWRITEBYTECODE = "1"
Clear-NetworkEnv

@'
import os
from pathlib import Path

from tinkoff.invest import Client
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX

token = os.environ["SETUP_TOKEN"].strip()
if not token:
    raise ValueError("Token is empty")

with Client(token, target=INVEST_GRPC_API_SANDBOX) as client:
    accounts = client.users.get_accounts().accounts
    if not accounts:
        client.sandbox.open_sandbox_account()
        accounts = client.users.get_accounts().accounts
    account_id = accounts[0].id

env_path = Path(".env")
env_path.write_text(
    f"TOKEN={token}\nACCOUNT_ID={account_id}\nSANDBOX=True\n",
    encoding="utf-8",
)

print(account_id)
'@ | & $pythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "Sandbox account bootstrap failed."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "ACCOUNT_ID written to .env."
Write-Host "Main run command:"
Write-Host '$env:GITHUB_TOKEN="github_pat_YOUR_PAT"'
Write-Host ".\\run_backtest_manual.ps1"

if ($Run) {
    if ([string]::IsNullOrWhiteSpace($env:GITHUB_TOKEN)) {
        throw "For -Run mode, set GITHUB_TOKEN (GitHub PAT) first."
    }
    & ".\\run_backtest_manual.ps1"
}
