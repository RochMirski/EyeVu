<#
.SYNOPSIS
    Launch the EyeVu pupil-detection Streamlit dashboard without needing the
    conda environment activated or conda/streamlit on PATH.

.DESCRIPTION
    Finds the vision_env Python interpreter (override with -EnvPython) and runs
    `python -m streamlit run pupillab/app.py`.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_app.ps1
#>
param(
    [string]$EnvName = "vision_env",
    [string]$EnvPython = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-EnvPython {
    param([string]$Explicit, [string]$Name)
    if ($Explicit -and (Test-Path $Explicit)) { return $Explicit }
    $candidates = @(
        "$env:USERPROFILE\anaconda3\envs\$Name\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$Name\python.exe",
        "$env:LOCALAPPDATA\anaconda3\envs\$Name\python.exe",
        "C:\ProgramData\anaconda3\envs\$Name\python.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

$py = Find-EnvPython -Explicit $EnvPython -Name $EnvName
if (-not $py) {
    Write-Error ("Could not find the '$EnvName' Python. Run setup_dev.ps1 first, " +
                 "or pass -EnvPython <path to python.exe>.")
    return
}

# Verify streamlit is installed in that interpreter before launching.
& $py -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error ("streamlit is not installed in $py. Run setup_dev.ps1 (or: " +
                 "& '$py' -m pip install -r requirements.txt).")
    return
}

Write-Host "Launching dashboard with $py ..." -ForegroundColor Cyan
& $py -m streamlit run (Join-Path $here "pupillab/app.py")
