<#
.SYNOPSIS
    Set up the EyeVu pupil-detection dev environment and (optionally) launch the
    Streamlit dashboard.

.DESCRIPTION
    Creates / updates the `vision_env` conda environment, installs the core deps
    (requirements.txt) and - unless -SkipML - the RITnet ML extras
    (requirements-ml.txt), then prints how to obtain the RITnet weights.

    Finds conda automatically even when it is not on PATH (i.e. you do not need
    the "Anaconda PowerShell Prompt"); pass -Conda to point at a specific
    conda.exe.

.EXAMPLE
    ./setup_dev.ps1            # core + ML deps, then prints next steps
    ./setup_dev.ps1 -SkipML    # core deps only (no torch)
    ./setup_dev.ps1 -Launch    # set up, then `streamlit run pupillab/app.py`
#>
param(
    [string]$EnvName = "vision_env",
    [string]$Conda = "",
    [switch]$SkipML,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Locate conda: explicit -Conda, then PATH, then common install locations.
function Find-Conda {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path $Explicit)) { return $Explicit }
    $onPath = Get-Command conda -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\Continuum\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

$CondaExe = Find-Conda -Explicit $Conda
if (-not $CondaExe) {
    Write-Error ("conda not found. Pass -Conda <path to conda.exe>, or install " +
                 "Miniconda. Looked on PATH and under anaconda3/miniconda3.")
    return
}
Write-Host "Using conda: $CondaExe" -ForegroundColor DarkGray

# Create the env if it does not already exist.
$envExists = (& $CondaExe env list) -match "(^|\\|/)$EnvName\s"
if (-not $envExists) {
    Write-Host "Creating conda env '$EnvName' (python 3.11)..." -ForegroundColor Cyan
    & $CondaExe create -y -n $EnvName python=3.11
} else {
    Write-Host "Conda env '$EnvName' already exists - reusing." -ForegroundColor Cyan
}

Write-Host "Installing core requirements..." -ForegroundColor Cyan
& $CondaExe run -n $EnvName python -m pip install -r (Join-Path $here "requirements.txt")

if (-not $SkipML) {
    Write-Host "Installing ML extras (torch, torchvision) for the RITnet module..." -ForegroundColor Cyan
    & $CondaExe run -n $EnvName python -m pip install -r (Join-Path $here "requirements-ml.txt")
}

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "  1. (RITnet) download best_model.pkl into models/ritnet/  - see models/ritnet/README.md"
Write-Host "  2. Launch the dashboard:  conda run -n $EnvName streamlit run pupillab/app.py"
Write-Host "  3. Or run headless:       conda run -n $EnvName python pupillab/run_batch.py"

if ($Launch) {
    Write-Host ""
    Write-Host "Launching Streamlit dashboard..." -ForegroundColor Cyan
    & $CondaExe run -n $EnvName streamlit run (Join-Path $here "pupillab/app.py")
}
