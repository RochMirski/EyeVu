# Copy RITnet ncnn model files to Raspberry Pi
# Usage: .\copy_models_to_pi.ps1 -RemoteHost 192.168.137.133 -RemoteUser roch

param(
    [Parameter(Mandatory=$false)]
    [string]$RemoteHost = "192.168.137.133",
    
    [Parameter(Mandatory=$false)]
    [string]$RemoteUser = "roch",
    
    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/home/roch/EyeVu"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Copying RITnet ncnn Models to Pi" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if model files exist
$paramFile = Join-Path $scriptDir "models\ritnet\ritnet.param"
$binFile = Join-Path $scriptDir "models\ritnet\ritnet.bin"

if (-not (Test-Path $paramFile)) {
    Write-Host "ERROR: ritnet.param not found at $paramFile" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $binFile)) {
    Write-Host "ERROR: ritnet.bin not found at $binFile" -ForegroundColor Red
    exit 1
}

Write-Host "Found model files:" -ForegroundColor Green
Write-Host "  ✓ $paramFile"
Write-Host "  ✓ $binFile"
Write-Host ""

# Copy files to Pi
Write-Host "Copying to $RemoteUser@$RemoteHost`:$RemotePath" -ForegroundColor Yellow
Write-Host ""

try {
    # Use scp to copy files
    $scp_param = "scp -o StrictHostKeyChecking=no -r `"$paramFile`" `"$RemoteUser@$RemoteHost`:$RemotePath/`""
    $scp_bin = "scp -o StrictHostKeyChecking=no -r `"$binFile`" `"$RemoteUser@$RemoteHost`:$RemotePath/`""
    
    Write-Host "Executing: $scp_param"
    & cmd /c $scp_param
    
    Write-Host "Executing: $scp_bin"
    & cmd /c $scp_bin
    
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "✓ Models copied successfully!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next: Run the installation script on the Pi:" -ForegroundColor Cyan
    Write-Host "  ssh $RemoteUser@$RemoteHost" -ForegroundColor White
    Write-Host "  bash install_pi_ncnn.sh" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "ERROR: Failed to copy files" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
