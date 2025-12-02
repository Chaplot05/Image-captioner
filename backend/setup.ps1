# Quick Start Script for AI Caption Studio Backend
# This script helps you get started quickly

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AI Caption Studio - Backend Setup" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run 'python create_dummy_data.py' to create test data" -ForegroundColor White
Write-Host "2. Run 'python augment_captions.py' to generate mood captions" -ForegroundColor White
Write-Host "3. Run 'python train.py' to train the model (optional, takes time)" -ForegroundColor White
Write-Host "4. Run 'python app.py' to start the API server" -ForegroundColor White
Write-Host ""
