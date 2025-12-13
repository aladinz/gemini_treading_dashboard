# Gemini Trading Dashboard Launcher (PowerShell)
# Right-click and select "Run with PowerShell"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Gemini Trading Dashboard Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to the app directory
Set-Location "C:\Users\aladi\Gemini_Treading_Dashboard"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Start the application
Write-Host ""
Write-Host "Starting dashboard..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Dashboard is running!" -ForegroundColor Green
Write-Host "  Open browser: http://127.0.0.1:8050" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

python app.py

# Keep window open if app crashes
Read-Host "Press Enter to close this window"
