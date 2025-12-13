@echo off
REM Gemini Trading Dashboard Launcher
REM Double-click this file to start the dashboard

echo ========================================
echo   Gemini Trading Dashboard Launcher
echo ========================================
echo.

REM Change to the app directory
cd /d "C:\Users\aladi\Gemini_Treading_Dashboard"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Start the application
echo.
echo Starting dashboard...
echo.
echo ========================================
echo   Dashboard is running!
echo   Open browser: http://127.0.0.1:8050
echo ========================================
echo.
echo Press Ctrl+C to stop the dashboard
echo.

python app.py

REM Keep window open if app crashes
pause
