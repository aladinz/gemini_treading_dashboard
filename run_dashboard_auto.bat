@echo off
REM Gemini Trading Dashboard Launcher (Auto-opens browser)

echo ========================================
echo   Gemini Trading Dashboard Launcher
echo ========================================
echo.

REM Change to the app directory
cd /d "C:\Users\aladi\Gemini_Treading_Dashboard"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Start the application in background
echo.
echo Starting dashboard...
echo.
echo ========================================
echo   Dashboard is running!
echo   Browser will open automatically...
echo ========================================
echo.

REM Wait 3 seconds and open browser
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8050"

REM Run the app
python app.py

REM Keep window open if app crashes
pause
