@echo off
echo.
echo =====================================================
echo   TrafficLens - Starting Backend
echo =====================================================
echo.
cd /d "%~dp0backend"
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server on http://localhost:8000
echo Open frontend/index.html in your browser
echo Press Ctrl+C to stop
echo.
uvicorn main:app --reload --port 8000
pause
