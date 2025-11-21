@echo off
echo Starting Omnilingual ASR Web Demo...
echo.
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python app.py
