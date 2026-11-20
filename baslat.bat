@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Once kur.bat calistir. & pause & exit /b 1)
.venv\Scripts\python -m texproj ui
pause
