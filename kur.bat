@echo off
cd /d "%~dp0"
where py >nul 2>nul || (echo Python bulunamadi. python.org dan Python 3.12 kur, "Add python.exe to PATH" kutusunu isaretle. & pause & exit /b 1)
py -3.12 -m venv .venv || py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .
echo.
echo Kurulum bitti. Simdi baslat.bat ile ac.
pause
