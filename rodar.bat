@echo off
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    py -3.12 -m venv "%~dp0.venv"
    if errorlevel 1 (
        python -m venv "%~dp0.venv"
    )
)

if not exist "%PYTHON%" (
    echo ERRO: nao foi possivel criar o ambiente virtual Python.
    pause
    exit /b 1
)

"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r "%~dp0requirements.txt"
"%PYTHON%" -m streamlit run "%~dp0streamlit_app.py"
pause
