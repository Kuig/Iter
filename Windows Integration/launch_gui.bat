@echo off
REM Launch the Iter GUI from the project folder:
REM   launch_gui.bat [input file or folder]
REM With a file or folder (the "Open with Iter" context menu entry), the GUI starts with it
REM as the input.
REM Everything in Iter resolves against the current working directory,
REM so the GUI must start from the project root (Presets\, config.json, secrets.json).
setlocal
set "INPUT=%~f1"
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo Error: virtual environment not found at .venv\Scripts\activate.bat
    echo Create it first with: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if "%INPUT%"=="" (
    iter gui
) else (
    iter gui "%INPUT%"
)
