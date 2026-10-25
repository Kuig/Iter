@echo off
REM Launch the Iter GUI from the project folder.
REM Everything in Iter resolves against the current working directory,
REM so the GUI must start from the project root (Presets\, config.json, secrets.json).
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo Error: virtual environment not found at .venv\Scripts\activate.bat
    echo Create it first with: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
iter gui
