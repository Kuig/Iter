@echo off
REM Launch the Iter GUI from the project folder.
REM Everything in Iter resolves against the current working directory,
REM so the GUI must start from the project root (Presets\, config.json, secrets.json).
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
iter gui
