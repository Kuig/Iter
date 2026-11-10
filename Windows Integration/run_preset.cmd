@echo off
REM Called by the Explorer context menu entries (see generate_registry_files.ps1):
REM   run_preset.cmd <source file> <preset name>
REM Runs the preset on the file through the project's CLI and writes the result next
REM to it as <name>_processed<ext>. The name is fixed, unlike TextConverter's numbered
REM outputs: the progress file is named after the output, so an interrupted run
REM resumes only if the next one writes to the same file. The CLI runs from the
REM project folder, where Presets\, config.json and secrets.json are. On failure the
REM window stays open so the error can be read.
setlocal
set "SOURCE=%~f1"
set "OUTPUT=%~dpn1_processed%~x1"
set "PRESET=%~2"
cd /d "%~dp0.."
set "EXE=%CD%\.venv\Scripts\iter.exe"

if not exist "%EXE%" (
    echo Error: %EXE% not found. Create the virtual environment and install the project first.
    pause
    exit /b 1
)
"%EXE%" run --input "%SOURCE%" --output "%OUTPUT%" --preset "%PRESET%"
if errorlevel 1 pause
